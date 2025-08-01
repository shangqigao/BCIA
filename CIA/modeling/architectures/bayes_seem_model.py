# --------------------------------------------------------
# SEEM -- Segment Everything Everywhere All at Once
# Licensed under The Apache License 2.0 [see LICENSE for details]
# Written by Xueyan Zou (xueyan@cs.wisc.edu)
# --------------------------------------------------------

import os
import random
from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from kornia.contrib import distance_transform
from tensorboardX import SummaryWriter

from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.data import MetadataCatalog

from .build import register_model

from ..utils import configurable, get_class_names, get_iou, Spatial_ImageList
from ..vision.decomposition import build_decomposition, build_visualizer, BayesDec
from ..vision.backbone import build_backbone, Backbone
from ..body import build_xdecoder_head
from ..modules import sem_seg_postprocess, SetCriterion, HungarianMatcher, bbox_postprocess
from ..language import build_language_encoder
from ..language.loss import vl_similarity
from utilities.prompt_engineering import prompt_engineering
from utilities.constants import COCO_PANOPTIC_CLASSES, BIOMED_CLASSES


class BayesianSEEM(nn.Module):

    @configurable
    def __init__(
        self,
        *,
        decomposition: BayesDec,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        criterion: nn.Module,
        visualizer: nn.Module,
        writer: SummaryWriter,
        losses: dict,
        num_queries: int,
        object_mask_threshold: float,
        overlap_threshold: float,
        metadata,
        task_switch: dict,
        phrase_prob: float,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        # inference
        semantic_on: bool,
        panoptic_on: bool,
        instance_on: bool,
        test_topk_per_image: int,
        train_dataset_name: str,
        interactive_mode: str,
        interactive_iter: str,
        dilation_kernel: torch.Tensor,
        train_max_iter: int,
        binary_classes: bool,
        standard_text_for_eval: bool,
        vis_every_n_steps: int
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            criterion: a module that defines the loss
            num_queries: int, number of queries
            object_mask_threshold: float, threshold to filter query based on classification score
                for panoptic segmentation inference
            overlap_threshold: overlap threshold used in general inference for panoptic segmentation
            metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                segmentation inference
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
            sem_seg_postprocess_before_inference: whether to resize the prediction back
                to original input size before semantic segmentation inference or after.
                For high-resolution dataset like Mapillary, resizing predictions before
                inference will cause OOM error.
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            semantic_on: bool, whether to output semantic segmentation prediction
            instance_on: bool, whether to output instance segmentation prediction
            panoptic_on: bool, whether to output panoptic segmentation prediction
            test_topk_per_image: int, instance segmentation parameter, keep topk instances per image
        """
        super().__init__()
        self.decomposition = decomposition
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        self.criterion = criterion
        self.visualizer = visualizer
        self.writer = writer
        self.losses = losses
        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold
        self.metadata = metadata
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        # additional args
        self.semantic_on = semantic_on
        self.instance_on = instance_on
        self.panoptic_on = panoptic_on

        # caption argument
        self.task_switch = task_switch
        self.phrase_prob = phrase_prob
        self.train_max_iter = train_max_iter

        self.test_topk_per_image = test_topk_per_image
        self.train_class_names = get_class_names(train_dataset_name)
        if binary_classes:
            self.train_class_names = ['target', 'background']
        self.interactive_mode = interactive_mode
        self.interactive_iter = interactive_iter

        if not self.semantic_on:
            assert self.sem_seg_postprocess_before_inference

        self.register_buffer("dilation_kernel", dilation_kernel)

        self.standard_text_for_eval = standard_text_for_eval

        self.vis_every_n_steps = vis_every_n_steps
        self.vis_counter = 0

    @classmethod
    def from_config(cls, cfg):
        decomp_cfg = cfg['MODEL']['DECOMPOSITION']
        enc_cfg = cfg['MODEL']['ENCODER']
        dec_cfg = cfg['MODEL']['DECODER']

        # Loss parameters:
        deep_supervision = dec_cfg['DEEP_SUPERVISION']
        no_object_weight = dec_cfg['NO_OBJECT_WEIGHT']

        # loss weights
        loss_weights = {'decomposition': {'bayes': decomp_cfg['BAYES_LOSS_WEIGHT']},
                        'mask': {'ce': dec_cfg['CLASS_WEIGHT'], 'dice': dec_cfg['DICE_WEIGHT'], 'bce': dec_cfg['MASK_WEIGHT']},
                        'bbox': {'l1': dec_cfg['BBOX_WEIGHT'], 'giou': dec_cfg['GIOU_WEIGHT']},
                        'spatial': {'ce': dec_cfg['SCLASS_WEIGHT'], 'dice': dec_cfg['SDICE_WEIGHT'], 'bce': dec_cfg['SMASK_WEIGHT']},
                        'grounding': {'ce': dec_cfg['GCLASS_WEIGHT'], 'dice': dec_cfg['GDICE_WEIGHT'], 'bce': dec_cfg['GMASK_WEIGHT']},
                        'openimage': {'ce': dec_cfg['OCLASS_WEIGHT'], 'dice': dec_cfg['ODICE_WEIGHT'], 'bce': dec_cfg['OMASK_WEIGHT']}}

        openimage_switch = {'grounding': dec_cfg['OPENIMAGE']['GROUNDING'].get('ENABLED', False),
                            'mask': dec_cfg['OPENIMAGE'].get('ENABLED', False)}

        task_switch = {'decomposition': decomp_cfg['BAYES'].get('ENABLED', True),
                       'bbox': dec_cfg.get('DETECTION', False),
                       'mask': dec_cfg['MASK'].get('ENABLED', True),
                       'spatial': dec_cfg['SPATIAL'].get('ENABLED', False),
                       'grounding': dec_cfg['GROUNDING'].get('ENABLED', False),
                       'openimage': openimage_switch}

        top_x_layers = {'mask': dec_cfg.get('TOP_MASK_LAYERS', 10),
                        'grounding': dec_cfg.get('TOP_GROUNDING_LAYERS', 10),
                        'openimage': dec_cfg.get('TOP_OPENIMAGE_LAYERS', 10),
                        'spatial': dec_cfg.get('TOP_SPATIAL_LAYERS', 10)}

        spatial_cost = {"class_weight": dec_cfg['COST_SPATIAL']['CLASS_WEIGHT'],
                        "mask_weight": dec_cfg['COST_SPATIAL']['MASK_WEIGHT'],
                        "dice_weight": dec_cfg['COST_SPATIAL']['DICE_WEIGHT']}

        extra = {'task_switch': task_switch}
        decomposition = build_decomposition(cfg)
        backbone = build_backbone(cfg)
        lang_encoder = build_language_encoder(cfg)        
        sem_seg_head = build_xdecoder_head(cfg, backbone.output_shape(), lang_encoder, extra=extra)

        # building criterion
        matcher = HungarianMatcher(
            cost_class=loss_weights['mask']['ce'],
            cost_mask=loss_weights['mask']['bce'],
            cost_dice=loss_weights['mask']['dice'],
            num_points=dec_cfg['TRAIN_NUM_POINTS'],
            spatial_cost=spatial_cost,
        )

        # init weight dict and criterion loss functions.
        losses = {'seg': [], 'openimage': []}
        if task_switch['decomposition']:
            losses['seg'] += ["decomposition"]
        if task_switch['mask']:
            losses['seg'] += ["labels", "masks"]
        if task_switch['spatial']:
            losses['seg'] += ["spatials"]
        if task_switch['grounding']:
            losses['seg'] += ["groundings"]
        if task_switch['openimage']:
            losses['openimage'] += ["labels_openimage", "masks"]
        if task_switch['openimage']['grounding']:
            losses['openimage'] += ["groundings"]

        weight_dict = {}
        for key, turn_on in task_switch.items():
            if turn_on:
                if isinstance(loss_weights[key], dict):
                    # HACK it should support bbox in the future
                    for key_, weight in loss_weights[key].items():
                        weight_dict["loss_{}_{}_0".format(key, key_)] = weight # NOTE: hard code for segmentation that has multiple loss
                else:
                    weight_dict["loss_{}_0".format(key)] = loss_weights[key]

        # generate full weight dict and remove not computed layers. 
        if deep_supervision:
            dec_layers = dec_cfg['DEC_LAYERS']
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                for k, v in weight_dict.items():
                    if "decomposition" in k: continue
                    if (i+1) > (top_x_layers[k.split('_')[1]] - 1): continue
                    aux_weight_dict.update({k.replace('_0', f"_{i+1}"): v})
            weight_dict.update(aux_weight_dict)

        grd_weight = {'text': dec_cfg['GROUNDING']['TEXT_WEIGHT'], 'class': dec_cfg['GROUNDING']['CLASS_WEIGHT']}
        # generate critenrion for loss function.
        criterion = SetCriterion(
            sem_seg_head.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            top_x_layers=top_x_layers,
            eos_coef=no_object_weight,
            losses=[],
            num_points=dec_cfg['TRAIN_NUM_POINTS'],
            oversample_ratio=dec_cfg['OVERSAMPLE_RATIO'],
            importance_sample_ratio=dec_cfg['IMPORTANCE_SAMPLE_RATIO'],
            grounding_weight=grd_weight,
        )

        # visualizer and writer
        visualizer = build_visualizer()
        writer = SummaryWriter(log_dir=os.path.join(cfg['SAVE_DIR'], "summary"))

        # extra logistic
        train_dataset_name = cfg['DATASETS']['TRAIN'][0] # HACK for only one training set.
        train_max_iter = dec_cfg['SPATIAL'].get('MAX_ITER', 3)
        phrase_prob = dec_cfg['CAPTION'].get('PHRASE_PROB', 0.5)
        interactive_mode = cfg['STROKE_SAMPLER']['EVAL']['MODE']
        interactive_iter = cfg['STROKE_SAMPLER']['EVAL']['MAX_ITER']

        dilation = 3
        dilation_kernel = torch.ones((1, 1, dilation, dilation), device=torch.cuda.current_device())

        return {
            "decomposition": decomposition,
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion": criterion,
            "visualizer": visualizer,
            "writer": writer,
            "losses": losses,
            "num_queries": dec_cfg['NUM_OBJECT_QUERIES'],
            "object_mask_threshold": dec_cfg['TEST']['OBJECT_MASK_THRESHOLD'],
            "overlap_threshold": dec_cfg['TEST']['OVERLAP_THRESHOLD'],
            "metadata": MetadataCatalog.get(cfg['DATASETS']['TRAIN'][0]),
            "size_divisibility": dec_cfg['SIZE_DIVISIBILITY'],
            "sem_seg_postprocess_before_inference": (
                dec_cfg['TEST']['SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE']
                or dec_cfg['TEST']['PANOPTIC_ON']
                or dec_cfg['TEST']['INSTANCE_ON']
            ),
            "pixel_mean": cfg['INPUT']['PIXEL_MEAN'],
            "pixel_std": cfg['INPUT']['PIXEL_STD'],
            "task_switch": task_switch,
            "phrase_prob": phrase_prob,
            # inference
            "semantic_on": dec_cfg['TEST']['SEMANTIC_ON'],
            "instance_on": dec_cfg['TEST']['INSTANCE_ON'],
            "panoptic_on": dec_cfg['TEST']['PANOPTIC_ON'],
            "test_topk_per_image": cfg['TEST']['DETECTIONS_PER_IMAGE'],
            "train_dataset_name": train_dataset_name,
            "interactive_mode": interactive_mode,
            "interactive_iter": interactive_iter,
            "dilation_kernel": dilation_kernel,
            "train_max_iter": train_max_iter,
            "binary_classes": enc_cfg['BINARY_CLASSES'],
            "standard_text_for_eval": cfg['STANDARD_TEXT_FOR_EVAL'],
            "vis_every_n_steps": decomp_cfg['VIS_EVERY_N_STEPS'],
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs, mode='default', **kwargs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
        """
        if self.training:
            losses = {}
            if self.task_switch['mask'] or self.task_switch['grounding'] or self.task_switch['spatial']:
                losses_seg = self.forward_seg(batched_inputs)
                losses.update(losses_seg)
            if self.task_switch['openimage'] and self.task_switch['openimage']['mask']:
                losses_openimage = self.forward_openimage(batched_inputs['openimage'])
                losses_openimage = {key.replace('mask', 'openimage'):value for key, value in losses_openimage.items()}
                losses_openimage = {key.replace('grounding', 'grounding_openimage'):value for key, value in losses_openimage.items()}
                losses.update(losses_openimage)
            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else: # remove this loss if not specified in `weight_dict`
                    losses.pop(k)
            return losses
        else:
            if mode == 'interactive':
                return self.evaluate_interactive(batched_inputs)
            elif mode == 'interactive_grounding':
                return self.evaluate_interactive_grounding(batched_inputs)
            elif mode == 'grounding_spatial':
                return self.evaluate_grounding_sptial(batched_inputs, mode)
            elif mode in ['grounding_phrasecut', 'grounding_refcoco']:
                return self.evaluate_grounding(batched_inputs, mode)
            else:
                return self.evaluate(batched_inputs)

        
    def forward_seg(self, batched_inputs):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)
        self.sem_seg_head.predictor.lang_encoder.get_text_embeddings(self.train_class_names, is_eval=False)

        extra = {}
        # mask classification target
        if "instances" in batched_inputs[0]:
            # input bounding box is checked to be correct.
            targets = self.prepare_targets(batched_inputs, images)

            if self.task_switch['grounding']:
                grounding_tokens = [x['grounding_query_embs'] for x in targets] # need to pad for more than one grounding token
                grounding_tokens = nn.utils.rnn.pad_sequence(grounding_tokens, padding_value=-1)
                non_zero_query_mask = (grounding_tokens.sum(dim=-1) == -grounding_tokens.shape[-1])
                grounding_tokens[non_zero_query_mask] = 0

                extra['grounding_tokens'] = grounding_tokens
                extra['grounding_nonzero_mask'] = non_zero_query_mask.t()

            if self.task_switch['spatial']:
                pos_masks = [x['spatial_query']['rand_shape'].to(self.device) for x in batched_inputs]
                neg_masks = [(x['spatial_query']['rand_shape'].to(self.device) & False) for x in batched_inputs]
                fp_masks = nn.utils.rnn.pad_sequence([(x['spatial_query']['rand_shape'].to(self.device) & False) for x in batched_inputs], padding_value=False, batch_first=True)
                extra.update({'spatial_query_pos_mask': pos_masks, 'spatial_query_neg_mask': neg_masks, 'false_positive_mask': fp_masks})

        # Bayesian image decomposition
        decomp_outputs = self.decomposition(images.tensor)
        self.vis_counter += 1
        if self.vis_counter % self.vis_every_n_steps == 0:
            self.visualizer(images.tensor, decomp_outputs['visualize'], self.vis_counter, self.writer)

        features = self.backbone(decomp_outputs['pred'])
        mask_features, _, multi_scale_features = self.sem_seg_head.pixel_decoder.forward_features(features)

        # forward spatial only without gradient
        if self.task_switch['spatial']:
            with torch.no_grad():
                # generate random integeter between [0,3]
                rand_iter_num = random.randint(0, self.train_max_iter)
                for i in range(rand_iter_num):
                    outputs = self.sem_seg_head.predictor(multi_scale_features, mask_features, extra=extra, task='spatial')
                    extra.update(outputs)
                    extra.update(self.prepare_next_spaital_mask(extra, batched_inputs))

        outputs = self.sem_seg_head.predictor(multi_scale_features, mask_features, extra=extra, task='seg')

        extra = {'lang_logit': self.sem_seg_head.predictor.lang_encoder.logit_scale,
                 'class_embeddings': getattr(self.sem_seg_head.predictor.lang_encoder, '{}_text_embeddings'.format('default')),
                 'false_positive_mask': extra['false_positive_mask'],
                 'decomposition': decomp_outputs}
        # bipartite matching-based loss
        self.criterion.losses = self.losses['seg'] # seg criterion losses

        if self.task_switch['mask']:
            losses = self.criterion(outputs, targets, extra)
        else:
            losses = self.criterion.forward_vlp(outputs, targets, extra)

        del decomp_outputs
        del outputs
        return losses

    def evaluate(self, batched_inputs):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        
        images = ImageList.from_tensors(images, self.size_divisibility)
        img_bs = images.tensor.shape[0]

        targets = targets_grounding = queries_grounding = None
        decomp_outputs = self.decomposition(images.tensor)
        features = self.backbone(decomp_outputs['pred'])
        outputs = self.sem_seg_head(features, target_queries=queries_grounding)

        mask_cls_results = outputs["pred_logits"]
        mask_pred_results = outputs["pred_masks"]
        box_pred_results = outputs["pred_boxes"] if self.task_switch['bbox'] else [None for i in range(len(mask_pred_results))]

        # upsample masks
        mask_pred_results = F.interpolate(
            mask_pred_results,
            size=(images.tensor.shape[-2], images.tensor.shape[-1]),
            mode="bilinear",
            align_corners=False,
        )

        input_size = mask_pred_results.shape[-2:]
        del decomp_outputs
        del outputs

        processed_results = []
        for mask_cls_result, mask_pred_result, box_pred_result, input_per_image, image_size in zip(
            mask_cls_results, mask_pred_results, box_pred_results, batched_inputs, images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            processed_results.append({})

            if self.sem_seg_postprocess_before_inference:
                mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                    mask_pred_result, image_size, height, width
                )
                mask_cls_result = mask_cls_result.to(mask_pred_result)

            # semantic segmentation inference
            if self.semantic_on:
                r = retry_if_cuda_oom(self.semantic_inference)(mask_cls_result, mask_pred_result)
                if not self.sem_seg_postprocess_before_inference:
                    r = retry_if_cuda_oom(sem_seg_postprocess)(r, image_size, height, width)
                processed_results[-1]["sem_seg"] = r

            # panoptic segmentation inference
            if self.panoptic_on:
                panoptic_r = retry_if_cuda_oom(self.panoptic_inference)(mask_cls_result, mask_pred_result)
                processed_results[-1]["panoptic_seg"] = panoptic_r
            
            # instance segmentation inference
            if self.instance_on:
                if self.task_switch['bbox']:
                    box_pred_result = bbox_postprocess(box_pred_result, input_size, image_size, height, width)
                instance_r = retry_if_cuda_oom(self.instance_inference)(mask_cls_result, mask_pred_result, box_pred_result)
                processed_results[-1]["instances"] = instance_r

        return processed_results

    def evaluate_interactive(self, batched_inputs):
        assert self.task_switch['spatial']
        assert 'spatial_query' in batched_inputs[0]
        assert len(batched_inputs) == 1, "only support batch size equal to 1"

        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)
        img_bs = images.tensor.shape[0]

        targets = targets_grounding = queries_grounding = None
        extra = {}

        decomp_outputs = self.decomposition(images.tensor)
        features = self.backbone(decomp_outputs['pred'])
        mask_features, transformer_encoder_features, multi_scale_features = self.sem_seg_head.pixel_decoder.forward_features(features)

        image_sizes = [x["image"].shape[-2:] for x in batched_inputs]

        all_batch_shape_iou = []
        pred_smask_pointer = None
        prev_smask_pointer = None
        pred_smask_all = None

        # visualization code
        # v_pred_mask = []
        # v_pos_mask = []
        # v_neg_mask = []
        # v_gt_mask = batched_inputs[0]['spatial_query']['gt_masks'][0]
        query_index = self.sem_seg_head.predictor.query_index
        if self.interactive_mode in ['best', 'best_random']:
            pos_masks = [x['spatial_query']['rand_shape'].to(self.device)[:,0] for x in batched_inputs]
            pos_masks = ImageList.from_tensors(pos_masks, self.size_divisibility).tensor.unbind(0)

            neg_masks = [(x['spatial_query']['rand_shape'].to(self.device) & False)[:,0] for x in batched_inputs]
    
            neg_masks = ImageList.from_tensors(neg_masks, self.size_divisibility).tensor.unbind(0)
            extra.update({'spatial_query_pos_mask': pos_masks, 'spatial_query_neg_mask': neg_masks})
        elif self.interactive_mode == 'random':
            assert False, "interactive mode not correctly implemented"
            pos_masks = (batched_inputs[0]['spatial_query']['rand_shape'].to(self.device)==1).unbind(0)
            pos_masks = ImageList.from_tensors(pos_masks, self.size_divisibility).tensor

            neg_masks = (batched_inputs[0]['spatial_query']['rand_shape'].to(self.device)==-1).unbind(0)
            neg_masks = ImageList.from_tensors(neg_masks, self.size_divisibility).tensor
            extra.update({'spatial_query_pos_mask': pos_masks[:,0:1].unbind(), 'spatial_query_neg_mask': neg_masks[:,0:1].unbind()})
        else:
            assert False, "invalid interactive mode"

        for i in range(self.interactive_iter):
            # v_pos_mask += [extra['spatial_query_pos_mask'][0][0][:image_sizes[0][0],:image_sizes[0][1]].float().cpu().numpy()]
            # v_neg_mask += [extra['spatial_query_neg_mask'][0][0][:image_sizes[0][0],:image_sizes[0][1]].float().cpu().numpy()]
            outputs = self.sem_seg_head.predictor(multi_scale_features, mask_features, target_queries=queries_grounding, extra=extra, task='spatial')
            extra.update(outputs)
            pred_smask = F.interpolate(outputs['prev_mask'], images.tensor.shape[-2:], mode='bilinear')
            # v_pred_mask += [(pred_smask[0,0][:image_sizes[0][0],:image_sizes[0][1]].sigmoid() > 0.5).float().cpu().numpy()]

            s = image_sizes[0]
            b = batched_inputs[0]
            pred_smask_all = F.interpolate(pred_smask[:,:,:s[0],:s[1]], (b['height'], b['width']), mode='bilinear')[0].sigmoid() > 0.5
            gt_smask = b['gt_masks_orisize']
            ious = get_iou(gt_smask, pred_smask_all)
            all_batch_shape_iou += [ious]
            if (ious > 0.9).sum() == len(ious):
                all_batch_shape_iou += [ious for j in range(self.interactive_iter-i-1)]
                break
            if self.interactive_mode in ['best', 'best_random']:
                extra.update(self.prepare_next_spaital_mask(extra, batched_inputs, mode=self.interactive_mode))
            elif self.interactive_mode == 'random':
                extra.update({'spatial_query_pos_mask': pos_masks[:,i+1:i+2].unbind(), 'spatial_query_neg_mask': neg_masks[:,i+1:i+2].unbind()})
            else:
                assert False, "invalid interactive mode"
        all_batch_shape_iou = torch.stack(all_batch_shape_iou)
        processed_results = [{"mask_iou": all_batch_shape_iou[:,i]} for i in range(len(all_batch_shape_iou[0]))]

        return processed_results

    def evaluate_interactive_single(self, batched_inputs, extra={}):
        assert self.task_switch['spatial']
        assert 'spatial_query' in batched_inputs[0]
        assert len(batched_inputs) == 1, "only support batch size equal to 1"

        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)
        img_bs = images.tensor.shape[0]

        targets = targets_grounding = queries_grounding = None

        decomp_outputs = self.decomposition(images.tensor)
        features = self.backbone(decomp_outputs['pred'])
        mask_features, transformer_encoder_features, multi_scale_features = self.sem_seg_head.pixel_decoder.forward_features(features)

        image_sizes = [x["image"].shape[-2:] for x in batched_inputs]
        nm = len(batched_inputs[0]['spatial_query']['rand_shape'])
        multi_scale_features = [m.repeat(nm,1,1,1) for m in multi_scale_features]
        mask_features = mask_features.repeat(nm,1,1,1)

        outputs = self.sem_seg_head.predictor(multi_scale_features, mask_features, target_queries=queries_grounding, extra=extra, task='spatial')
        pred_smask = F.interpolate(outputs['prev_mask'], images.tensor.shape[-2:], mode='bicubic')

        s = image_sizes[0]
        b = batched_inputs[0]
        pred_smask_ori = F.interpolate(pred_smask[:,:,:s[0],:s[1]], (b['height'], b['width']), mode='bicubic')[:,0].sigmoid() > 0.5
        pred_smask_batch = pred_smask[:,:,:s[0],:s[1]].sigmoid() > 0.5
        ious = []
        if 'gt_masks_orisize' in b:
            gt_smask = b['gt_masks_orisize'].to(pred_smask_ori.device)
            ious = get_iou(gt_smask, pred_smask_ori)
        processed_results = [{"mask_iou": ious, 'pred_mask_ori': pred_smask_ori, 'pred_mask_batch': pred_smask_batch}]
        return processed_results

    def evaluate_interactive_grounding(self, batched_inputs):
        assert self.task_switch['spatial']
        assert 'spatial_query' in batched_inputs[0]
        assert len(batched_inputs) == 1, "only support batch size equal to 1"

        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)
        img_bs = images.tensor.shape[0]

        targets = targets_grounding = queries_grounding = None
        extra = {}

        decomp_outputs = self.decomposition(images.tensor)
        features = self.backbone(decomp_outputs['pred'])
        mask_features, transformer_encoder_features, multi_scale_features = self.sem_seg_head.pixel_decoder.forward_features(features)

        image_sizes = [x["image"].shape[-2:] for x in batched_inputs]
        nm = len(batched_inputs[0]['spatial_query']['rand_shape'])
        multi_scale_features = [m.repeat(nm,1,1,1) for m in multi_scale_features]
        mask_features = mask_features.repeat(nm,1,1,1)

        all_batch_shape_iou = []
        pred_smask_pointer = None
        prev_smask_pointer = None
        pred_smask_all = None

        # visualization code
        # v_pred_mask = []
        # v_pos_mask = []
        # v_neg_mask = []
        # v_gt_mask = batched_inputs[0]['spatial_query']['gt_masks'][0]
        query_index = self.sem_seg_head.predictor.query_index
        if self.interactive_mode in ['best', 'best_random']:
            pos_masks = (batched_inputs[0]['spatial_query']['rand_shape'].to(self.device)).unbind(0)
            pos_masks = ImageList.from_tensors(pos_masks, self.size_divisibility).tensor.unbind(0)

            neg_masks = (batched_inputs[0]['spatial_query']['rand_shape'].to(self.device) & False).unbind(0)
            neg_masks = ImageList.from_tensors(neg_masks, self.size_divisibility).tensor.unbind(0)
            extra.update({'spatial_query_pos_mask': pos_masks, 'spatial_query_neg_mask': neg_masks})
        elif self.interactive_mode == 'random':
            pos_masks = (batched_inputs[0]['spatial_query']['rand_shape'].to(self.device)==1).unbind(0)
            pos_masks = ImageList.from_tensors(pos_masks, self.size_divisibility).tensor

            neg_masks = (batched_inputs[0]['spatial_query']['rand_shape'].to(self.device)==-1).unbind(0)
            neg_masks = ImageList.from_tensors(neg_masks, self.size_divisibility).tensor
            extra.update({'spatial_query_pos_mask': pos_masks[:,0:1].unbind(), 'spatial_query_neg_mask': neg_masks[:,0:1].unbind()})
        else:
            assert False, "invalid interactive mode"

        grd_texts = batched_inputs[0]['classes']
        gtext = self.sem_seg_head.predictor.lang_encoder.get_text_token_embeddings(grd_texts, name='grounding', token=False, norm=False)
        token_emb = gtext['token_emb']
        tokens = gtext['tokens']
        query_emb = nn.utils.rnn.pad_sequence([_token_emb[_tokens.bool()] for _token_emb, _tokens in zip(token_emb, tokens['attention_mask'])], padding_value=-1)
        non_zero_query_mask = (query_emb.sum(dim=-1) < 0)

        extra['grounding_tokens'] = query_emb
        extra['grounding_nonzero_mask'] = non_zero_query_mask.t()

        for i in range(self.interactive_iter):
            # v_pos_mask += [extra['spatial_query_pos_mask'][0][0][:image_sizes[0][0],:image_sizes[0][1]].float().cpu().numpy()]
            # v_neg_mask += [extra['spatial_query_neg_mask'][0][0][:image_sizes[0][0],:image_sizes[0][1]].float().cpu().numpy()]
            outputs = self.sem_seg_head.predictor(multi_scale_features, mask_features, target_queries=queries_grounding, extra=extra, task='spatial')
            extra.update(outputs)
            pred_smask = F.interpolate(outputs['prev_mask'], images.tensor.shape[-2:], mode='bilinear')
            # v_pred_mask += [(pred_smask[0,0][:image_sizes[0][0],:image_sizes[0][1]].sigmoid() > 0.5).float().cpu().numpy()]

            s = image_sizes[0]
            b = batched_inputs[0]
            pred_smask_all = F.interpolate(pred_smask[:,:,:s[0],:s[1]], (b['height'], b['width']), mode='bilinear')[:,0].sigmoid() > 0.5
            gt_smask = b['gt_masks_orisize']
            ious = get_iou(gt_smask, pred_smask_all)
            all_batch_shape_iou += [ious]
            if (ious > 0.9).sum() == len(ious):
                all_batch_shape_iou += [ious for j in range(self.interactive_iter-i-1)]
                break
            if self.interactive_mode in ['best', 'best_random']:
                extra.update(self.prepare_next_spaital_mask(extra, batched_inputs, mode=self.interactive_mode))
            elif self.interactive_mode == 'random':
                extra.update({'spatial_query_pos_mask': pos_masks[:,i+1:i+2].unbind(), 'spatial_query_neg_mask': neg_masks[:,i+1:i+2].unbind()})
            else:
                assert False, "invalid interactive mode"
        all_batch_shape_iou = torch.stack(all_batch_shape_iou)
        processed_results = [{"mask_iou": all_batch_shape_iou[:,i]} for i in range(len(all_batch_shape_iou[0]))]

        # visualization
        # VL.step()
        # import cv2
        # v_masks = []
        # v_pos_masks = []
        # v_neg_masks = []
        # txt = []

        # img = batched_inputs[0]['image'].permute(1,2,0).cpu().numpy()
        # mask_img = VL.overlay_single_mask_to_image(img[:,:,::-1], v_gt_mask.cpu().float().numpy())
        # acc_pos_mask = np.zeros(v_pos_mask[0].shape)
        # acc_neg_mask = np.zeros(v_neg_mask[0].shape)
        # for x,y,z,iou in zip(v_pos_mask, v_neg_mask, v_pred_mask, all_batch_shape_iou):
        #     # dilate x,y
        #     x = cv2.dilate(x, np.ones((5,5), np.uint8), iterations=3)
        #     y = cv2.dilate(y, np.ones((5,5), np.uint8), iterations=3)
        #     acc_pos_mask += x
        #     acc_neg_mask += y

        #     v_masks += [z]
        #     v_pos_masks += [acc_pos_mask.clip(0,1)]
        #     v_neg_masks += [acc_neg_mask.clip(0,1)]
        #     txt += ["pred_{}".format(str(iou[0].item())[0:5])]

        # VL.add_image(img[:,:,::-1])
        # VL.insert(mask_img, "gt_mask")
        # VL.overlay_obj_mask_to_image_withposneg(img[:,:,::-1], v_masks, v_pos_masks, v_neg_masks, txt, max_len=20)
        return processed_results

    def evaluate_referring_image(self, batched_inputs, extra={}):
        assert self.task_switch['spatial']
        assert len(batched_inputs) == 1, "only support batch size equal to 1"
        assert self.interactive_mode == 'best'

        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)
        img_bs = images.tensor.shape[0]

        targets = targets_grounding = queries_grounding = None
        decomp_outputs = self.decomposition(images.tensor)
        features = self.backbone(decomp_outputs['pred'])
        mask_features, transformer_encoder_features, multi_scale_features = self.sem_seg_head.pixel_decoder.forward_features(features)

        if 'spatial_query' in batched_inputs[0]:
            image_sizes = [x["image"].shape[-2:] for x in batched_inputs]
            nm = len(batched_inputs[0]['spatial_query']['rand_shape'])
            multi_scale_features = [m.repeat(nm,1,1,1) for m in multi_scale_features]
            mask_features = mask_features.repeat(nm,1,1,1)

            query_index = self.sem_seg_head.predictor.query_index
            pos_masks = (batched_inputs[0]['spatial_query']['rand_shape'].to(self.device)).unbind(0)
            pos_masks = ImageList.from_tensors(pos_masks, self.size_divisibility).tensor.unbind(0)

            neg_masks = (batched_inputs[0]['spatial_query']['rand_shape'].to(self.device) & False).unbind(0)
            neg_masks = ImageList.from_tensors(neg_masks, self.size_divisibility).tensor.unbind(0)
            extra.update({'spatial_query_pos_mask': pos_masks, 'spatial_query_neg_mask': neg_masks})

        outputs = self.sem_seg_head.predictor(multi_scale_features, mask_features, target_queries=queries_grounding, extra=extra, task='refimg')
        return outputs, images.tensor.shape

    def evaluate_grounding(self, batched_inputs, mode):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)
        assert len(images.tensor) == 1, "grounding evaluation only support single batch size now"

        extra = {}
        # mask_pred_results = []
        # for idx, batch_per_image in enumerate(batched_inputs):
        #     grd_texts = batch_per_image['groundings']['texts']
        #     grd_masks = []
        #     for anno_text in grd_texts:
        #         gtext = self.sem_seg_head.predictor.lang_encoder.get_text_token_embeddings([anno_text[0]], name='grounding', token=False, norm=False)
        #         token_emb = gtext['token_emb']
        #         tokens = gtext['tokens']
            
        #         grd_emb = token_emb[0][tokens['attention_mask'].bool()[0]]
        #         extra['grounding_tokens'] = grd_emb[:,None]

        #         assert len(images.tensor) == 1, "grounding evaluation only support single batch size now"
        #         features = self.backbone(images.tensor)
        #         outputs = self.sem_seg_head(features, extra=extra, task='grounding_eval')
                
        #         pred_gmasks = outputs['pred_masks'][idx,self.num_queries:2*self.num_queries-1]
        #         v_emb = outputs['pred_captions'][idx,self.num_queries:2*self.num_queries-1]
        #         t_emb = grd_emb[-1:]

        #         t_emb = t_emb / (t_emb.norm(dim=-1, keepdim=True) + 1e-7)
        #         v_emb = v_emb / (v_emb.norm(dim=-1, keepdim=True) + 1e-7)            

        #         temperature = self.sem_seg_head.predictor.lang_encoder.logit_scale
        #         out_prob = vl_similarity(v_emb, t_emb, temperature=temperature)
                
        #         matched_id = out_prob.max(0)[1]
        #         grd_masks += [pred_gmasks[matched_id,:,:]]
        #     mask_pred_results += [torch.cat(grd_masks)]

        # comment for multi object inference.
        mask_pred_results = []
        for idx, batch_per_image in enumerate(batched_inputs):
            grd_texts = batch_per_image['groundings']['texts']
            if self.standard_text_for_eval:
                standard_texts = []
                for grd in batch_per_image['grounding_info']:
                    mask_file = grd['mask_file'].split('.')[0].split('/')[-1]
                    target = mask_file.split('_')[-1].replace('+', ' ')
                    site = mask_file.split('_')[-2].replace('+', ' ')
                    modality = mask_file.split('_')[-3].replace('+', ' ')
                    standard_texts.append(f'{target} in {site} {modality}')
                grd_texts = standard_texts
                batch_per_image['groundings']['texts'] = standard_texts


            gtext = self.sem_seg_head.predictor.lang_encoder.get_text_token_embeddings(grd_texts, name='grounding', token=False, norm=False)
            token_emb = gtext['token_emb']
            tokens = gtext['tokens']
            query_emb = token_emb[tokens['attention_mask'].bool()]
            non_zero_query_mask = torch.zeros(query_emb[:,None].shape[:-1], dtype=torch.bool, device=query_emb.device)

            extra['grounding_tokens'] = query_emb[:,None]
            extra['grounding_nonzero_mask'] = non_zero_query_mask.t()

            decomp_outputs = self.decomposition(images.tensor)
            features = self.backbone(decomp_outputs['pred'])
            outputs = self.sem_seg_head(features, extra=extra, task='grounding_eval')

            pred_gmasks = outputs['pred_gmasks'][idx]
            v_emb = outputs['pred_gtexts'][idx]
            t_emb = gtext['class_emb']

            t_emb = t_emb / (t_emb.norm(dim=-1, keepdim=True) + 1e-7)
            v_emb = v_emb / (v_emb.norm(dim=-1, keepdim=True) + 1e-7)            

            temperature = self.sem_seg_head.predictor.lang_encoder.logit_scale
            out_prob = vl_similarity(v_emb, t_emb, temperature=temperature)
            
            matched_id = out_prob.max(0)[1]
            mask_pred_results += [pred_gmasks[matched_id,:,:]]

        for i in range(len(mask_pred_results)):
            # upsample masks
            mask_pred_results[i] = F.interpolate(
                mask_pred_results[i][None,],
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )[0]

        processed_results = []
        for mask_pred_result, input_per_image, image_size in zip(
            mask_pred_results, batched_inputs, images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            processed_results.append({})

            mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                mask_pred_result, image_size, height, width
            )
            processed_results[-1]['grounding_mask'] = mask_pred_result

            # compute bbox
            # bbox = BitMasks(mask_pred_result > 0).get_bounding_boxes()
            # bbox = BoxMode.convert(bbox.tensor, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
            # processed_results[-1]['grounding_box'] = bbox

        return processed_results

    def evaluate_grounding_sptial(self, batched_inputs, mode):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)
        assert len(images.tensor) == 1, "grounding evaluation only support single batch size now"

        extra = {}
        dilation = 3
        pos_masks = (batched_inputs[0]['spatial_query']['rand_shape'].to(self.device)).unbind(0)
        pos_masks = ImageList.from_tensors(pos_masks, self.size_divisibility).tensor
        pos_masks = (F.conv2d(pos_masks.float(), self.dilation_kernel, padding=dilation//2) > 0).unbind(0)

        neg_masks = (batched_inputs[0]['spatial_query']['rand_shape'].to(self.device) & False).unbind(0)
        neg_masks = ImageList.from_tensors(neg_masks, self.size_divisibility).tensor.unbind(0)

        mask_pred_results = []
        for idx, batch_per_image in enumerate(batched_inputs):
            grd_texts = batch_per_image['groundings']['texts']
            grd_masks = []
            for idx2, anno_text in enumerate(grd_texts):
                extra.update({'spatial_query_pos_mask': [pos_masks[idx2]], 'spatial_query_neg_mask': [neg_masks[idx2]]})

                gtext = self.sem_seg_head.predictor.lang_encoder.get_text_token_embeddings([anno_text[0]], name='grounding', token=False, norm=False)
                token_emb = gtext['token_emb']
                tokens = gtext['tokens']
            
                grd_emb = token_emb[0][tokens['attention_mask'].bool()[0]]
                non_zero_query_mask = torch.zeros(grd_emb[:,None].shape[:-1], dtype=torch.bool, device=grd_emb.device)
                extra['grounding_tokens'] = grd_emb[:,None]
                extra['grounding_nonzero_mask'] = non_zero_query_mask.t()

                assert len(images.tensor) == 1, "grounding evaluation only support single batch size now"
                decomp_outputs = self.decomposition(images.tensor)
                features = self.backbone(decomp_outputs['pred'])
                outputs = self.sem_seg_head(features, extra=extra, task='grounding_eval')
                
                pred_gmasks = outputs['pred_gmasks'][idx]
                v_emb = outputs['pred_gtexts'][idx]
                t_emb = gtext['class_emb']

                t_emb = t_emb / (t_emb.norm(dim=-1, keepdim=True) + 1e-7)
                v_emb = v_emb / (v_emb.norm(dim=-1, keepdim=True) + 1e-7)            

                temperature = self.sem_seg_head.predictor.lang_encoder.logit_scale
                out_prob = vl_similarity(v_emb, t_emb, temperature=temperature)
                
                matched_id = out_prob.max(0)[1]
                grd_masks += [pred_gmasks[matched_id,:,:]]
                # grd_masks += [outputs['prev_mask'][0]]

            mask_pred_results += [torch.cat(grd_masks)]

        # comment for multi object inference.
        # mask_pred_results = []
        # for idx, batch_per_image in enumerate(batched_inputs):
        #     grd_texts = batch_per_image['groundings']['texts']
        #     grd_texts = [x[0] for x in grd_texts]

        #     gtext = self.sem_seg_head.predictor.lang_encoder.get_text_token_embeddings(grd_texts, name='grounding', token=False, norm=False)
        #     token_emb = gtext['token_emb']
        #     tokens = gtext['tokens']
        #     query_emb = token_emb[tokens['attention_mask'].bool()]
        #     non_zero_query_mask = torch.zeros(query_emb[:,None].shape[:-1], dtype=torch.bool, device=query_emb.device)

        #     extra['grounding_tokens'] = query_emb[:,None]
        #     extra['grounding_nonzero_mask'] = non_zero_query_mask.t()

        #     features = self.backbone(images.tensor)
        #     outputs = self.sem_seg_head(features, extra=extra, task='grounding_eval')

        #     pred_gmasks = outputs['pred_gmasks'][idx]
        #     v_emb = outputs['pred_gtexts'][idx]
        #     t_emb = gtext['class_emb']

        #     t_emb = t_emb / (t_emb.norm(dim=-1, keepdim=True) + 1e-7)
        #     v_emb = v_emb / (v_emb.norm(dim=-1, keepdim=True) + 1e-7)            

        #     temperature = self.sem_seg_head.predictor.lang_encoder.logit_scale
        #     out_prob = vl_similarity(v_emb, t_emb, temperature=temperature)
            
        #     matched_id = out_prob.max(0)[1]
        #     mask_pred_results += [pred_gmasks[matched_id,:,:]]

        for i in range(len(mask_pred_results)):
            # upsample masks
            mask_pred_results[i] = F.interpolate(
                mask_pred_results[i][None,],
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )[0]

        processed_results = []
        for mask_pred_result, input_per_image, image_size in zip(
            mask_pred_results, batched_inputs, images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            processed_results.append({})

            mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                mask_pred_result, image_size, height, width
            )
            processed_results[-1]['grounding_mask'] = mask_pred_result

        return processed_results

    def prepare_targets(self, batched_inputs, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        for idx, batch_per_image in enumerate(batched_inputs):            
            target_dict = {}
            if self.task_switch['mask']:
                targets_per_image = batch_per_image['instances'].to(self.device)
                # pad gt
                gt_masks = targets_per_image.gt_masks.tensor
                padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
                padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks

                gt_boxes = targets_per_image.gt_boxes.tensor
                ratio = torch.tensor([w_pad,h_pad,w_pad,h_pad]).to(gt_boxes.device)[None,:]
                gt_boxes = gt_boxes / ratio
                xc,yc,w,h = (gt_boxes[:,0] + gt_boxes[:,2])/2, (gt_boxes[:,1] + gt_boxes[:,3])/2, gt_boxes[:,2] - gt_boxes[:,0], gt_boxes[:,3] - gt_boxes[:,1]
                gt_boxes = torch.stack([xc,yc,w,h]).permute(1,0)

                target_dict.update({
                        "labels": targets_per_image.gt_classes,
                        "is_things": targets_per_image.is_things,
                        "masks": padded_masks,
                        "boxes": gt_boxes,
                        })

            if self.task_switch['spatial']:
                # prepare targets for spatial query
                target_dict['gt_spatial_masks'] = batch_per_image['spatial_query']['gt_masks']

            if self.task_switch['grounding']:
                grd_masks = batch_per_image['groundings']['masks']
                grd_texts = batch_per_image['groundings']['texts']
                grd_hash = batch_per_image['groundings']['hash']
                grd_task = batch_per_image['groundings']['mode']
                
                if len(grd_masks) == 0:
                    padded_masks = None
                else:
                    padded_masks = torch.zeros((grd_masks.shape[0], h_pad, w_pad), dtype=grd_masks.dtype, device=grd_masks.device)
                    padded_masks[:, : grd_masks.shape[1], : grd_masks.shape[2]] = grd_masks

                gtext = self.sem_seg_head.predictor.lang_encoder.get_text_token_embeddings(grd_texts, name='grounding', token=False, norm=False)
                token_emb = gtext['token_emb']
                tokens = gtext['tokens']
                
                unique_hash_id = np.unique(grd_hash, return_index=True)[1]
                selected_mask = np.zeros(len(grd_hash)).astype(bool)
                selected_mask[unique_hash_id] = True

                selected_token_emb = token_emb[selected_mask]
                selected_attn_mask = tokens['attention_mask'][selected_mask]
                query_emb = selected_token_emb[selected_attn_mask.bool()]
                
                class_idx = tokens['attention_mask'].sum(dim=-1) - 1
                class_idx = torch.stack((torch.arange(len(class_idx), device=class_idx.device), class_idx)).tolist()
                class_emb = token_emb[class_idx]
                
                target_dict['grounding_masks'] = padded_masks
                target_dict['grounding_query_embs'] = query_emb
                target_dict['grounding_class_embs'] = class_emb
                target_dict['grounding_hash'] = grd_hash
                target_dict['grounding_task'] = grd_task

            new_targets.append(target_dict)
        return new_targets

    def prepare_next_spaital_mask(self, outputs, batched_inputs, mode='best'):
        gt_masks = [batched_inputs[i]['spatial_query']['gt_masks'] for i in range(len(batched_inputs))]
        gt_masks = Spatial_ImageList.from_tensors(gt_masks, self.size_divisibility).tensor

        pred_masks = (F.interpolate(outputs['prev_mask'], size=gt_masks.shape[-2:], mode='bilinear', align_corners=False).sigmoid() > 0.5)
        prev_masks = nn.utils.rnn.pad_sequence(outputs['spatial_query_pos_mask'], padding_value=False, batch_first=True) | \
                        nn.utils.rnn.pad_sequence(outputs['spatial_query_neg_mask'], padding_value=False, batch_first=True)

        fn = gt_masks & (~(gt_masks & pred_masks)) & (~prev_masks) # fn: False Negative, gt:1, pred:0, prev:0
        fp = (~gt_masks & pred_masks) & (~prev_masks) # fp: False Positive, gt:0, pred:1, prev:0

        # compute iou between gt and pred
        iou = (gt_masks & pred_masks).sum(list(range(2,len(fn.shape)))) / ((gt_masks | pred_masks).sum(dim=list(range(2,len(fn.shape)))) + 1e-8)
        fn_sum = fn.sum(dim=list(range(2,len(fn.shape))))
        fp_sum = fp.sum(dim=list(range(2,len(fp.shape))))

        is_postive = fn_sum > fp_sum
        select_mask = torch.zeros_like(fn)
        select_mask[is_postive] = fn[is_postive]
        select_mask[~is_postive] = fp[~is_postive]
        # is_postive = torch.ones(len(fn_sum), device=torch.cuda.current_device()).bool()

        # conv implementation
        bs,ns,h,w = select_mask.shape
        mask_dt = (distance_transform((~F.pad(select_mask, pad=(1, 1, 1, 1), mode='constant', value=0)).float())[:,:,1:-1,1:-1]).reshape(bs*ns,-1)
        if mode == 'best':
            max_xy_idx = torch.stack([torch.arange(bs*ns), mask_dt.max(dim=-1)[1].cpu()]).tolist()
        elif mode == 'best_random':
            max_xy_idx = torch.stack([torch.arange(bs*ns), torch.cat([(mask_dt[i] > 0).nonzero()[torch.randint(0, len((mask_dt[i] > 0).nonzero()), (1,))][0] for i in range(len(mask_dt))]).cpu()]).tolist()
        next_mask = torch.zeros(gt_masks.shape, device=torch.cuda.current_device()).bool()
        next_mask = next_mask.view(bs*ns,-1)
        next_mask[max_xy_idx] = True
        next_mask = next_mask.reshape((bs*ns,1,h,w)).float()
        dilation = 3
        next_mask = F.conv2d(next_mask, self.dilation_kernel, padding=dilation//2).reshape(bs,ns,h,w) > 0

        # determine whether next mask is zero
        keep = (iou < 0.925)
        next_mask = next_mask & keep.view(bs,ns,1,1)

        pos_mask = []
        neg_mask = []
        for idx, ip in enumerate(is_postive):
            mask_len = len(outputs['spatial_query_pos_mask'][idx])
            pos_mask += [outputs['spatial_query_pos_mask'][idx] | (next_mask[idx][:mask_len] & ip[:mask_len,None,None])]
            neg_mask += [outputs['spatial_query_neg_mask'][idx] | (next_mask[idx][:mask_len] & (~ip[:mask_len,None,None]))]

        if 'false_positive_mask' in outputs:
            fp = outputs['false_positive_mask'] | fp
        return {'spatial_query_pos_mask': pos_mask, 'spatial_query_neg_mask': neg_mask, 'false_positive_mask': fp}

    def semantic_inference(self, mask_cls, mask_pred):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg

    def panoptic_inference(self, mask_cls, mask_pred):
        scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
        mask_pred = mask_pred.sigmoid()

        keep = labels.ne(self.sem_seg_head.num_classes) & (scores > self.object_mask_threshold)
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]
        cur_mask_cls = mask_cls[keep]
        cur_mask_cls = cur_mask_cls[:, :-1]

        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
        segments_info = []

        current_segment_id = 0

        if cur_masks.shape[0] == 0:
            # We didn't detect any mask :(
            return panoptic_seg, segments_info
        else:
            # take argmax
            cur_mask_ids = cur_prob_masks.argmax(0)
            stuff_memory_list = {}
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()
                isthing = pred_class in self.metadata.thing_dataset_id_to_contiguous_id.values()
                mask_area = (cur_mask_ids == k).sum().item()
                original_area = (cur_masks[k] >= 0.5).sum().item()
                mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)

                if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                    if mask_area / original_area < self.overlap_threshold:
                        continue

                    # merge stuff regions
                    if not isthing:
                        if int(pred_class) in stuff_memory_list.keys():
                            panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                            continue
                        else:
                            stuff_memory_list[int(pred_class)] = current_segment_id + 1

                    current_segment_id += 1
                    panoptic_seg[mask] = current_segment_id

                    segments_info.append(
                        {
                            "id": current_segment_id,
                            "isthing": bool(isthing),
                            "category_id": int(pred_class),
                        }
                    )

            return panoptic_seg, segments_info

    def instance_inference(self, mask_cls, mask_pred, box_pred):
        # mask_pred is already processed to have the same shape as original input
        image_size = mask_pred.shape[-2:]

        # [Q, K]
        scores = F.softmax(mask_cls, dim=-1)[:, :-1]
        labels = torch.arange(self.sem_seg_head.num_classes, device=self.device).unsqueeze(0).repeat(self.num_queries, 1).flatten(0, 1)
        # scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.num_queries, sorted=False)
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.test_topk_per_image, sorted=False)

        labels_per_image = labels[topk_indices]
        topk_indices = (topk_indices // self.sem_seg_head.num_classes)
        # mask_pred = mask_pred.unsqueeze(1).repeat(1, self.sem_seg_head.num_classes, 1).flatten(0, 1)
        mask_pred = mask_pred[topk_indices]
        if box_pred is not None:
            box_pred = box_pred[topk_indices]

        # if this is panoptic segmentation, we only keep the "thing" classes
        if self.panoptic_on:
            keep = torch.zeros_like(scores_per_image).bool()
            for i, lab in enumerate(labels_per_image):
                keep[i] = lab in self.metadata.thing_dataset_id_to_contiguous_id.values()

            scores_per_image = scores_per_image[keep]
            labels_per_image = labels_per_image[keep]
            mask_pred = mask_pred[keep]

            if box_pred is not None:
                box_pred = box_pred[keep]

        result = Instances(image_size)
        # mask (before sigmoid)
        result.pred_masks = (mask_pred > 0).float()
        # result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4))
        # Uncomment the following to get boxes from masks (this is slow)

        if box_pred is not None:
            result.pred_boxes = BitMasks(mask_pred > 0).get_bounding_boxes()
        else:
            result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4))

        # calculate average mask prob
        mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * result.pred_masks.flatten(1)).sum(1) / (result.pred_masks.flatten(1).sum(1) + 1e-6)
        result.scores = scores_per_image * mask_scores_per_image
        result.pred_classes = labels_per_image

        return result

    def prepare_targets4query(self, targets, images, topk=5):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        new_queries = []
        for targets_per_image in targets:
            # we randomly sample maximally topk concepts
            unique_target_classes = [k for k in set(targets_per_image.gt_classes.tolist())]
            selected_target_classes = random.sample(unique_target_classes, min(topk, len(unique_target_classes)))
            new_targets_per_image = []
            new_queries_per_image = []
            for clss in selected_target_classes:
                indices = (targets_per_image.gt_classes == clss).nonzero().view(-1)
                # pad gt
                gt_masks = targets_per_image.gt_masks[indices]
                padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
                padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks

                # convert class into concept name and then token seq
                self.sem_seg_head.predictor.lang_encoder.get_text_embeddings([BIOMED_CLASSES[clss]], name='grounding')
                query = getattr(self.sem_seg_head.predictor.lang_encoder, 'grounding_text_embeddings')

                new_targets.append(
                    {
                        "labels": targets_per_image.gt_classes[indices],
                        "masks": padded_masks,
                    }
                )
                new_queries_per_image.append(query)
            new_queries.append(new_queries_per_image)

        return new_targets, new_queries



@register_model
def get_seem_model(cfg, **kwargs):
    return BayesianSEEM(cfg)
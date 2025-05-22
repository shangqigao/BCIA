import sys
sys.path.append('./')

import os
import json
import torch
import pathlib
import argparse
import numpy as np
import nibabel as nib
from PIL import Image
from modeling.BaseModel import BaseModel
from modeling import build_model
from utilities.distributed import init_distributed
from utilities.arguments import load_opt_from_config_files
from utilities.constants import BIOMED_CLASSES

from inference_utils.inference import interactive_infer_image
from inference_utils.processing_utils import read_rgb
from inference_utils.output_processing import mask_stats

from scipy.stats import beta

def estimate_BiomedParse_Beta_parameters(img_paths, text_prompts, format='rgb', device="gpu"):
    """Estimate Beta parameters using RGB images and their segmentation probability maps
        img_paths: a list of paths for single-phase images
            or a list of lists, where each list has paths of multi-phase images.
            For multi-phase images, only nifti format is allowed.
        text_prompts: a list of strings with the same length as the img_paths
    """

    # Build model config
    opt = load_opt_from_config_files(["configs/biomedparse_inference.yaml"])
    opt = init_distributed(opt)

    # Load model from pretrained weights
    # pretrained_pth = 'output_singlephase_breastcancer/biomed_seg_lang_v1.yaml_conf~/run_1/00039390/default/model_state_dict.pt'
    pretrained_pth = 'output_multiphase_breastcancer/biomed_seg_lang_v1.yaml_conf~/run_1/00039390/default/model_state_dict.pt'

    if device == 'gpu':
        model = BaseModel(opt, build_model(opt)).from_pretrained(pretrained_pth).eval().cuda()
    else:
        model = BaseModel(opt, build_model(opt)).from_pretrained(pretrained_pth).eval().cpu()
    with torch.no_grad():
        model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(BIOMED_CLASSES + ["background"], is_eval=True)
    
    states = []
    for i, (img_path, text_prompt) in enumerate(zip(img_paths, text_prompts)):
        # read slices from dicom or nifti
        if format == 'rgb':
            image = read_rgb(img_path, keep_size=True)
        else:
            raise ValueError(f'Only support DICOM or NIFTI, but got {format}')

        print(f"Processing image [{i+1}/{len(img_paths)}] ...")
        # resize_mask=False would keep mask size to be (1024, 1024)
        pred_prob = interactive_infer_image(model, Image.fromarray(image), text_prompt, resize_mask=True, return_feature=False)
        pred_prob = (255*np.squeeze(pred_prob)).astype(np.uint8)
        state = mask_stats(pred_prob, image)
        states.append(state)
    
    # fitting beta distributions
    print(f"Fitting Beta distribution on {len(states)} samples ...")
    states = np.array(states)
    beta_params = []
    for i in range(states.shape[1]):
        data = np.clip(states[:, i], 1e-6, 1 - 1e-6)
        a, b, _, _ = beta.fit(data, floc=0, fscale=1)
        beta_params.append([a, b])

    return beta_params

if __name__ == "__main__":
    ## argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', default="/home/s/sg2162/projects/TCIA_NIFTI/image")
    parser.add_argument('--modality', default="MRI", choices=["CT", "MRI"], type=str)
    parser.add_argument('--format', default="rgb", choices=["dicom", "nifti"], type=str)
    parser.add_argument('--site', default="breast", type=str)
    parser.add_argument('--target', default="tumor", type=str)
    parser.add_argument('--save_dir', default="/home/sg2162/rds/hpc-work/Experiments/radiomics", type=str)
    args = parser.parse_args()

    if args.format == 'rgb':
        img_paths = sorted(pathlib.Path(args.img_dir).glob('*.png'))
        img_paths = list(img_paths)
    else:
        raise ValueError("Only support RGB images currently")
    
    text_prompts = [[f'{args.site} {args.target}']]*len(img_paths)

    # fit beta distributions
    beta_params = estimate_BiomedParse_Beta_parameters(
        img_paths=img_paths,
        text_prompts=text_prompts,
        format=args.format
    )

    # save beta parameters
    location = f"{args.modality}-{args.site}"
    target = args.target
    beta_dict = {location: {target: beta_params}}
    save_beta_path = f"{args.save_dir}/Beta_params.json"
    print(f"Saving Beta parameters to {save_beta_path} ...")
    with open(save_beta_path, 'w') as f:
        json.dump(beta_dict, f, indent=4)
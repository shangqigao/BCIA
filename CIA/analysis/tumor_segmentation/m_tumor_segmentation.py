import os
import sys

# Get the directory where the current script resides
script_dir = os.path.dirname(os.path.abspath(__file__))

# Add a relative subdirectory to sys.path
relative_path = os.path.join(script_dir, '../../')
sys.path.append(relative_path)

import ast
import json
import torch
import pathlib
import argparse
import numpy as np
import nibabel as nib
import pandas as pd
from PIL import Image
from modeling.BaseModel import BaseModel
from modeling import build_model
from utilities.distributed import init_distributed
from utilities.arguments import load_opt_from_config_files
from utilities.constants import BIOMED_CLASSES

from inference_utils.inference import interactive_infer_image
from inference_utils.processing_utils import read_dicom
from inference_utils.processing_utils import read_nifti_inplane

from analysis.tumor_segmentation.m_post_processing import remove_inconsistent_objects
from peft import PeftModel, PeftConfig

def extract_radiology_segmentation(
        img_paths, 
        text_prompts,
        class_name,
        model_mode, 
        save_dir,
        is_CT=True,
        site='kidney',
        meta_list=None,
        img_format='nifti',
        beta_params=None,
        prompt_ensemble=False
    ):
    """extract segmentation from radiology images
    Args:
        img_paths (list): a list of image paths
        text_prompts (list): a list of text prompts
        class_name (str): target of segmentation
        model_mode (str): name of segmentation model
        save_dir (str): directory of saving masks
        is_CT (bool): if the modality is CT
        site (str): the site of scan, such as kidney
        img_format (str): only support nifti or dicom
    """
    if model_mode == "BiomedParse":
        _ = extract_BiomedParse_segmentation(
            img_paths,
            text_prompts,
            save_dir,
            format=img_format,
            is_CT=is_CT,
            site=site,
            meta_list=meta_list,
            beta_params=beta_params,
            prompt_ensemble=prompt_ensemble
        )
    else:
        raise ValueError(f"Invalid model mode: {model_mode}")
    return

def extract_BiomedParse_segmentation(img_paths, text_prompts, save_dir,
                                  format='nifti', is_CT=True, site=None, 
                                  meta_list=None, beta_params=None, 
                                  prompt_ensemble=False, device="gpu"):
    """extracting radiomic features slice by slice in a size of (1024, 1024)
        img_paths: a list of paths for single-phase images
            or a list of lists, where each list has paths of multi-phase images.
            For multi-phase images, only nifti format is allowed.
        text_prompts: a list of strings with the same length as the img_paths
        meta_list (list): a list of imaging metadata, 
            such as 'field_strength', 'bilateral', 'scanner_manufacturer'
        prompt_ensemble: if true, use prompt ensemble
        beta_params: the parameters of Beta distribution, 
            if provided, it would be used to compute p-values of segmented objects
            if p-value is less than alpha, i.e., 0.05, the object would be removed
    """

    # Build model config
    opt = load_opt_from_config_files([os.path.join(relative_path, "configs/biomedparse_inference.yaml")])
    opt = init_distributed(opt)

    # Load model from pretrained weights
    pretrained_pth = os.path.join(relative_path, 'checkpoints/LoRA_multiphase_breast/biomedparse_v1.pt')
    lora_pth = os.path.join(relative_path, 'checkpoints/LoRA_multiphase_breast')

    if device == 'gpu':
        if not opt.get('LoRA', False):
            model = BaseModel(opt, build_model(opt)).from_pretrained(pretrained_pth).eval().cuda()
        else:
            model = BaseModel(opt, build_model(opt)).from_pretrained(pretrained_pth)
            model = PeftModel.from_pretrained(model, lora_pth).model.eval().cuda()
    else:
        raise ValueError(f'Require gpu, but got {device}')
    
    with torch.no_grad():
        model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(BIOMED_CLASSES + ["background"], is_eval=True)
    
    for idx, (img_path, text_prompt) in enumerate(zip(img_paths, text_prompts)):
        # read slices from dicom or nifti
        if format == 'dicom':
            dicom_dir = pathlib.Path(img_path)
            assert pathlib.Path(img_path).is_dir()
            dicom_paths = sorted(dicom_dir.glob('*.dcm'))
            images = [read_dicom(p, is_CT, site, keep_size=True, return_spacing=True) for p in dicom_paths]
            slice_axis, affine = 0, np.eye(4)
        elif format == 'nifti':
            images, slice_axis, affine = read_nifti_inplane(img_path, is_CT, site, keep_size=True, return_spacing=True)
        else:
            raise ValueError(f'Only support DICOM or NIFTI, but got {format}')

        mask_3d = []
        image_4d = []
        prob_3d = []
        for i, element in enumerate(images):
            assert len(element) == 3
            img, spacing, phase = element

            # use prompt ensemble
            if prompt_ensemble:
                meta_data = {} if meta_list is None else meta_list[idx]
                assert isinstance(meta_data, dict)
                meta_data['view'] = phase
                meta_data['slice_index'] = f'{i:03}'
                meta_data['modality'] = 'CT' if is_CT else 'MRI'
                meta_data['site'] = site
                meta_data['target'] = text_prompt
                if len(spacing) == 2:
                    meta_data['pixel_spacing'] = spacing
                else:
                    assert len(spacing) == 3
                    pixel_index = list(set([0, 1, 2]) - {slice_axis})
                    pixel_spacing = [spacing[i] for i in pixel_index]
                    meta_data['pixel_spacing'] = pixel_spacing
                text_prompts = create_prompts(meta_data)
                text_prompts = [text_prompts[9]]
            else:
                text_prompts = [text_prompt]
            # print(f"Segmenting slice [{i+1}/{len(images)}] ...")

            # resize_mask=False would keep mask size to be (1024, 1024)
            ensemble_prob = []
            for text_prompt in text_prompts:
                pred_prob = interactive_infer_image(model, Image.fromarray(img), text_prompt, resize_mask=True, return_feature=False)
                ensemble_prob.append(pred_prob)
            pred_prob = np.max(np.concatenate(ensemble_prob, axis=0), axis=0, keepdims=True)
            if beta_params is not None:
                image_4d.append(img)
                prob_3d.append(pred_prob)
            pred_mask = (1*(pred_prob > 0.5)).astype(np.uint8)
            mask_3d.append(pred_mask)
        
        # post-processing predicted masks
        mask_3d = np.concatenate(mask_3d, axis=0)
        if beta_params is not None:
            prob_3d = np.concatenate(prob_3d, axis=0)
            image_4d = np.stack(image_4d, axis=0)
            print("Post-processing by removing both unconfident predictions and spatially inconsistent objects")
            mask_3d = remove_inconsistent_objects(mask_3d, prob_3d=prob_3d, image_4d=image_4d, beta_params=beta_params)
        else:
            print("Post-processing by removing spatially inconsistent objects")
            if format == 'dicom':
                voxel_spacing = None
            else:
                voxel_spacing = spacing.tolist()
                z_spacing = voxel_spacing.pop(slice_axis)
                voxel_spacing.insert(0, z_spacing)
            mask_3d = remove_inconsistent_objects(mask_3d, spacing=voxel_spacing)
        final_mask = np.moveaxis(mask_3d, 0, slice_axis)
        
        if isinstance(img_path, list):
            img_name = pathlib.Path(img_path[0]).name.replace("_0000.nii.gz", "")
        else:
            img_name = pathlib.Path(img_path).name.replace(".nii.gz", "")
        save_mask_path = f"{save_dir}/{img_name}.nii.gz"
        print(f"Saving predicted segmentation to {save_mask_path}")
        nifti_img = nib.Nifti1Image(final_mask, affine)
        nib.save(nifti_img, save_mask_path)
    return

def load_beta_params(modality, site, target):
    beta_path = os.path.join(relative_path, 'analysis/tumor_segmentation/Beta_params.json')
    with open(beta_path, 'r') as f:
        data = json.load(f)
        beta_params = data[f"{modality}-{site}"][target]

    return beta_params

def create_prompts(meta_data):
    keys = ['view', 'slice_index', 'modality', 'site', 'target']
    assert all(meta_data.get(k) is not None for k in keys), f"all basic info {keys} should be provided"
    view = meta_data['view']
    slice_index = meta_data['slice_index']
    modality = meta_data['modality']
    site = meta_data['site']
    target_name = meta_data['target']
    target = 'tumor' if 'tumor' in target_name else target_name

    basic_prompts = [
        f"{target_name} in {site} {modality}",
        f"{view} slice {slice_index} showing {target} in {site}",
        f"{target} located in the {site} on {modality}",
        f"{view} {site} {modality} with {target}",
        f"{target} visible in slice {slice_index} of {modality}",
    ]

    # meta information
    keys = ['pixel_spacing', 'field_strength', 'bilateral', 'scanner_manufacturer']
    meta_prompts = []
    if all(meta_data.get(k) is not None for k in keys):
        pixel_spacing = meta_data['pixel_spacing']
        x_spacing, y_spacing = pixel_spacing[0], pixel_spacing[1]
        field_strength = meta_data['field_strength']
        bilateral_mri = meta_data['bilateral']
        lateral = 'bilateral' if bilateral_mri == 1 else 'unilateral'
        manufacturer = meta_data['scanner_manufacturer']
        meta_prompts = [
            f"a {modality} scan of the {lateral} {site}, {view} view, slice {slice_index}, pixel spacing {x_spacing:.2f}x{y_spacing:.2f} mm, showing {target}",
            f"{lateral} {site} {modality} in {view} view at slice {slice_index} with spacing {x_spacing:.2f}x{y_spacing:.2f} mm, includes {target}",
            f"{view} slice {slice_index} from a {field_strength}T {manufacturer} {modality} of the {lateral} {site}, pixel spacing {x_spacing:.2f}x{y_spacing:.2f} mm, showing {target}",
            f"{lateral} {site} {modality} in {view} view, slice {slice_index}, using {field_strength}T {manufacturer} scanner, spacing {x_spacing:.2f}x{y_spacing:.2f} mm, showing {target}",
            f"{modality} of the {lateral} {site} at slice {slice_index}, {view} view, spacing: {x_spacing:.2f}x{y_spacing:.2f} mm, scanned by {field_strength}T {manufacturer} scanner, shows {target}"
        ]
    
    return basic_prompts + meta_prompts

if __name__ == "__main__":
    ## argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', default="/home/s/sg2162/projects/TCIA_NIFTI/image")
    parser.add_argument('--beta_params', default="/home/s/sg2162/projects/TCIA_NIFTI/image")
    parser.add_argument('--modality', default="MRI", choices=["CT", "MRI"], type=str)
    parser.add_argument('--phase', default="multiphase", choices=["single", "multiple"], type=str)
    parser.add_argument('--format', default="nifti", choices=["dicom", "nifti"], type=str)
    parser.add_argument('--site', default="breast", type=str)
    parser.add_argument('--target', default="tumor", type=str)
    parser.add_argument('--meta_info', default=None)
    parser.add_argument('--save_dir', default="/home/sg2162/rds/hpc-work/Experiments/radiomics", type=str)
    parser.add_argument('--model_mode', default="BiomedParse", choices=["SegVol", "BiomedParse"], type=str)
    args = parser.parse_args()

    if args.format == 'dicom':
        img_paths = pathlib.Path(args.img_dir).glob('*')
        img_paths = [p for p in img_paths if p.is_dir()]
        patient_ids = [p.name for p in img_paths]
    else:
        if args.phase == "single":
            img_paths = sorted(pathlib.Path(args.img_dir).rglob('*_0001.nii.gz'))
            patient_ids = [p.parent.name for p in img_paths]
        else:
            case_paths = sorted(pathlib.Path(args.img_dir).glob('*'))
            case_paths = [p for p in case_paths if p.is_dir()]
            img_paths = []
            patient_ids = []
            for path in case_paths:
                patient_ids.append(path.name)
                nii_paths = path.glob("*.nii.gz")
                multiphase_keys = ["_0000.nii.gz", "_0001.nii.gz", "_0002.nii.gz"]
                nii_paths = [p for p in nii_paths if any(k in p.name for k in multiphase_keys)]
                img_paths.append(sorted(nii_paths))
    # text_prompts = [[f'{args.site} {args.target} in {args.site} {args.modality}']]*len(img_paths)
    text_prompts = [[f'{args.site} {args.target}']]*len(img_paths)
    save_dir = pathlib.Path(args.save_dir)

    # read clinical and imaging info
    if args.meta_info is not None:
        df_meta = pd.read_excel(args.meta_info, sheet_name='dataset_info')
        df_meta['pixel_spacing'] = df_meta['pixel_spacing'].apply(ast.literal_eval)
        meta_list = []
        for patient_id in patient_ids:
            field_strength = df_meta.loc[df_meta["patient_id"] == patient_id, 'field_strength'].values[0]
            bilateral_mri = df_meta.loc[df_meta["patient_id"] == patient_id, 'bilateral_mri'].values[0]
            lateral = 'bilateral' if bilateral_mri == 1 else 'unilateral'
            manufacturer = df_meta.loc[df_meta["patient_id"] == patient_id, 'manufacturer'].values[0]
            meta_data = {
                'field_strength': field_strength, 
                'bilateral': lateral, 
                'scanner_manufacturer': manufacturer
            }
            meta_list.append(meta_data)
    else:
        meta_list = None

    with open(args.beta_params, 'r') as f:
        data = json.load(f)
        beta_params = data[f"{args.modality}-{args.site}"][args.target]

    # extract radiology segmentation
    bs = 8
    nb = len(img_paths) // bs if len(img_paths) % bs == 0 else len(img_paths) // bs + 1
    for i in range(0, nb):
        print(f"Processing images of batch [{i+1}/{nb}] ...")
        start = i * bs
        end = min(len(img_paths), (i + 1) * bs)
        batch_img_paths = img_paths[start:end]
        batch_txt_prompts = text_prompts[start:end]
        extract_radiology_segmentation(
            img_paths=batch_img_paths,
            text_prompts=batch_txt_prompts,
            class_name=args.target,
            model_mode=args.model_mode,
            save_dir=save_dir,
            is_CT=args.modality == 'CT',
            site=args.site,
            meta_list=meta_list,
            img_format=args.format,
            beta_params=None,
            prompt_ensemble=True
        )
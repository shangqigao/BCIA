import sys
sys.path.append('./')

import argparse
import pathlib
import nibabel as nib
import numpy as np
from scipy import stats
from scipy.ndimage import label, zoom
from skimage.measure import regionprops
from inference_utils.processing_utils import get_orientation

def dice_coeff(a, b):
    inter = np.logical_and(a, b).sum()
    if a.sum() + b.sum() == 0:
        return 1.0
    return 2 * inter / (a.sum() + b.sum())

def compute_beta_pvalue(mask_3d, image_4d, beta_params):
    """Compute p-value in terms of given params of Beta distributions
        mask_3d: a prediction mask with pixel values in [0, 255] for probability in [0, 1]
        image_4d: a 4D image with shape [z, x, y, 3]
        beta_params: a list of Beta distribution parameters
    """
    if mask_3d.max() <= 127:
        states = [0, 0, 0, 0]
    else:
        s1 = mask_3d[mask_3d>=128].mean()/256
        s2 = image_4d[:,:,:,0][mask_3d>=128].mean()/256
        s3 = image_4d[:,:,:,1][mask_3d>=128].mean()/256
        s4 = image_4d[:,:,:,2][mask_3d>=128].mean()/256
        states = [s1, s2, s3, s4]

    ps = [stats.ks_1samp([states[i]], stats.beta(param[0], param[1]).cdf).pvalue for i, param in enumerate(beta_params)]
    p_value = np.prod(ps)
    adj_p_value = p_value**0.25

    return adj_p_value


def remove_inconsistent_objects(mask_3d, min_slices=3, min_dice=0.2,
                                prob_3d=None, image_4d=None, beta_params=None, alpha=0.05):
    """
    Remove 3D objects that are not consistent across slices.

    Args:
        mask_3d: 3D numpy array (binary mask).
        min_slices: Minimum number of slices an object must appear in.
        min_dice: Minimum Dice similarity between slices to consider consistent.
        prob_3d: a segmentation probability map with values in [0, 1]
        image_4d: a 4D image with shape [z, x, y, 3]
        beta_params: a list of Beta distribution parameters in terms of probability, r, g, b values
        alpha: statistical significance level
    Returns:
        Cleaned 3D mask.
    """
    labeled, num = label(mask_3d)
    cleaned = np.zeros_like(mask_3d)
    num_objects = 0
    for region_id in range(1, num + 1):
        component = (labeled == region_id)

        z_indices = np.any(component, axis=(1, 2)).nonzero()[0]
        
        if len(z_indices) < min_slices:
            continue  # Too few slices → skip

        # calculate p-value of statistical test
        if prob_3d is not None and image_4d is not None and beta_params is not None:
            object_prob = np.zeros_like(mask_3d)
            object_prob[component] = prob_3d[component]
            object_prob = (255*object_prob).astype(np.int8)
            pvalue = compute_beta_pvalue(object_prob, image_4d, beta_params)
            if pvalue < alpha:
                num_objects += 1
                continue # significantly different and inconfident prediction  → skip
            
        consistent = True
        for i in range(len(z_indices) - 1):
            z1, z2 = z_indices[i], z_indices[i + 1]
            slice1 = component[z1]
            slice2 = component[z2]
            if dice_coeff(slice1, slice2) < min_dice:
                consistent = False
                break # significantly inconsistant slices

        if consistent:
            cleaned[component] = 1

    print(f"Removed {num_objects} of {num} objects with statistical significant difference!")
    return cleaned

if __name__ == "__main__":
    ## argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--nifti_dir', default="/Users/sg2162/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/CancerDatasets/sanity-check/predictions")
    parser.add_argument('--save_dir', default="/Users/sg2162/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/CancerDatasets/sanity-check/post-processed", type=str)
    args = parser.parse_args()

    # args.nifti_dir = "/Users/sg2162/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/CancerDatasets/MAMA-MIA/segmentations/expert"
    pathlib.Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    nifti_files = pathlib.Path(args.nifti_dir).glob("*.nii.gz")
    nifti_files = list(nifti_files)
    tumor_sizes = []
    new_spacing = (1.0, 1.0, 1.0)
    for i, nifti in enumerate(nifti_files):
        nii_name = nifti.name
        print(f"Processing [{i+1}/{len(nifti_files)}]")
        nii = nib.load(nifti)
        affine = nii.affine
        phase, slice_axis, pixel_spacing = get_orientation(affine)
        zoom_factors = tuple(os/ns for os, ns in zip(pixel_spacing, new_spacing))
        nii_img = nii.get_fdata()
        original_shape = nii_img.shape
        # resample to (1, 1, 1)
        nii_img = zoom(nii_img, zoom=zoom_factors, order=0)
        new_shape = nii_img.shape
        nii_img = np.moveaxis(nii_img, slice_axis, 0)
        tsize = (np.sum(nii_img, axis=(1,2)) > 0).sum()
        tumor_sizes.append(tsize)
        processed_img = remove_inconsistent_objects(nii_img, min_slices=13)
        processed_img = np.moveaxis(processed_img, 0, slice_axis)
        zoom_factors = tuple(os/ns for os, ns in zip(original_shape, new_shape))
        processed_img = zoom(processed_img, zoom=zoom_factors, order=0)
        processed_img = nib.Nifti1Image(processed_img, affine)
        save_path = f"{args.save_dir}/{nii_name}"
        nib.save(processed_img, save_path)
    tumor_sizes = np.array(tumor_sizes)
    lowerbound = np.percentile(tumor_sizes, 1)
    upperbound = np.percentile(tumor_sizes, 99)
    print(f"Tumor size @1percentile={lowerbound}, @99percentile={upperbound}")
    print(f"Minimal tumor size:", min(tumor_sizes))
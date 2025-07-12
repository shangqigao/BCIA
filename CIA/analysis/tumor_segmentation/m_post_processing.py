import argparse
import pathlib
import nibabel as nib
import numpy as np
from scipy import stats, ndimage
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

def keep_largest_components(mask, num_components=1):
    # Label connected components
    labeled_mask, num = ndimage.label(mask)
    
    if num <= num_components:
        return mask  # Nothing to filter

    # Measure size of each component
    sizes = ndimage.sum(mask, labeled_mask, range(1, num + 1))
    
    # Get labels of the N largest components
    largest_labels = np.argsort(sizes)[-num_components:] + 1  # +1 because labels start at 1

    # Create a new mask keeping only largest N components
    output_mask = np.isin(labeled_mask, largest_labels).astype(np.uint8)
    return output_mask

def remove_inconsistent_objects(mask_3d, min_slices=4, min_dice=0.0, spacing=None, new_spacing=(1.0, 1.0, 1.0),
                                prob_3d=None, image_4d=None, beta_params=None, alpha=0.05):
    """
    Remove 3D objects that are not consistent across slices.

    Args:
        mask_3d: 3D numpy array (binary mask).
        min_slices: Minimum number of slices an object must appear in. 1 percentile=13, min=6
        min_dice: Minimum Dice similarity between slices to consider consistent.
        spacing: orginal voxel spacing (z, x, y)
        new_spacing: new spacing for resampling
        prob_3d: a segmentation probability map with values in [0, 1]
        image_4d: a 4D image with shape [z, x, y, 3]
        beta_params: a list of Beta distribution parameters in terms of probability, r, g, b values
        alpha: statistical significance level
    Returns:
        Cleaned 3D mask.
    """
    # resample to new spacing
    if spacing is not None:
        zoom_factors = tuple(os/ns for os, ns in zip(spacing, new_spacing))
        original_shape = mask_3d.shape
        mask_3d = zoom(mask_3d, zoom=zoom_factors, order=0)
        new_shape = mask_3d.shape

    labeled, num = label(mask_3d)
    cleaned = np.zeros_like(mask_3d)
    num_objects = 0
    for region_id in range(1, num + 1):
        component = (labeled == region_id)

        z_indices = np.any(component, axis=(1, 2)).nonzero()[0]
        
        if len(z_indices) < min_slices:
            num_objects += 1
            continue  # Too few slices → skip

        # calculate p-value of statistical test
        if prob_3d is not None and image_4d is not None and beta_params is not None:
            object_prob = np.zeros_like(prob_3d)
            object_prob[component] = prob_3d[component]
            object_prob = (255*object_prob).astype(np.uint8)
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
                num_objects += 1
                break # significantly inconsistant slices

        if consistent:
            cleaned[component] = 1

    print(f"Removed {num_objects} of {num} objects with spatial inconsistency!")
    cleaned = keep_largest_components(cleaned)
    # resample to orginal shape
    if spacing is not None:
        zoom_factors = tuple(os/ns for os, ns in zip(original_shape, new_shape))
        cleaned = zoom(cleaned, zoom=zoom_factors, order=0)
    return cleaned
import numpy as np
from skimage import transform
import pydicom
from io import BytesIO
from PIL import Image
import nibabel as nib
import SimpleITK as sitk
from skimage import measure
    

"""
    This script contains utility functions for reading and processing different imaging modalities.
"""


CT_WINDOWS = {'abdomen': [-150, 250],
              'lung': [-1000, 1000],
              'pelvis': [-55, 200],
              'liver': [-25, 230],
              'colon': [-68, 187],
              'pancreas': [-100, 200]}

def process_intensity_image(image_data, is_CT, site=None, keep_size=True):
    # process intensity-based image. If CT, apply site specific windowing
    
    # image_data: 2D numpy array of shape (H, W)
    
    # return: 3-channel numpy array of shape (H, W, 3) as model input
    
    if is_CT:
        # process image with windowing
        if site and site in CT_WINDOWS:
            window = CT_WINDOWS[site]
        else:
            raise ValueError(f'Please choose CT site from {CT_WINDOWS.keys()}')
        lower_bound, upper_bound = window
    else:
        # process image with intensity range 0.5-99.5 percentile
        lower_bound, upper_bound = np.percentile(
            image_data[image_data > 0], 0.5
        ), np.percentile(image_data[image_data > 0], 99.5)
        
    image_data_pre = np.clip(image_data, lower_bound, upper_bound)
    image_data_pre = (
        (image_data_pre - image_data_pre.min())
        / (image_data_pre.max() - image_data_pre.min())
        * 255.0
    )
    
    if keep_size:
        resize_image = image_data_pre
    else:
        # pad to square with equal padding on both sides
        shape = image_data_pre.shape
        if shape[0] > shape[1]:
            pad = (shape[0]-shape[1])//2
            pad_width = ((0,0), (pad, pad))
        elif shape[0] < shape[1]:
            pad = (shape[1]-shape[0])//2
            pad_width = ((pad, pad), (0,0))
        else:
            pad_width = None
        
        if pad_width is not None:
            image_data_pre = np.pad(image_data_pre, pad_width, 'constant', constant_values=0)
            
        # resize image to 1024x1024
        image_size = 1024
        resize_image = transform.resize(image_data_pre, (image_size, image_size), order=3, 
                                        mode='constant', preserve_range=True, anti_aliasing=True)
    
    # convert to 3-channel image
    resize_image = np.stack([resize_image]*3, axis=-1)
        
    return resize_image.astype(np.uint8)

def process_intensity_3Dimage(image_data, is_CT, site=None, keep_size=True):
    # process intensity-based 3D image. If CT, apply site specific windowing
    
    # image_data: 3D numpy array of shape (H, W, N)
    
    # return: N-channel numpy array of shape (H, W, N) as model input
    
    if is_CT:
        # process image with windowing
        if site and site in CT_WINDOWS:
            window = CT_WINDOWS[site]
        else:
            raise ValueError(f'Please choose CT site from {CT_WINDOWS.keys()}')
        lower_bound, upper_bound = window
    else:
        # process image with intensity range 0.5-99.5 percentile
        lower_bound, upper_bound = np.percentile(
            image_data[image_data > 0], 0.5
        ), np.percentile(image_data[image_data > 0], 99.5)
        
    image_data_pre = np.clip(image_data, lower_bound, upper_bound)
    image_data_pre = (
        (image_data_pre - image_data_pre.min())
        / (image_data_pre.max() - image_data_pre.min())
        * 255.0
    )
    
    if keep_size:
        resize_image = image_data_pre
    else:
        # pad to square with equal padding on both sides
        shape = image_data_pre.shape
        if shape[0] > shape[1]:
            pad = (shape[0]-shape[1])//2
            pad_width = ((0,0), (pad, pad), (0,0))
        elif shape[0] < shape[1]:
            pad = (shape[1]-shape[0])//2
            pad_width = ((pad, pad), (0,0), (0,0))
        else:
            pad_width = None
        
        if pad_width is not None:
            image_data_pre = np.pad(image_data_pre, pad_width, 'constant', constant_values=0)
            
        # resize image to 1024x1024
        image_size = 1024
        N = image_data_pre.shape[2]
        resize_image = np.zeros((image_size, image_size, N), dtype=np.uint8)
        for i in range(N):
            resize_image[:,:,i] = transform.resize(image_data_pre[:,:,i], (image_size, image_size), order=3, 
                                        mode='constant', preserve_range=True, anti_aliasing=True)
    
    # convert to 3-channel image
    # resize_image = np.stack([resize_image]*3, axis=-1)
        
    return resize_image.astype(np.uint8)

def read_dicom(image_path, is_CT, site=None, keep_size=False, return_spacing=False):
    # read dicom file and return pixel data
    
    # dicom_file: str, path to dicom file
    # is_CT: bool, whether image is CT or not
    # site: str, one of CT_WINDOWS.keys()
    # return: 2D numpy array of shape (H, W)
    
    ds = pydicom.dcmread(image_path)
    spacing = ds.PixelSpacing

    image_array = ds.pixel_array * ds.RescaleSlope + ds.RescaleIntercept
    
    image_array = process_intensity_image(image_array, is_CT, site, keep_size)
    
    if return_spacing:
        return image_array, spacing
    else:
        return image_array


def read_nifti(image_path, is_CT, slice_idx, site=None, HW_index=(0, 1), channel_idx=None):
    # read nifti file and return pixel data
    
    # image_path: str, path to nifti file
    # is_CT: bool, whether image is CT or not
    # slice_idx: int, slice index to read
    # site: str, one of CT_WINDOWS.keys()
    # HW_index: tuple, index of height and width in the image shape
    # return: 2D numpy array of shape (H, W)
    
    
    nii = nib.load(image_path)
    image_array = nii.get_fdata()
    
    if HW_index != (0, 1):
        image_array = np.moveaxis(image_array, HW_index, (0, 1))
    
    # get slice
    if channel_idx is None:
        image_array = image_array[:, :, slice_idx]
    else:
        image_array = image_array[:, :, slice_idx, channel_idx]
        
    image_array = process_intensity_image(image_array, is_CT, site)
    return image_array

def get_orientation(affine):
    ornt = nib.orientations.io_orientation(affine)
    axcodes = nib.orientations.ornt2axcodes(ornt)

    # Compute voxel spacing from affine
    voxel_sizes = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))

    # Find the two axes with the closest spacing (likely in-plane)
    diffs = np.abs(voxel_sizes[:, None] - voxel_sizes[None, :])
    diffs[np.eye(3, dtype=bool)] = np.inf  # ignore diagonal
    i1, i2 = np.unravel_index(np.argmin(diffs), diffs.shape)
    in_plane_axes = {i1, i2}
    slice_axis = (set([0, 1, 2]) - in_plane_axes).pop()
    slice_dir = axcodes[slice_axis]
    pixel_spacing = voxel_sizes

    if slice_dir in ('I', 'S'):
        return 'axial', slice_axis, pixel_spacing
    elif slice_dir in ('A', 'P'):
        return 'coronal', slice_axis, pixel_spacing
    elif slice_dir in ('L', 'R'):
        return 'sagittal', slice_axis, pixel_spacing
    else:
        return 'unknown', slice_axis, pixel_spacing

def read_nifti_inplane(image_path, is_CT, site=None, keep_size=False, return_spacing=False):
    """read single-phase or multi-phase nifti file and return pixel data
        image_path: str, path to single-phase nifti file
            or a list of paths to multi-phase nifti files
        is_CT: bool, whether image is CT or not
        site: str, one of CT_WINDOWS.keys()
        return: 2D numpy array of shape (H, W)
            or 3D numpy array of shape (H, W, N)
    """
    if isinstance(image_path, list):
        assert len(image_path) > 1
        multiphase = True
        image_list, phases, slice_axises = [], [], []
        for path in image_path:
            nii = nib.load(path)
            affine = nii.affine
            phase, slice_axis, pixel_spacing = get_orientation(affine)
            phases.append(phase)
            slice_axises.append(slice_axis)
            image_list.append(nii.get_fdata())
        assert len(list(set(phases))) == 1, f"Inconsistent scanning phase: {phases}"
        assert len(list(set(slice_axises))), f"Inconsistent slice axis: {slice_axises}"
        image_array = np.stack(image_list, axis=-1)
    else:
        multiphase = False
        nii = nib.load(image_path)
        affine = nii.affine
        phase, slice_axis, pixel_spacing = get_orientation(affine)
        image_array = nii.get_fdata()

    image_list = []
    if phase in ['axial', 'sagittal', 'coronal']:
        for i in range(image_array.shape[slice_axis]):
            if slice_axis == 0:
                slice_img = image_array[i, :, :]
            elif slice_axis == 1:
                slice_img = image_array[:, i, :]
            elif slice_axis == 2:
                slice_img = image_array[:, :, i]
            else:
                raise ValueError(f"Slice axis is {slice_axis}, maximum shoud be 2")
            
            if multiphase:
                slice_img = process_intensity_3Dimage(slice_img, is_CT, site, keep_size)
            else:
                slice_img = process_intensity_image(slice_img, is_CT, site, keep_size)

            if return_spacing:
                image_list.append((slice_img, pixel_spacing))
            else:
                image_list.append(slice_img)
    else:
        raise ValueError("Unsupported or unknown scanning phase")

    return image_list, slice_axis, affine
    


def read_rgb(image_path, keep_size=False):
    # read RGB image and return resized pixel data
    
    # image_path: str, path to RGB image
    # return: BytesIO buffer
    
    # read image into numpy array
    image = Image.open(image_path)
    image = np.array(image)
    if len(image.shape) == 2:
        image = np.stack([image]*3, axis=-1)
    elif image.shape[2] == 4:
        image = image[:,:,:3]
    
    # pad to square with equal padding on both sides
    shape = image.shape
    if shape[0] > shape[1]:
        pad = (shape[0]-shape[1])//2
        pad_width = ((0,0), (pad, pad), (0,0))
    elif shape[0] < shape[1]:
        pad = (shape[1]-shape[0])//2
        pad_width = ((pad, pad), (0,0), (0,0))
    else:
        pad_width = None
        
    if pad_width is not None:
        image = np.pad(image, pad_width, 'constant', constant_values=0)
        
    # resize image to 1024x1024 for each channel
    if keep_size:
        resize_image = image
    else:
        image_size = 1024
        resize_image = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        for i in range(3):
            resize_image[:,:,i] = transform.resize(image[:,:,i], (image_size, image_size), order=3, 
                                        mode='constant', preserve_range=True, anti_aliasing=True)
        
    return resize_image



def get_instances(mask):
    # get intances from binary mask
    seg = sitk.GetImageFromArray(mask)
    filled = sitk.BinaryFillhole(seg)
    d = sitk.SignedMaurerDistanceMap(filled, insideIsPositive=False, squaredDistance=False, useImageSpacing=False)

    ws = sitk.MorphologicalWatershed( d, markWatershedLine=False, level=1)
    ws = sitk.Mask( ws, sitk.Cast(seg, ws.GetPixelID()))
    ins_mask = sitk.GetArrayFromImage(ws)
    
    # filter out instances with small area outliers
    props = measure.regionprops_table(ins_mask, properties=('label', 'area'))
    mean_area = np.mean(props['area'])
    std_area = np.std(props['area'])
    
    threshold = mean_area - 2*std_area - 1
    ins_mask_filtered = ins_mask.copy()
    for i, area in zip(props['label'], props['area']):
        if area < threshold:
            ins_mask_filtered[ins_mask == i] = 0
            
    return ins_mask_filtered
    
    
# ==================== MAMA-MIA CHALLENGE SAMPLE SUBMISSION ====================
#
# This is the official sample submission script for the **MAMA-MIA Challenge**, 
# covering both tasks:
#
#   1. Primary Tumour Segmentation (Task 1)
#   2. Treatment Response Classification (Task 2)
#
# ----------------------------- SUBMISSION FORMAT -----------------------------
# Participants must implement a class `Model` with one or two of these methods:
#
#   - `predict_segmentation(output_dir)`: required for Task 1
#       > Must output NIfTI files named `{patient_id}.nii.gz` in a folder
#       > called `pred_segmentations/`
#
#   - `predict_classification(output_dir)`: required for Task 2
#       > Must output a CSV file `predictions.csv` in `output_dir` with columns:
#           - `patient_id`: patient identifier
#           - `pcr`: binary label (1 = pCR, 0 = non-pCR)
#           - `score`: predicted probability (flaot between 0 and 1)
#
#   - `predict_classification(output_dir)`: if a single model handles both tasks
#       > Must output NIfTI files named `{patient_id}.nii.gz` in a folder
#       > called `pred_segmentations/`
#       > Must output a CSV file `predictions.csv` in `output_dir` with columns:
#           - `patient_id`: patient identifier
#           - `pcr`: binary label (1 = pCR, 0 = non-pCR)
#           - `score`: predicted probability (flaot between 0 and 1)
#
# You can submit:
#   - Only Task 1 (implement `predict_segmentation`)
#   - Only Task 2 (implement `predict_classification`)
#   - Both Tasks (implement both methods independently or define `predict_segmentation_and_classification` method)
#
# ------------------------ SANITY-CHECK PHASE ------------------------
#
# ✅ Before entering the validation or test phases, participants must pass the **Sanity-Check phase**.
#   - This phase uses **4 samples from the test set** to ensure your submission pipeline runs correctly.
#   - Submissions in this phase are **not scored**, but must complete successfully within the **20-minute timeout limit**.
#   - Use this phase to debug your pipeline and verify output formats without impacting your submission quota.
#
# 💡 This helps avoid wasted submissions on later phases due to technical errors.
#
# ------------------------ SUBMISSION LIMITATIONS ------------------------
#
# ⚠️ Submission limits are strictly enforced per team:
#   - **One submission per day**
#   - **Up to 15 submissions total on the validation set**
#   - **Only 1 final submission on the test set**
#
# Plan your development and testing accordingly to avoid exhausting submissions prematurely.
#
# ----------------------------- RUNTIME AND RESOURCES -----------------------------
#
# > ⚠️ VERY IMPORTANT: Each image has a **timeout of 5 minutes** on the compute worker.
#   - **Validation Set**: 58 patients → total budget ≈ 290 minutes
#   - **Test Set**: 516 patients → total budget ≈ 2580 minutes
#
# > The compute worker environment is based on the Docker image:
#       `lgarrucho/codabench-gpu:latest`
#
# > You can install additional dependencies via `requirements.txt`.
#   Please ensure all required packages are listed there.
#
# ----------------------------- SEGMENTATION DETAILS -----------------------------
#
# This example uses `nnUNet v2`, which is compatible with the GPU compute worker.
# Note the following nnUNet-specific constraints:
#
# ✅ `predict_from_files_sequential` MUST be used for inference.
#     - This is because nnUNet’s multiprocessing is incompatible with the compute container.
#     - In our environment, a single fold prediction using `predict_from_files_sequential` 
#       takes approximately **1 minute per patient**.
#
# ✅ The model uses **fold 0 only** to reduce runtime.
# 
# ✅ Predictions are post-processed by applying a breast bounding box mask using 
#    metadata provided in the per-patient JSON file.
#
# ----------------------------- CLASSIFICATION DETAILS -----------------------------
#
# If using predicted segmentations for Task 2 classification:
#   - Save them in `self.predicted_segmentations` inside `predict_segmentation()`
#   - You can reuse them in `predict_classification()`
#   - Or perform Task 1 and Task 2 inside `predict_segmentation_and_classification`
#
# ----------------------------- DATASET INTERFACE -----------------------------
# The provided `dataset` object is a `RestrictedDataset` instance and includes:
#
#   - `dataset.get_patient_id_list() → list[str]`  
#         Patient IDs for current split (val/test)
#
#   - `dataset.get_dce_mri_path_list(patient_id) → list[str]`  
#         Paths to all image channels (typically pre and post contrast)
#         - iamge_list[0] corresponds to the pre-contrast image path
#         - iamge_list[1] corresponds to the first post-contrast image path and so on
#
#   - `dataset.read_json_file(patient_id) → dict`  
#         Metadata dictionary per patient, including:
#         - breast bounding box (`primary_lesion.breast_coordinates`)
#         - scanner metadata (`imaging_data`), etc...
#
# Example JSON structure:
# {
#   "patient_id": "XXX_XXX_SXXXX",
#   "primary_lesion": {
#     "breast_coordinates": {
#         "x_min": 1, "x_max": 158,
#         "y_min": 6, "y_max": 276,
#         "z_min": 1, "z_max": 176
#     }
#   },
#   "imaging_data": {
#     "bilateral": true,
#     "dataset": "HOSPITAL_X",
#     "site": "HOSPITAL_X",
#     "scanner_manufacturer": "SIEMENS",
#     "scanner_model": "Aera",
#     "field_strength": 1.5,
#     "echo_time": 1.11,
#     "repetition_time": 3.35
#   }
# }
#
# ----------------------------- RECOMMENDATIONS -----------------------------
# ✅ We recommend to always test your submission first in the Sanity-Check Phase.
#    As in Codabench the phases need to be sequential and they cannot run in parallel,
#    we will open a secondary MAMA-MIA Challenge Codabench page with a permanen Sanity-Check phase.
#   That way you won't lose submission trials to the validation or even wore, the test set.
# ✅ We recommend testing your solution locally and measuring execution time per image.
# ✅ Use lightweight models or limit folds if running nnUNet.
# ✅ Keep all file paths, patient IDs, and formats **exactly** as specified.
# ✅ Ensure your output folders are created correctly (e.g. `pred_segmentations/`)
# ✅ For faster runtime, only select a single image for segmentation.
#
# ------------------------ COPYRIGHT ------------------------------------------
#
# © 2025 Lidia Garrucho. All rights reserved.
# Unauthorized use, reproduction, or distribution of any part of this competition's 
# materials is prohibited without explicit permission.
#
# ------------------------------------------------------------------------------

# === MANDATORY IMPORTS ===
import os
import pandas as pd
import shutil

import sys
# Get the directory where the current script resides
script_dir = os.path.dirname(os.path.abspath(__file__))

# Add a relative subdirectory to sys.path
relative_path = os.path.join(script_dir, 'CIA')
sys.path.append(relative_path)

# Set the environment variable before importing Detectron2
os.environ["DETECTRON2_DISABLE_CV2"] = "1"

# === OPTIONAL IMPORTS: only needed if you modify or extend nnUNet input/output handling ===
# You can remove unused imports above if not needed for your solution
import numpy as np
import torch
import SimpleITK as sitk
# === PanCIA IMPORTS ===
from CIA.analysis.tumor_segmentation.m_tumor_segmentation import extract_BiomedParse_segmentation
from CIA.analysis.tumor_segmentation.m_tumor_segmentation import load_beta_params

class Model:
    def __init__(self, dataset):
        """
        Initializes the model with the restricted dataset.
        
        Args:
            dataset (RestrictedDataset): Preloaded dataset instance with controlled access.
        """
        # MANDATOR
        self.dataset = dataset  # Restricted Access to Private Dataset
        self.predicted_segmentations = None  # Optional: stores path to predicted segmentations
        self.modality = "MRI"
        self.site = "breast"
        self.target = "tumor"

    def predict_segmentation(self, output_dir):
        """
        Task 1 — Predict tumor segmentation with nnUNetv2.
        You MUST define this method if participating in Task 1.

        Args:
            output_dir (str): Directory where predictions will be stored.

        Returns:
            str: Path to folder with predicted segmentation masks.
        """
        # === Participants can modify how they prepare input ===
        patient_ids = self.dataset.get_patient_id_list()
        multiphase_images = []
        meta_list = []
        for patient_id in patient_ids:
            images = self.dataset.get_dce_mri_path_list(patient_id)
            multiphase_images.append(images[:3])
            patient_info = self.dataset.read_json_file(patient_id)
            meta_data = patient_info["imaging_data"]
            coords = patient_info["primary_lesion"]["breast_coordinates"]
            meta_data.update({"breast_coordinates": coords})
            meta_list.append(meta_data)
        text_prompts = [f"{self.site} {self.target}"]*len(multiphase_images)
        # beta_params = load_beta_params(self.modality, self.site, self.target)
        # === Output folder for raw nnUNet segmentations ===
        output_dir_cia = os.path.join(output_dir, 'cia_seg')
        os.makedirs(output_dir_cia, exist_ok=True)

        extract_BiomedParse_segmentation(
            img_paths=multiphase_images,
            text_prompts=text_prompts,
            save_dir=output_dir_cia,
            is_CT=self.modality == 'CT',
            site=self.site,
            meta_list=meta_list,
            beta_params=None,
            prompt_ensemble=True
        )
        
       # === Final output folder (MANDATORY name) ===
        output_dir_final = os.path.join(output_dir, 'pred_segmentations')
        os.makedirs(output_dir_final, exist_ok=True)

        # === Optional post-processing step ===
        # For example, you can threshold the predictions or apply morphological operations
        # Here, we iterate through the predicted segmentations and apply the breast mask to each segmentation
        # to remove false positives outside the breast region
        for patient_id in self.dataset.get_patient_id_list():
            seg_path = os.path.join(output_dir_cia, f"{patient_id}.nii.gz")
            if not os.path.exists(seg_path):
                print(f'{seg_path} NOT FOUND!')
                continue
            
            segmentation = sitk.ReadImage(seg_path)
            segmentation_array = sitk.GetArrayFromImage(segmentation)
            
            patient_info = self.dataset.read_json_file(patient_id)
            if not patient_info or "primary_lesion" not in patient_info:
                continue
            
            coords = patient_info["primary_lesion"]["breast_coordinates"]
            x_min, x_max = coords["x_min"], coords["x_max"]
            y_min, y_max = coords["y_min"], coords["y_max"]
            z_min, z_max = coords["z_min"], coords["z_max"]
            
            masked_segmentation = np.zeros_like(segmentation_array)
            masked_segmentation[x_min:x_max, y_min:y_max, z_min:z_max] = \
                segmentation_array[x_min:x_max, y_min:y_max, z_min:z_max]            
            masked_seg_image = sitk.GetImageFromArray(masked_segmentation)
            masked_seg_image.CopyInformation(segmentation)

            # MANDATORY: the segmentation masks should be named using the patient_id
            final_seg_path = os.path.join(output_dir_final, f"{patient_id}.nii.gz")
            sitk.WriteImage(masked_seg_image, final_seg_path)

        # Save path for Task 2 if needed
        self.predicted_segmentations = output_dir_final

        return output_dir_final
    
    # def predict_classification(self, output_dir):
    #     """
    #     Task 2 — Predict treatment response (pCR).
    #     You MUST define this method if participating in Task 2.

    #     Args:
    #         output_dir (str): Directory to save output predictions.

    #     Returns:
    #         pd.DataFrame: DataFrame with patient_id, pcr prediction, and score.
    #     """
    #     patient_ids = self.dataset.get_patient_id_list()
    #     predictions = []
        
    #     for patient_id in patient_ids:
    #         if self.predicted_segmentations:
    #             # === Example using segmentation-derived feature (volume) ===
    #             seg_path = os.path.join(self.predicted_segmentations, f"{patient_id}.nii.gz")
    #             if not os.path.exists(seg_path):
    #                 continue
                
    #             segmentation = sitk.ReadImage(seg_path)
    #             segmentation_array = sitk.GetArrayFromImage(segmentation)
    #             # You can use the predicted segmentation to compute features if task 1 is done
    #             # For example, compute the volume of the segmented region
    #             # ...

    #             # RANDOM CLASSIFIER AS EXAMPLE
    #             # Replace with real feature extraction + ML model
    #             probability = np.random.rand()
    #             pcr_prediction = int(probability > 0.5)

    #         else:
    #             # === Example using raw image intensity for rule-based prediction ===
    #             image_paths = self.dataset.get_dce_mri_path_list(patient_id)
    #             if not image_paths:
    #                 continue
                
    #             image = sitk.ReadImage(image_paths[1])
    #             image_array = sitk.GetArrayFromImage(image)
    #             mean_intensity = np.mean(image_array)
    #             pcr_prediction = 1 if mean_intensity > 500 else 0
    #             probability = np.random.rand() if pcr_prediction == 1 else np.random.rand() / 2
            
    #         # === MANDATORY output format ===
    #         predictions.append({
    #             "patient_id": patient_id,
    #             "pcr": pcr_prediction,
    #             "score": probability
    #         })

    #     return pd.DataFrame(predictions)

# IMPORTANT: The definition of this method will skip the execution of `predict_segmentation` and `predict_classification` if defined
    # def predict_segmentation_and_classification(self, output_dir):
    #     """
    #     Define this method if your model performs both Task 1 (segmentation) and Task 2 (classification).
    #     
    #     This naive combined implementation:
    #         - Generates segmentation masks using thresholding.
    #         - Applies a rule-based volume threshold for response classification.
    #     
    #     Args:
    #         output_dir (str): Path to the output directory.
    #     
    #     Returns:
    #         str: Path to the directory containing the predicted segmentation masks (Task 1).
    #         DataFrame: Pandas DataFrame containing predicted labels and scores (Task 2).
    #     """
    #     # Folder to store predicted segmentation masks
    #     output_dir_final = os.path.join(output_dir, 'pred_segmentations')
    #     os.makedirs(output_dir_final, exist_ok=True)

    #     predictions = []

    #     for patient_id in self.dataset.get_patient_id_list():
    #         # Load DCE-MRI series (assuming post-contrast is the second timepoint)
    #         image_paths = self.dataset.get_dce_mri_path_list(patient_id)
    #         if not image_paths or len(image_paths) < 2:
    #             continue

    #         image = sitk.ReadImage(image_paths[1])
    #         image_array = sitk.GetArrayFromImage(image)

    #         # Step 1: Naive threshold-based segmentation
    #         threshold_value = 150
    #         segmentation_array = (image_array > threshold_value).astype(np.uint8)

    #         # Step 2: Mask segmentation to breast region using provided lesion coordinates
    #         patient_info = self.dataset.read_json_file(patient_id)
    #         if not patient_info or "primary_lesion" not in patient_info:
    #             continue

    #         coords = patient_info["primary_lesion"]["breast_coordinates"]
    #         x_min, x_max = coords["x_min"], coords["x_max"]
    #         y_min, y_max = coords["y_min"], coords["y_max"]
    #         z_min, z_max = coords["z_min"], coords["z_max"]

    #         masked_segmentation = np.zeros_like(segmentation_array)
    #         masked_segmentation[x_min:x_max, y_min:y_max, z_min:z_max] = \
    #             segmentation_array[x_min:x_max, y_min:y_max, z_min:z_max]

    #         # Save predicted segmentation
    #         masked_seg_image = sitk.GetImageFromArray(masked_segmentation)
    #         masked_seg_image.CopyInformation(image)
    #         seg_path = os.path.join(output_dir_final, f"{patient_id}.nii.gz")
    #         sitk.WriteImage(masked_seg_image, seg_path)

    #         # Step 3: Classify based on tumour volume (simple rule-based)
    #         tumor_volume = np.sum(masked_segmentation > 0)
    #         pcr_prediction = 1 if tumor_volume < 1000 else 0
    #         probability = 0.5  # Example: fixed confidence

    #         predictions.append({
    #             "patient_id": patient_id,
    #             "pcr": pcr_prediction,
    #             "score": probability
    #         })

    #     return output_dir_final, pd.DataFrame(predictions)
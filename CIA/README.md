# **Cancer Image Analysis**

This repository provides code for cancer image analysis, which implements a text-prompted tumor segmentation model.

### Analysis

This folder contains the code for tumor segmentation in `analysis/tumor_segmentation/m_tumor_segmentation.py`, including post-processing steps in `analysis/tumor_segmentation/m_post_processing.py`

### Checkpoints

This folder is required to include checkpoints for running our model. Our pre-trained model will be released after the MAMA-MIA challenge, following the challenge rules.

### Others

The others folders, such as `configs`, `inference_utils`, `modeling`, and `utilities`, provide the details of model archietectures, training and test settings.

### Acknowlegement
This implementation is based on [BiomedParse](https://github.com/microsoft/BiomedParse). We sincerely thank the authors for sharing their source code.
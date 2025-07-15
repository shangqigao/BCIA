# **Breast Cancer Image Analysis (BCIA)**

This repository provides source code of participating the MAMA-MIA 2025 challenge. Following the rules of this challenge, the details of training and test keep confidential at this stage and will be updated after the challenge.

### For challenge organizers

All required dependencies have been included in `requirements.txt`. This code has been sucessfully tested at sanity-check phase and validation phase on Codabench. Our code particularly depends on detectron2, and we have compiled a wheel (`detectron2-0.6-cp310-cp310-linux_x86_64.whl`) based on the provided python environment by the organizers. 

### Installation

1. `git clone https://github.com/shangqigao/BCIA`
2. Create an environment from the YAML file by `conda env create -f environment.yml`

If gcc version is lower than 9, try

```
conda install -c conda-forge compilers
```

### Data preparation

The details of datasets we have used will be updated after the challenge

### Training

Our training details will be updated after the challenge

### Testing

Our testing detials will be updated after the challenge

### Download model

We will provide pre-trained model for testing after the challenge

## 🛡️ License

This project is under the Apache-2.0 license. See [LICENSE](LICENSE) for details.

# Acknowlegement

Our implementation is based on [BiomedParse](https://github.com/microsoft/BiomedParse). We thank the authors for sharing their source code.

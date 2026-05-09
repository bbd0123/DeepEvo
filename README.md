# DeepEvo
An interpretable deep learning framework specifically-designed for cross-species comparison, providing an effective approach for predicting evolutionary *cis*-regulatory variants.
<img width="1979" height="767" alt="image" src="https://github.com/user-attachments/assets/7da1e26b-0eb7-4c5b-889c-8942eea6b11b" />

# System Requirements & Installation
DeepEvo is implemented in Python and utilizes PyTorch for deep learning. We highly recommend using [Conda](https://docs.conda.io/en/latest/miniconda.html) to manage the environment and ensure reproducibility.

**Tested Operating Systems:** CentOS Linux 7 (Core)\
**Typical Install Time:** 30-40 minutes (depending on network speed)

### 1. Clone the repository

```bash
git clone https://github.com/bbd0123/DeepEvo.git
cd DeepEvo
```

### 2. Create and activate the Conda environment
# Demos
| Name            | Description                                                                 |
|-----------------|-----------------------------------------------------------------------------|
| [Demo_code](https://github.com/bbd0123/DeepEvo/tree/main/Demo_code)       | A step-by-step tutorial demonstrating expression difference prediction and base-level effect inference using ISM with DeepEvo. |
| [Demo_data](https://github.com/bbd0123/DeepEvo/tree/main/Demo_data)       | Example datasets used in the demonstration code.                             |
| [trained_models](https://github.com/bbd0123/DeepEvo/tree/main/trained_models)  | Two pretrained models generated using DeepEvo.                              |
| [training_script](https://github.com/bbd0123/DeepEvo/tree/main/training_script) | Scripts for training DeepEvo models from scratch.                           |



This will install Python, PyTorch (v2.4.0), and all required dependencies.

```bash
conda env create -f environment.yml
conda activate deepevo_env
```

*(Note: The default `environment.yml(https://github.com/bbd0123/DeepEvo/blob/main/environment.yml)` is configured for GPUs with CUDA 11.8. If you are running on a CPU-only machine or a different CUDA version, please refer to the PyTorch official website to modify the `pytorch-cuda` version accordingly.)*


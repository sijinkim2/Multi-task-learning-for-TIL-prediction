# Multi-task-learning-for-TIL-prediction
Multi-task learning framework for tumor-infiltrating lymphocyte (TIL) prediction.
This repository contains the implementation of a Multi-Task Learning (MTL) framework for histopathology image analysis, designed to jointly perform tissue segmentation and
lymphocyte detection in order to improve Tumor-Infiltrating Lymphocytes (TIL) prediction.
By learning related tasks simultaneously, the proposed framework aims to leverage shared
representations and enhance predictive performance compared to conventional single-task approaches.

For comparison purposes, a Single-Task Learning (STL) baseline is also provided, where each task is trained independently using the same backbone architecture.

The proposed models are based on DeepLabv3+, employing a shared encoder and ASPP module, followed by task-specific heads. The framework is designed for patch-level pathology image analysis and supports customized loss functions for segmentation and detection tasks.

##Features

- Multi-task learning framework based on DeepLabv3+
- Shared encoder and ASPP module with task-specific output heads
- Patch-level processing for histopathology images
- Custom loss functions for tissue segmentation and lymphocyte detection
- Side-by-side implementation of MTL vs STL training pipelines for fair comparison

##Dataset

This project uses the TIGER Challenge dataset, which consists of multiple pathology-related subsets for different tasks. An example dataset path is shown below:


The dataset is organized as follows:

- BCSS  
  Used for tissue segmentation, containing pathology images and corresponding segmentation masks at both WSI and ROI levels.

- Cells
  Used for lymphocyte detection, including images and annotation files provided in JSON format, available at WSI and ROI levels.

- **TILS**  
  Used for TIL prediction, consisting of whole-slide images (WSIs) and associated TIL score lists provided in XML format.

Please ensure that you have appropriate access rights to the dataset before running the code. The dataset can be accessed from [Dataset](https://registry.opendata.aws/tiger/).

---

##Repository Structure

```text
MTL/                     # Multi-task learning (Unified MTL)
 ├─ Dataset.py
 ├─ Datamodule.py
 ├─ DeepLabv3_plus.py
 ├─ ASPP.py
 ├─ Loss_function.py
 └─ main.py              # Entry point for MTL training/eval

STL/                     # Single-task learning baselines
 ├─ Dataset.py
 ├─ Datamodule.py
 ├─ DeepLabv3_plus.py
 ├─ ASPP.py
 ├─ Loss_function.py
 ├─ Seg_main.py          # Segmentation-only entry point
 └─ det_main.py          # Detection-only entry point
```

The `MTL` directory contains the multi-task learning pipeline, while the `STL` directory provides independent training scripts for segmentation and detection.
The `TIL_prediction` directory includes scripts for downstream TIL score prediction and evaluation.

---

# How to Run

To train the **Multi-Task Learning** model, run:

python main.py
To train the Single-Task Learning models separately:
Tissue segmentation:
python Seg_main.py

Lymphocyte detection:

python det_main.py

Dataset paths and hyperparameters need to be adjusted manually.

##**Pretrained Model**

This work uses the pretrained model released by [Kang et al., 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Kang_Benchmarking_Self-Supervised_Learning_on_Diverse_Pathology_Datasets_CVPR_2023_paper.html). 
Please refer to the official repository for model details and instructions for obtaining and loading the pretrained weights: [Pretrained Model](https://github.com/lunit-io/benchmark-ssl-pathology).

## **Released Checkpoint (Best-performing Unified MTL)**

This repository provides the checkpoint for the **best-performing Unified MTL** model used in our experiments (as reported in the manuscript).
Download: **[link-to-checkpoint]**



##Environment
All required dependencies are listed in requirements.txt.

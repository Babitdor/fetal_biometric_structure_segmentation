## Fetal Biometric Structure Segmentation using U-Nets

# Overview
This project focuses on the automatic segmentation of fetal biometric structures from ultrasound images using deep learning, specifically U-Net architectures. The model is trained on the Fetal Head Ultrasound Dataset and targets accurate delineation of fetal head structures (e.g., head circumference) to support prenatal diagnostic workflows.

The repository includes:

Data preprocessing and augmentation scripts

U-Net model implementation (customizable)

Training and evaluation pipelines

Visualization utilities for segmentation results

# Dataset
We use the Fetal Head Ultrasound Dataset, which contains ultrasound images and corresponding ground truth annotations for fetal head regions.

📂 Note: Dataset access might require permission or registration depending on the source.

Preprocessing steps include:

Resizing images to uniform dimensions

# Normalization

Data augmentation (random rotations, flips, intensity scaling)

# Model
The segmentation model is based on a standard U-Net architecture with the following characteristics:

Encoder-decoder structure with skip connections

Dice loss and binary cross-entropy loss combination for training

Batch normalization and dropout for regularization

Adjustable depth and number of filters

# Optional:

Support for deeper U-Net variants (e.g., Attention U-Net, Residual U-Net)

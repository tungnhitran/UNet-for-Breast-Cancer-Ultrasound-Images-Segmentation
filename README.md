# U-Net for Breast Cancer Ultrasound Image Segmentation
This repository contains related files implementing a U-Net model using TensorFlow 2 and Keras for semantic segmentation of breast ultrasound images. The model is trained on the Breast Ultrasound Images Dataset to identify tumors in ultrasound images.

## Overview
The notebook `pinkstrap.ipynb` is the main code designed for breast cancer ultrasound image segmentation using the U-Net architecture. This notebook demonstrates:
- Data loading and preprocessing from datasets (original, augmented, or merged versions).
- Building a U-Net architecture with convolutional blocks.
- Model compilation, training, and evaluation.
- Visualizing loss/accuracy curves and predictions.
The original version of this notebook is first shown in Kaggle, where the datasets are saved.

## Dataset
The notebook uses the Breast Ultrasound Images Dataset from Kaggle, with optional augmented or merged versions:

- Original dataset (without merging masks).
- Unfiltered images with merged masks.
- Augmented data for improved training.

Images and masks are resized to 256x256 pixels and split into train/validation/test sets (70/10/20 ratio).
Note: The dataset includes classes: normal, benign, and malignant. Ensure the dataset is downloaded and structured correctly (e.g., images and masks in class-specific directories).

## Requirements
To run the notebook, ensure the following dependencies are installed:
- Python 3.x
- Jupyter Notebook
- TensorFlow/Keras
- OpenCV
- And common libraries such as: Numpy, Matplotlib, Pandas, Seaborn.

## Model Structure
The U-Net consists of:
- Enocder: Convolutional blocks with max pooling.
- Decoder: Transpose convolutions with skip connections.
- Output: Sigmoid activation for binary segmentation (tumor vs. background).
  
The number of channels is used as a hyperparameter to determine the optimal configuration of U-Net architecture for the task. 

## Results
- Training on augmented data shows improved stability compared to unfiltered merged data.
- Example visualizations in the notebook demonstrate predicted masks vs. ground truth.
- Accuracy up to 97.89%.

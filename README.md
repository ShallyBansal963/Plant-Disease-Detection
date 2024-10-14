# Pneumonia Detection using Convolutional Neural Network (CNN)

This project aims to detect pneumonia from chest X-ray images using a Convolutional Neural Network (CNN) implemented in Python with TensorFlow and Keras. The dataset is stored in Google Drive, and the model is trained and evaluated using Google Colab.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Conclusion](#conclusion)

## Introduction
Pneumonia is a severe respiratory infection that can lead to hospitalization and can be fatal if not diagnosed and treated promptly. This project uses deep learning techniques to classify X-ray images into two categories: Normal and Pneumonia. 

## Dataset
The dataset used in this project is the [Chest X-ray Images (Pneumonia) dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia), which contains:
- **Train Directory**: Contains labeled images for training (Normal and Pneumonia).
- **Validation Directory**: Contains labeled images for validation.
- **Test Directory**: Contains labeled images for testing.

### Directory Structure
/content/drive/My Drive/chest_xray/ ├── train/ │ ├── NORMAL/ │ └── PNEUMONIA/ ├── val/ │ ├── NORMAL/ │ └── PNEUMONIA/ └── test/ ├── NORMAL/ └── PNEUMONIA/

## Installation
To run this project, make sure you have the following installed:
- Python 3.x
- TensorFlow
- Keras
- NumPy
- Matplotlib
- scikit-learn

You can install the required libraries using pip:
```bash
pip install tensorflow numpy matplotlib scikit-learn

## Usage
Mount your Google Drive in Google Colab.
Place your chest X-ray dataset in the specified directory.
Run the provided code to train the model.

## Code Overview
The code includes:
Data preprocessing with ImageDataGenerator.
Model definition using Sequential API.
Training the model with class weights to handle class imbalance.
Evaluation of the model on the test data.

## Model Architecture

The CNN model consists of:

Convolutional layers with ReLU activation.
MaxPooling layers.
BatchNormalization layers.
Dropout layers for regularization.
Fully connected layers with a sigmoid activation function for binary classification.

## Summary of the Model
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 conv2d (Conv2D)             (None, 222, 222, 32)      896
 max_pooling2d (MaxPooling2D) (None, 111, 111, 32)    0
 batch_normalization (BatchNormalization) (None, 111, 111, 32) 128
 ...
 dense_2 (Dense)              (None, 1)                129
=================================================================
Total params: 22,279,617
Trainable params: 22,278,657
Non-trainable params: 960
_________________________________________________________________

## Results

The model was evaluated on the test set, yielding the following results:

              precision    recall  f1-score   support

      NORMAL       0.89      1.00      0.94         8
   PNEUMONIA       1.00      0.88      0.93         8

    accuracy                           0.94        16
   macro avg       0.94      0.94      0.94        16
weighted avg       0.94      0.94      0.94        16

## Confusion Matrix

Confusion Matrix:
 [[8 0]
 [1 7]]
Test Accuracy: 0.94

## Conclusion

This project demonstrates the effectiveness of using a CNN for pneumonia detection from chest X-ray images. Future improvements can include experimenting with different architectures, hyperparameters, and utilizing transfer learning.

## Acknowledgments
Kaggle for the dataset.
TensorFlow and Keras for the deep learning framework.


Feel free to adjust any sections or add additional details as needed!












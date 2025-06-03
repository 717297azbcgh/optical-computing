# Deep Learning and Optical Computing Collaborative Processing Guide

This project provides a training and inference framework that integrates deep learning with optical computing for both medical images (BraTS dataset) and handwritten digits (MNIST dataset).  
It includes data preprocessing, model definition and training, optical computing input/output conversion, and hybrid inference analysis combining neural networks with optical computing.

---

## I. BraTS Dataset Scripts

### Data Processing & Model Training

- `get_benign_data.py`  
  Splits the original BraTS dataset into training, validation, and test sets, and generates index files.

- `dataset.py`  
  Loads the split BraTS image data and constructs PyTorch-compatible data loaders.

- `model.py`  
  Defines a U-Net-based model architecture for medical image segmentation.

- `train.py`  
  Configures training parameters and executes model training.

- `train_benign.py`  
  A training script that supports checkpoint-based resume training by loading an existing model.

- `utils.py`  
  A collection of helper functions, including:  
  - Model saving and loading  
  - Mean and variance computation  
  - Data loader construction  
  - Accuracy evaluation  
  - Saving visualized validation results  

- `my_checkpoint.pth.tar`  
  A trained U-Net model checkpoint for inference or continued training.

- `training_metrics.csv`  
  Performance log recorded during BraTS model training.

### Optical Computing Integration

- `read.ipynb`  
  A Jupyter Notebook for:  
  - Loading the trained model  
  - Exporting the first convolutional layer input (as optical computing input)  
  - Importing optical computing output and resuming the remaining network inference  
  - Analyzing the impact of optical computing on overall model performance  

---

## II. MNIST Dataset Scripts

- `train_and_read_mnist.ipynb`  
  A Jupyter Notebook that:  
  - Builds and trains a compact CNN for MNIST digit classification  
  - Outputs the first convolutional layer's input for optical processing  
  - Loads optical convolution results and completes inference using the rest of the network  

- `mnist_cnn.pth`  
  The trained CNN model weights for the MNIST dataset, available for inference or further training.

---

## Recommended Workflow

1. **BraTS Dataset Training & Inference**:  
   - Use `get_benign_data.py` for data preparation  
   - Train the model using `train.py` or `train_benign.py`  
   - Perform hybrid inference using `read.ipynb` with optical convolution output  

2. **MNIST Dataset Training & Inference**:  
   - Run `train_and_read_mnist.ipynb` to complete training and hybrid inference analysis  

---

## Environment Requirements

- Use the Conda environment configured by `pytorch_GPU.yaml`

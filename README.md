# Neural Network Models

This repository contains Jupyter notebooks that implement neural networks using **PyTorch**. The goal is to demonstrate deep learning techniques for both **tabular data classification** and **image classification**.

## Table of Contents

- [NNModel.ipynb](#nnmodelipynb) – A simple feedforward neural network for **Iris dataset classification**.
- [CNN.ipynb](#cnnipynb) – A **Convolutional Neural Network (CNN)** for **image classification** using the **MNIST** dataset.

---

## NNModel.ipynb

This notebook demonstrates training a **feedforward neural network** with PyTorch to classify the **Iris dataset**.

### Overview

The notebook walks through the following steps:

1. **Import Libraries** – Uses `torch`, `pandas`, and `matplotlib`.
2. **Load Data** – Reads the Iris dataset from a CSV file.
3. **Data Preprocessing** – Converts categorical labels into numerical values.
4. **Define the Model** – Implements a feedforward neural network using `torch.nn.Module`.
5. **Train-Test Split** – Splits the dataset into features and labels.
6. **Training & Evaluation** – Trains the model and evaluates accuracy.
7. **Visualization** – Plots training loss and accuracy.

### Model Architecture

- **Input Layer**: 4 features (sepal length, sepal width, petal length, petal width).
- **Hidden Layer 1**: 8 neurons (ReLU activation).
- **Hidden Layer 2**: 9 neurons (ReLU activation).
- **Output

   # Convolutional Neural Network (CNN) - MNIST

This Jupyter Notebook (`CNN.ipynb`) provides a step-by-step guide to building and training a Convolutional Neural Network (CNN) using the MNIST dataset.

## Overview
The notebook is structured into the following sections:

1. **Introduction** – Overview of CNN models and their applications.
2. **Imports** – Necessary libraries such as `torch`, `torchvision`, `matplotlib`, and `sklearn.metrics`.
3. **Data Preparation** – Loading and preprocessing the MNIST dataset.
4. **Model Definition** – Implementing a CNN with convolutional and fully connected layers.
5. **Training** – Defining the loss function, optimizer, and running the training loop.
6. **Evaluation** – Testing the model and measuring accuracy.
7. **Visualization** – Plotting training loss, accuracy, and predictions.

## CNN Architecture

- **Conv Layer 1**: 16 filters, 3x3 kernel, ReLU activation.
- **Conv Layer 2**: 32 filters, 3x3 kernel, ReLU activation.
- **Pooling**: Max pooling applied after each convolution.
- **Fully Connected Layers**: Extracts features before final classification.
- **Output Layer**: 10 classes (digits 0-9) with softmax activation.

## Requirements

Ensure the following dependencies are installed before running the notebook:

```bash
pip install torch torchvision matplotlib scikit-learn
```

## Usage

1. Open the Jupyter Notebook:

   ```bash
   jupyter notebook CNN.ipynb
   ```

2. Run the cells sequentially to train and evaluate the model.

## Results

The trained CNN achieves high accuracy on the MNIST dataset, and the notebook provides visualizations of loss, accuracy trends, and predictions.

## License

This project is open-source and available for educational purposes.


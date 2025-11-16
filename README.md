# 📘 CIFAR-10 Image Classification using TensorFlow/Keras

A complete deep-learning pipeline built to classify images from the CIFAR-10 dataset, featuring data preprocessing, CNN model creation, training with callbacks, performance visualization, and inference.
This project serves as a clean, reproducible template for image classification tasks.

🧩 Overview

This repository contains a fully implemented Convolutional Neural Network (CNN) trained on the CIFAR-10 dataset using TensorFlow/Keras.
The notebook includes:

Data loading and understanding

Visualization of dataset samples

Preprocessing (normalization, reshaping, splitting)

Model architecture creation

Model training with callbacks

Model performance evaluation

Predictions and visualization

Saved model loading & inference

It is suitable for students, researchers, and engineers who want a clean starter project for image classification.

🛠️ Technologies Used

Python 3.8+

TensorFlow / Keras

NumPy

Matplotlib

Scikit-learn

📦 Installation

Install necessary dependencies:

pip install tensorflow keras numpy matplotlib scikit-learn


If you're using Google Colab, all required libraries are already pre-installed except scikit-learn:

pip install scikit-learn

📊 Dataset: CIFAR-10

CIFAR-10 is a labeled dataset consisting of:

60,000 images (32×32 RGB)

10 classes

50,000 training images, 10,000 test images

Class names:
['airplane','automobile','bird','cat','deer',
 'dog','frog','horse','ship','truck']

🧪 Data Preparation

The notebook does the following:

Loads the dataset using keras.datasets.cifar10

Splits training data into train/validation

Normalizes pixel values from 0–255 to 0–1

Visualizes sample images with labels

Example:

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = X_train / 255.0
X_test = X_test / 255.0

🧠 Model Architecture

The CNN includes:

Multiple Conv2D layers

ReLU activation

MaxPooling2D layers

Dropout (to reduce overfitting)

Fully connected Dense layers

Final softmax layer for 10 classes

This architecture achieves high accuracy on CIFAR-10 with minimal compute.

🎛️ Callbacks Used

The training process is optimized using:

✔ ModelCheckpoint

Saves the best-performing model automatically.

✔ ReduceLROnPlateau

Reduces learning rate when validation loss stops improving.

✔ EarlyStopping

Stops training early and avoids overfitting.

Together, these improve performance and stability.

📈 Training & Evaluation

The notebook plots:

Accuracy Curves

Training accuracy

Validation accuracy

Loss Curves

Training loss

Validation loss

These help visualize overfitting and learning behavior.

🔍 Prediction Pipeline

The notebook includes a complete inference pipeline:

Load saved model (model.h5)

Preprocess a test image

Predict the class

Display the image with the predicted label

Example:

from keras.models import load_model

model = load_model("model.h5")
prediction = model.predict(img_array)

🖼️ Sample Visualization

The notebook shows:

Images from the dataset

Predictions + true labels

Misclassified examples (if enabled)

This helps understand model behavior beyond accuracy metrics.

💾 Saving & Loading the Model

Saving is done automatically via ModelCheckpoint.

To manually reload:

model = load_model("model.h5")

🚀 How to Use This Project
1. Clone or Download
git clone https://github.com/meddadaek/Image-Classifier-CIFAR-10.git

2. Open the Notebook

Use:

VS Code

Jupyter Notebook

Google Colab

3. Run all cells

This trains the model and generates evaluation visualizations.

4. Load the model for inference

Use the prediction section to classify new images.

🎯 Goals of This Project

Build a clean, understandable CNN classifier

Provide a reproducible workflow

Serve as a template for larger image classification projects

Introduce students to deep learning concepts step by step

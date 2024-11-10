# Binary Classifier for In-Vivo Microscopic Images for Esophageal Cancer Diagnosis

This repository contains a binary classifier that distinguishes between healthy and dysplastic/cancerous esophageal tissue using in-vivo microscopic images from the [Data Challenge by Mauna Kea](https://challenge.maunakeatech.com). A linear Perceptron model is used for the classification.

## Dataset Overview

This classifier uses a subset of the dataset containing only images of healthy tissue and images of tissue with dysplasia/cancer:
- **Class 0**: Healthy tissue (1,469 images)
- **Class 1**: Dysplastic or cancerous tissue (3,594 images)

The original images, which were 519x521 pixels, were scaled down to 260x260 pixels to optimize memory usage and reduce processing time.

### Data Files
- **ImageFolder.zip**: Contains the image dataset used for this classification task (stored outside this repository).
- **Data/ClasesImagenes.csv**: A CSV file listing each image's filename and its respective class label.

## Preprocessing

- **Resizing**: Images were resized to 260x260 pixels.
- **Flattening**: Each resized image was flattened into a 1D vector to be used as input for the Perceptron.

## Model Description

- **Algorithm**: A linear Perceptron was implemented using Scikit-learn.
- **Objective**: Perform binary classification of healthy vs. dysplastic/cancerous tissue images.
  
### Model Hyperparameters
- The Perceptron model was optimized for parameters such as:
  - `penalty`: L2 regularization
  - `alpha`: Regularization strength
  - `max_iter`: Maximum number of training iterations
  - `random_state`: Ensures reproducibility
  
## Evaluation Metrics

The model was evaluated using the following metrics:
- **Accuracy**: Measures the overall percentage of correct predictions.
- **Recall (Sensitivity)**: Assesses the model’s ability to correctly identify dysplastic/cancerous tissue.
- **Precision**: Determines the accuracy of predictions for cancerous cases.
- **F1-Score**: A balanced metric that combines recall and precision, particularly useful for imbalanced classes.

Cross-validation (5-fold) was performed to ensure the robustness of these metrics, and hyperparameter tuning was achieved through `GridSearchCV` to find the optimal values.

## Results

- **Training Accuracy**: X%
- **Test Accuracy**: Y%
- **F1-Score**: Z%
  
(Include specific metrics here after testing and validation)

## Usage

1. **Dependencies**: Install the required libraries.
   ```bash
   pip install -r requirements.txt

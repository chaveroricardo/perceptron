# Binary Classifier for In-Vivo Microscopic Images for Esophageal Cancer Diagnosis

This repository contains a binary classifier that distinguishes between healthy and dysplastic/cancerous esophageal tissue using in-vivo microscopic images from the [Data Challenge by Mauna Kea](https://challenge.maunakeatech.com). Multiple models, including Perceptron, Random Forest, and SVM, are used for classification, providing a comprehensive evaluation across different algorithms.

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
- **Flattening**: Each resized image was flattened into a 1D vector suitable for the models.

## Model Description

This project explores multiple models for classification, each implemented using Scikit-learn:
- **Perceptron**: A linear model with regularization optimized for binary classification.
- **Random Forest**: An ensemble method that combines multiple decision trees for robust classification.
- **SVM (Linear SVM and SGD Classifier)**: Linear Support Vector Machine and SGD Classifier with hinge loss, approximating SVM for high-dimensional data.
  
### Model Hyperparameters
Each model was tuned for optimal performance using parameters such as:
  - **Perceptron**:
    - `penalty`: L2 regularization
    - `alpha`: Regularization strength
    - `max_iter`: Maximum number of training iterations
  - **Random Forest**:
    - `n_estimators`: Number of trees
    - `max_depth`: Depth of each tree
  - **SVM**:
    - `C`: Regularization parameter (for Linear SVM)
    - `max_iter`: Maximum number of training iterations

## Evaluation Metrics

Each model was evaluated using the following metrics:
- **Accuracy**: Overall percentage of correct predictions.
- **Recall (Sensitivity)**: The model’s ability to correctly identify dysplastic/cancerous tissue.
- **Precision**: Accuracy of predictions for cancerous cases.
- **F1-Score**: A balanced metric combining recall and precision, useful for imbalanced classes.
- **ROC AUC**: Area under the ROC curve, indicating the model’s ability to discriminate between classes.

Cross-validation (5-fold) was used to ensure robustness. Hyperparameter tuning was achieved with `GridSearchCV` for optimal values.

## Results

For each model, the following metrics were obtained:
- **Training Accuracy**: X%
- **Test Accuracy**: Y%
- **F1-Score**: Z%
- **ROC AUC**: W%

(Include specific results for each model after testing and validation)

## Usage

1. **Dependencies**: Install the required libraries.
   ```bash
   pip install -r requirements.txt

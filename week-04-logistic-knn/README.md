# Week 4: Logistic Regression and K-Nearest Neighbors

## Overview

Implementation of two fundamental classification algorithms: Logistic Regression for probabilistic binary classification and K-Nearest Neighbors (KNN) for instance-based learning.

## Algorithms

**Logistic Regression:**
- Sigmoid function: σ(z) = 1 / (1 + e^(-z))
- Log-loss cost function
- Gradient descent optimization
- Probabilistic predictions

**K-Nearest Neighbors:**
- Distance-based classification
- Euclidean distance metric
- Majority voting for prediction
- No explicit training phase

## Datasets

**1. Banking Dataset**
- Binary classification: Customer subscription prediction
- Multiple features: Age, balance, campaign contacts
- Imbalanced classes handling

**2. Framingham Heart Study**
- Medical diagnosis: 10-year CHD risk prediction
- Clinical and lifestyle features
- Binary outcome classification

**3. Admission Prediction**
- University admission probability
- Features: GRE, TOEFL, CGPA, research experience
- Regression and classification variants

## Implementation Details

**Logistic Regression:**
- Feature scaling with StandardScaler
- Cross-entropy loss minimization
- Decision boundary visualization
- Probability threshold tuning

**KNN:**
- Optimal k selection using cross-validation
- Distance metric comparison
- Feature normalization importance
- Computational complexity analysis

## Results

Comparative analysis:
- Accuracy, Precision, Recall, F1-Score
- ROC curve and AUC
- Confusion matrix analysis
- Training vs inference time comparison

## Key Concepts

- Sigmoid activation function
- Maximum likelihood estimation
- Distance metrics in feature space
- Bias-variance tradeoff
- Curse of dimensionality (KNN)
- Model interpretability comparison

# Week 9: Multi-Layer Perceptron

## Overview

Implementation of Multi-Layer Perceptron (MLP) neural networks for both classification and regression tasks. Demonstrates deep learning fundamentals with backpropagation.

## Algorithm

MLP extends the Perceptron with hidden layers and non-linear activation functions:

Architecture: Input → Hidden Layers → Output

Forward propagation:
- h = activation(W1 · x + b1)
- y = activation(W2 · h + b2)

Backpropagation:
- Compute gradients via chain rule
- Update weights using gradient descent
- Minimize loss function (cross-entropy or MSE)

## Datasets

**1. Dry Bean Classification**
- 13,611 samples
- 16 geometric features (area, perimeter, shape factors)
- 7 bean types: SEKER, BARBUNYA, BOMBAY, CALI, HOROZ, SIRA, DERMASON
- Multi-class classification

**2. SAT-GPA Regression**
- 84 student records
- Feature: SAT score
- Target: College GPA (continuous)
- Regression task

## Implementation Details

**Classification (Dry Bean):**
- Architecture: 16-100-50-7 neurons
- Activation: ReLU (hidden), Softmax (output)
- Optimizer: Adam
- Loss: Cross-entropy
- Early stopping enabled
- L2 regularization (alpha=0.0001)

**Regression (SAT-GPA):**
- Architecture: 1-50-25-1 neurons
- Activation: ReLU (hidden), Linear (output)
- Optimizer: Adam
- Loss: Mean Squared Error
- Adaptive learning rate

## Results

**Classification Performance:**
- Test accuracy comparison across configurations
- Confusion matrix for 7 classes
- Per-class accuracy analysis
- Training loss convergence curves

**Regression Performance:**
- R² score, MSE, RMSE, MAE metrics
- Actual vs predicted scatter plots
- Residual analysis
- Comparison with linear regression

Model Comparisons:
- Small vs medium vs large networks
- ReLU vs Tanh activation
- MLP vs traditional ML algorithms

## Key Concepts

- Backpropagation algorithm
- Activation functions (ReLU, Tanh, Sigmoid)
- Gradient descent optimization (Adam, SGD)
- Overfitting prevention (early stopping, regularization)
- Hyperparameter tuning
- Deep learning fundamentals
- Universal approximation theorem
- Batch normalization concepts

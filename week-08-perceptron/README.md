# Week 8: Perceptron

## Overview

Implementation of the Perceptron algorithm, the foundation of neural networks. Applied to binary classification problems with linearly separable data.

## Algorithm

The Perceptron is a linear binary classifier:

f(x) = sign(w · x + b)

Learning rule:
- w = w + η * (y - ŷ) * x
- b = b + η * (y - ŷ)

Where η is the learning rate, y is true label, and ŷ is prediction.

## Datasets

**1. Sonar Dataset**
- 208 samples: Rock vs Mine classification
- 60 features: Sonar signal frequencies
- Binary classification challenge
- Non-trivial decision boundary

**2. Portfolio Analysis Dataset**
- Financial task classification
- Multiple numerical features
- Binary outcome prediction

## Implementation Details

- Weight initialization strategies
- Learning rate selection
- Convergence criteria
- Epoch-based training
- Decision boundary visualization
- Performance monitoring per epoch

## Results

Training analysis:
- Convergence behavior over epochs
- Final accuracy and error rate
- Weight vector interpretation
- Decision boundary geometry

Performance metrics:
- Classification accuracy
- Confusion matrix
- True positive/negative rates
- Training time analysis

## Key Concepts

- Linear separability requirement
- Online learning algorithm
- Weight update rule derivation
- Perceptron convergence theorem
- Limitations with non-linearly separable data
- Foundation for multi-layer neural networks
- Bias term importance

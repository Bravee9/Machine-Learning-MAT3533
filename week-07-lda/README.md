# Week 7: Linear Discriminant Analysis

## Overview

Implementation of Linear Discriminant Analysis (LDA) for supervised dimensionality reduction and classification. Applied to MNIST digit recognition and facial recognition tasks.

## Algorithm

LDA finds linear combinations of features that maximize class separation:

Objective: Maximize between-class variance while minimizing within-class variance

Steps:
1. Compute class means and overall mean
2. Calculate within-class scatter matrix (S_W)
3. Calculate between-class scatter matrix (S_B)
4. Solve generalized eigenvalue problem: S_B * v = λ * S_W * v
5. Project data onto discriminant axes

## Datasets

**1. MNIST Handwritten Digits**
- 70,000 samples (28x28 grayscale images)
- 784 original features
- 10 classes (digits 0-9)
- Reduced to 9 discriminant components (k-1 for k classes)

**2. Face Recognition Dataset**
- Multiple subjects with various expressions
- High-dimensional image data
- Identity classification task
- Eigenfaces approach comparison

## Implementation Details

- Feature standardization
- Scatter matrix computation
- Eigenvalue problem solving
- Dimensionality reduction: 784 to 9 components
- Classification using reduced features
- Visualization in 2D/3D discriminant space

## Results

Performance metrics:
- Classification accuracy before/after LDA
- Confusion matrix analysis
- Training time reduction
- Memory efficiency improvement

Visualization:
- 2D projection of classes
- Class separation quality
- Decision boundaries
- Misclassification analysis

Comparison with PCA:
- Supervised vs unsupervised reduction
- Class separation preservation
- Classification accuracy differences

## Key Concepts

- Supervised dimensionality reduction
- Fisher's linear discriminant
- Between-class and within-class scatter
- Generalized eigenvalue problem
- Maximum class separability
- Number of components limitation (k-1)

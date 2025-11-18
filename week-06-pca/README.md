# Week 6: Principal Component Analysis

## Overview

Implementation of Principal Component Analysis (PCA) for dimensionality reduction. Applied to high-dimensional speech feature data for Parkinson's disease detection.

## Algorithm

PCA transforms data into a new coordinate system where:
- First principal component has maximum variance
- Subsequent components have maximum remaining variance
- Components are orthogonal to each other

Steps:
1. Standardize the dataset
2. Compute covariance matrix
3. Calculate eigenvalues and eigenvectors
4. Sort components by explained variance
5. Project data onto principal components

## Dataset

**Parkinson's Disease Speech Features**
- Total samples: 756 voice recordings
- Original features: 754 speech characteristics
- Classes: Parkinson's patient (1) / Healthy control (0)
- High dimensionality challenge

## Implementation Details

- Feature standardization (zero mean, unit variance)
- Eigenvalue decomposition
- Scree plot for component selection
- Cumulative explained variance analysis
- Dimensionality reduction: 754 to optimal k components

## Results

Analysis:
- Explained variance ratio per component
- Cumulative variance preserved
- Optimal number of components selection
- Classification performance before/after PCA
- Visualization in 2D/3D principal component space

Benefits demonstrated:
- Reduced computational cost
- Noise filtering
- Visualization of high-dimensional data
- Comparable or improved classification accuracy

## Key Concepts

- Eigenvalue and eigenvector computation
- Variance maximization
- Linear transformation
- Curse of dimensionality mitigation
- Feature extraction vs feature selection
- Interpretability of principal components

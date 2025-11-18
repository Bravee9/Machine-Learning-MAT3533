# Week 5: Support Vector Machines and Decision Trees

## Algorithms

### 1. Support Vector Machines (SVM)
Maximum margin classifier with kernel methods for linear and non-linear decision boundaries.

### 2. Decision Trees
Tree-based classifier using recursive feature splitting. Non-parametric supervised learning method.

## Datasets

### 1. Glass Classification
- 214 glass samples
- 6 types of glass
- 9 chemical composition features
- Multi-class classification

### 2. Iris Dataset
- 150 iris flower samples
- 3 species
- 4 morphological features
- Classic ML benchmark dataset

### 3. MNIST Handwritten Digits
- 70,000 digit images (28x28 pixels)
- 10 classes (digits 0-9)
- 784 features per image
- Image classification benchmark

Files:
- `glass.csv`
- `iris.csv`
- `train-images.idx3-ubyte`
- `train-labels.idx1-ubyte`
- `t10k-images.idx3-ubyte`
- `t10k-labels.idx1-ubyte`

## Implementation

### Key Concepts

**Support Vector Machines:**
- Maximum margin hyperplane
- Kernel trick (linear, RBF, polynomial)
- Support vectors
- C parameter and regularization

**Decision Trees:**
- Information gain and Gini impurity
- Recursive partitioning
- Tree pruning
- Feature importance

### Methods
- Linear and non-linear SVM
- Kernel selection and tuning
- Decision tree depth optimization
- Ensemble methods introduction

## Results

- Multi-class classification accuracy
- Kernel comparison (linear vs RBF)
- Decision tree visualization
- Feature importance ranking
- MNIST classification performance

## Usage

```python
jupyter notebook svm_decision_tree.ipynb
```

## Key Learnings

- Kernel methods for non-linear problems
- SVM hyperparameter sensitivity
- Decision tree interpretability
- Overfitting prevention techniques
- High-dimensional data handling (MNIST)

# Machine Learning Fundamentals

Comprehensive implementation of fundamental machine learning algorithms from probability-based methods to deep neural networks.

## Overview

This repository contains 9 weeks of practical machine learning implementations covering core algorithms and techniques. Each week builds upon previous concepts, progressing from classical statistical methods to modern deep learning approaches.

Course materials from VNU University of Science, Hanoi - Machine Learning (MAT 3533) - Academic Year 2025-2026

## Course Structure

| Week | Topic | Algorithms | Dataset |
|------|-------|-----------|---------|
| 1 | Probability-based Classification | Gaussian Naive Bayes | Email Spam (4601 samples) |
| 2 | Binary Feature Classification | Bernoulli Naive Bayes | Medical Diagnosis (1000 cases) |
| 3 | Regression Analysis | Linear Regression | SAT-GPA, Fuel Consumption |
| 4 | Classification Methods | Logistic Regression, KNN | Banking, Admission Prediction |
| 5 | Advanced Classifiers | SVM, Decision Trees | Glass, Iris, MNIST |
| 6 | Dimensionality Reduction | Principal Component Analysis | Parkinson's Speech (754 features) |
| 7 | Linear Discriminant | LDA | MNIST Digits, Face Recognition |
| 8 | Neural Networks Basics | Perceptron | Sonar, Portfolio Analysis |
| 9 | Deep Learning | Multi-Layer Perceptron | Dry Bean (7 classes), SAT-GPA |

## Technical Stack

**Core Libraries:**
- Python 3.8+
- NumPy - Numerical computing
- Pandas - Data manipulation
- Matplotlib, Seaborn - Visualization
- Scikit-learn - Machine learning algorithms

## Installation

```bash
git clone https://github.com/Bravee9/Machine-Learning-MAT3533.git
cd Machine-Learning-MAT3533
pip install -r requirements.txt
```

## Project Structure

```
ML-Fundamentals/
├── week-01-naive-bayes/
│   ├── gaussian_nb.ipynb
│   ├── data/
│   └── README.md
├── week-02-bernoulli-nb/
│   ├── bernoulli_nb.ipynb
│   ├── data/
│   └── README.md
├── week-03-linear-regression/
│   ├── linear_regression.ipynb
│   ├── data/
│   └── README.md
├── week-04-logistic-knn/
│   ├── logistic_knn.ipynb
│   ├── data/
│   └── README.md
├── week-05-svm-decision-tree/
│   ├── svm_decision_tree.ipynb
│   ├── data/
│   └── README.md
├── week-06-pca/
│   ├── pca.ipynb
│   ├── data/
│   └── README.md
├── week-07-lda/
│   ├── lda.ipynb
│   ├── data/
│   └── README.md
├── week-08-perceptron/
│   ├── perceptron.ipynb
│   ├── data/
│   └── README.md
└── week-09-mlp/
    ├── mlp.ipynb
    ├── data/
    └── README.md
```

## Usage

Navigate to any week's directory and open the Jupyter notebook:

```bash
cd week-01-naive-bayes
jupyter notebook gaussian_nb.ipynb
```

Each notebook includes:
- Problem statement and dataset description
- Data exploration and visualization
- Implementation from scratch and using scikit-learn
- Model evaluation with multiple metrics
- Hyperparameter tuning
- Results visualization

## Key Concepts Covered

**Supervised Learning:**
- Classification (Binary and Multi-class)
- Regression (Linear and Non-linear)

**Unsupervised Learning:**
- Dimensionality Reduction
- Feature Extraction

**Model Evaluation:**
- Confusion Matrix
- Precision, Recall, F1-Score
- ROC-AUC Curves
- Cross-Validation
- Grid Search

**Feature Engineering:**
- Standardization and Normalization
- Feature Selection
- Principal Components
- Linear Discriminants

## Datasets

All datasets are included in respective week directories. Sources include:
- UCI Machine Learning Repository
- MNIST Handwritten Digits
- Custom medical and academic datasets
- Real-world classification and regression problems

## Results Highlights

**Classification Performance:**
- Gaussian NB: 98.2% accuracy on email spam detection
- SVM with RBF kernel: 99.1% on MNIST subset
- MLP: 92.5% on 7-class bean classification

**Regression Performance:**
- Linear Regression: R² = 0.76 on SAT-GPA prediction
- MLP Regressor: R² = 0.82, RMSE = 0.23

**Dimensionality Reduction:**
- PCA: 754 features reduced to 10 components (95% variance retained)
- LDA: 784 features reduced to 9 components for 10-class MNIST

## Learning Outcomes

- Implementation of ML algorithms from theoretical foundations
- Data preprocessing and feature engineering techniques
- Model selection and hyperparameter optimization
- Performance evaluation and interpretation
- Visualization of complex datasets and results
- Understanding trade-offs between different algorithms

## Requirements

See `requirements.txt` for detailed dependencies.

## License

MIT License

## Author

Bui Quang Chien
- GitHub: [@Bravee9](https://github.com/Bravee9)
- Student ID: 23001837

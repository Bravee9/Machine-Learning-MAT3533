<div align="center">

# Machine Learning — MAT3533

### Hanoi University of Science, VNU | 2025–2026

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

</div>

**English** | [Tiếng Việt](README.vi.md)

---

## Overview

Comprehensive, hands-on implementations of fundamental machine learning algorithms spanning **12 weeks of practical work** — from probability-based classifiers to deep neural networks and clustering techniques.

Course: **Machine Learning (MAT 3533)** — VNU University of Science, Hanoi | Academic Year 2025–2026.

Each week includes a Jupyter Notebook with fully documented implementations, covering theory and experiments on real-world datasets.

---

## Course Structure

| Week | Topic | Algorithms | Key Dataset |
|:----:|-------|-----------|-------------|
| **01** | Probability Classification | Gaussian Naive Bayes | Email Spam (4601 samples) |
| **02** | Binary Classification | Bernoulli Naive Bayes | Medical Diagnosis |
| **03** | Regression | Linear Regression | SAT-GPA Prediction |
| **04** | Classification Methods | Logistic Regression, KNN | Banking, Admission |
| **05** | Advanced Classifiers | SVM, Decision Trees | MNIST, Iris, Glass |
| **06** | Dimensionality Reduction | PCA | Parkinson's Speech (754 features) |
| **07** | Linear Discriminant | LDA | MNIST, Face Recognition |
| **08** | Neural Network Basics | Perceptron | Sonar (Rock vs Mine) |
| **09** | Deep Learning | Multi-Layer Perceptron | Dry Bean (7 classes) |
| **10** | Distance Clustering | K-Means, DBSCAN | MNIST, Synthetic Data |
| **11** | Probabilistic Clustering | Gaussian Mixture Model | Iris, Shopping Data |
| **12** | Hard Margin SVM | SVM with CVXOPT | Breast Cancer, Sonar |

---

## Highlights

**From-Scratch Implementations**
- Gaussian Naive Bayes — built from probability fundamentals
- Bernoulli Naive Bayes — binary feature classification
- Perceptron — single-layer neural network from scratch
- Multi-Layer Perceptron (MLP) — forward/back-propagation without frameworks
- K-Means & GMM — custom clustering implementations
- **Hard Margin SVM solved via CVXOPT quadratic programming** (Week 12)

**Notable Technical Work**
- PCA applied to the Parkinson's Speech dataset with **754 features** — dimensionality reduction to visualizable space
- MNIST digit classification tackled with multiple algorithms (SVM, LDA, K-Means, MLP)
- GMM with Expectation-Maximization on Iris and real shopping data

---

## Project Structure

```
├── week-01-naive-bayes/        Gaussian Naive Bayes — Email Spam
├── week-02-bernoulli-nb/       Bernoulli Naive Bayes — Medical Diagnosis
├── week-03-linear-regression/  Linear Regression — SAT-GPA
├── week-04-logistic-knn/       Logistic Regression & KNN — Banking, Admission
├── week-05-svm-decision-tree/  SVM & Decision Trees — MNIST, Iris, Glass
├── week-06-pca/                PCA — Parkinson's Speech (754 features)
├── week-07-lda/                LDA — MNIST, Face Recognition
├── week-08-perceptron/         Perceptron — Sonar (Rock vs Mine)
├── week-09-mlp/                Multi-Layer Perceptron — Dry Bean
├── week-10-kmean-dbscan/       K-Means & DBSCAN — MNIST, Synthetic Data
├── week-11-gmm/                Gaussian Mixture Model — Iris, Shopping
└── week-12-svm/                Hard Margin SVM (CVXOPT) — Breast Cancer, Sonar
```

---

## Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.8+ |
| Core Libraries | NumPy, Pandas, Matplotlib, Scikit-learn |
| Optimization | CVXOPT |
| Environment | Jupyter Notebook, Google Colab |

---

## Quick Start

```bash
git clone https://github.com/Bravee9/Machine-Learning-MAT3533.git
cd Machine-Learning-MAT3533
pip install -r requirements.txt
```

Open any notebook:
```bash
cd week-XX-topic
jupyter notebook notebook_name.ipynb
```

---

## Author

**Bùi Quang Chiến** — MSSV 23001837  
Computer Science, Hanoi University of Science — VNU  

[![GitHub](https://img.shields.io/badge/GitHub-Bravee9-181717?style=flat-square&logo=github)](https://github.com/Bravee9)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Brave9-0A66C2?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/brave9/)

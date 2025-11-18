# Machine Learning MAT3533

Comprehensive implementation of fundamental machine learning algorithms from probability-based methods to deep neural networks. Course materials from VNU University of Science, Hanoi Machine Learning course (MAT 3533) - Academic Year 2025-2026.

## 🎯 Overview

This repository provides a complete implementation of essential machine learning algorithms built from theoretical foundations. Each algorithm is implemented from scratch using NumPy to provide deep understanding of the underlying mathematics and principles.

## 📚 Features

### Supervised Learning
- **Probability-based Methods**
  - Naive Bayes Classifier (Gaussian)
  - Logistic Regression with regularization (L1, L2)

- **Classification Algorithms**
  - K-Nearest Neighbors (KNN)
  - Decision Trees (CART with Gini and Entropy)
  - Random Forests (Ensemble learning)
  - Support Vector Machines (SVM)

- **Regression Algorithms**
  - Linear Regression (Normal Equation & Gradient Descent)
  - Ridge Regression (L2 regularization)
  - Lasso Regression (L1 regularization)

### Unsupervised Learning
- **Clustering**
  - K-Means (with elbow method)
  - DBSCAN (density-based)
  - Hierarchical Clustering (agglomerative)

- **Dimensionality Reduction**
  - Principal Component Analysis (PCA)
  - t-SNE (t-Distributed Stochastic Neighbor Embedding)

### Neural Networks
- Feedforward Neural Networks
- Custom layer architecture (Dense, Activation)
- Multiple activation functions (Sigmoid, ReLU, Tanh, Softmax)
- Backpropagation algorithm
- Multiple loss functions (MSE, Cross-Entropy)

### Utilities
- **Data Preprocessing**
  - StandardScaler (z-score normalization)
  - MinMaxScaler (range scaling)
  - LabelEncoder
  - OneHotEncoder
  - Train-test split

- **Model Evaluation**
  - Accuracy, Precision, Recall, F1-Score
  - Confusion Matrix
  - Mean Squared Error (MSE), Mean Absolute Error (MAE)
  - R² Score
  - Classification Report

- **Model Selection**
  - K-Fold Cross-Validation
  - Grid Search CV
  - Randomized Search CV

- **Visualization**
  - Decision boundaries
  - Confusion matrices
  - Learning curves
  - Feature importance
  - Clustering results
  - PCA variance plots
  - ROC curves

## 🚀 Getting Started

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Bravee9/Machine-Learning-MAT3533.git
cd Machine-Learning-MAT3533
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Quick Start

Here's a simple example using Logistic Regression:

```python
from algorithms.supervised import LogisticRegression
from utils.preprocessing import StandardScaler, train_test_split
from utils.evaluation import accuracy_score

# Load your data
X, y = load_your_data()

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Preprocess
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LogisticRegression(learning_rate=0.01, n_iterations=1000)
model.fit(X_train_scaled, y_train)

# Evaluate
predictions = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy:.4f}")
```

## 📖 Examples

The `examples/` directory contains comprehensive demonstrations:

1. **Classification** - Multiple classification algorithms comparison
2. **Regression** - Linear regression with different methods
3. **Clustering** - Unsupervised learning techniques
4. **Dimensionality Reduction** - PCA and t-SNE
5. **Neural Networks** - Deep learning from scratch
6. **Model Selection** - Hyperparameter tuning and cross-validation

Run any example:
```bash
cd examples
python 01_classification_example.py
```

## 📁 Project Structure

```
Machine-Learning-MAT3533/
├── algorithms/
│   ├── supervised/
│   │   ├── naive_bayes.py
│   │   ├── logistic_regression.py
│   │   ├── linear_regression.py
│   │   ├── decision_tree.py
│   │   ├── random_forest.py
│   │   ├── knn.py
│   │   └── svm.py
│   ├── unsupervised/
│   │   ├── kmeans.py
│   │   ├── dbscan.py
│   │   ├── hierarchical.py
│   │   ├── pca.py
│   │   └── tsne.py
│   └── neural_networks/
│       ├── neural_network.py
│       ├── layers.py
│       └── activations.py
├── utils/
│   ├── preprocessing.py
│   ├── evaluation.py
│   ├── model_selection.py
│   └── visualization.py
├── examples/
│   ├── 01_classification_example.py
│   ├── 02_regression_example.py
│   ├── 03_clustering_example.py
│   ├── 04_dimensionality_reduction_example.py
│   ├── 05_neural_network_example.py
│   └── 06_model_selection_example.py
├── requirements.txt
└── README.md
```

## 🔬 Algorithm Implementations

### Key Features

- **From Scratch**: All algorithms implemented using NumPy for educational purposes
- **Well Documented**: Comprehensive docstrings with mathematical foundations
- **Production-Ready**: Efficient implementations with proper error handling
- **Flexible**: Easy to extend and customize for specific use cases

### Performance Considerations

- Algorithms are optimized for clarity and understanding
- For production use, consider scikit-learn or other optimized libraries
- Suitable for small to medium-sized datasets
- Great for learning and experimentation

## 🎓 Learning Resources

### Understanding Trade-offs

Each algorithm has strengths and weaknesses:

- **Naive Bayes**: Fast, works well with small data, assumes feature independence
- **Logistic Regression**: Interpretable, linear decision boundary
- **Decision Trees**: Interpretable, handles non-linear data, prone to overfitting
- **Random Forests**: Robust, reduces overfitting, less interpretable
- **KNN**: Simple, no training phase, computationally expensive for prediction
- **SVM**: Effective in high dimensions, good with clear margins
- **Neural Networks**: Powerful for complex patterns, requires more data

### When to Use Which Algorithm

**Classification**:
- Small dataset with independent features → Naive Bayes
- Linear separable data → Logistic Regression
- Non-linear data with feature importance → Decision Trees/Random Forests
- Complex patterns with large dataset → Neural Networks

**Regression**:
- Linear relationships → Linear Regression
- Need regularization → Ridge/Lasso
- Complex non-linear patterns → Neural Networks

**Clustering**:
- Known number of clusters, spherical clusters → K-Means
- Arbitrary shapes, noise handling → DBSCAN
- Hierarchical relationships → Hierarchical Clustering

**Dimensionality Reduction**:
- Linear relationships, interpretability → PCA
- Visualization, non-linear → t-SNE

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is available for educational purposes.

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.

## 🙏 Acknowledgments

- VNU University of Science, Hanoi
- Machine Learning course (MAT 3533)
- Academic Year 2025-2026

---

**Note**: This repository is designed for educational purposes to understand ML algorithms from theoretical foundations. For production use, consider using established libraries like scikit-learn, TensorFlow, or PyTorch.

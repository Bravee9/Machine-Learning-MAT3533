# Examples Directory

This directory contains comprehensive examples demonstrating the usage of machine learning algorithms implemented in this repository.

## Running the Examples

### Prerequisites

Install the required dependencies:
```bash
pip install -r ../requirements.txt
```

### Example Files

#### 1. Classification (`01_classification_example.py`)
Demonstrates supervised classification with multiple algorithms:
- Naive Bayes Classifier
- Logistic Regression
- K-Nearest Neighbors
- Decision Tree
- Random Forest

**Run:**
```bash
python 01_classification_example.py
```

#### 2. Regression (`02_regression_example.py`)
Demonstrates linear regression with different methods:
- Normal Equation (closed-form solution)
- Gradient Descent optimization
- Ridge Regression (L2 regularization)

**Run:**
```bash
python 02_regression_example.py
```

#### 3. Clustering (`03_clustering_example.py`)
Demonstrates unsupervised clustering algorithms:
- K-Means
- DBSCAN
- Hierarchical Clustering
- Elbow method for optimal K selection

**Run:**
```bash
python 03_clustering_example.py
```

#### 4. Dimensionality Reduction (`04_dimensionality_reduction_example.py`)
Demonstrates dimensionality reduction techniques:
- Principal Component Analysis (PCA)
- t-SNE visualization
- Variance analysis
- Reconstruction error

**Run:**
```bash
python 04_dimensionality_reduction_example.py
```

#### 5. Neural Networks (`05_neural_network_example.py`)
Demonstrates deep learning with feedforward neural networks:
- Multi-layer architecture
- Backpropagation training
- Multi-class classification
- Loss curve analysis

**Run:**
```bash
python 05_neural_network_example.py
```

#### 6. Model Selection (`06_model_selection_example.py`)
Demonstrates hyperparameter tuning and model selection:
- Cross-validation
- Grid Search
- Randomized Search
- Performance comparison

**Run:**
```bash
python 06_model_selection_example.py
```

## Output

Each example prints:
- Algorithm parameters and configuration
- Training progress
- Performance metrics (accuracy, MSE, R², etc.)
- Comparisons between different approaches
- Detailed results and insights

## Customization

Feel free to modify the examples to:
- Use different hyperparameters
- Try your own datasets
- Add visualization (uncomment plotting code)
- Experiment with different algorithms
- Compare performance metrics

## Learning Path

Recommended order for beginners:
1. Start with `01_classification_example.py` to understand supervised learning
2. Try `02_regression_example.py` for regression tasks
3. Explore `03_clustering_example.py` for unsupervised learning
4. Learn dimensionality reduction with `04_dimensionality_reduction_example.py`
5. Dive into neural networks with `05_neural_network_example.py`
6. Master model selection with `06_model_selection_example.py`

## Additional Resources

For more information on the algorithms and utilities:
- Check the source code in `../algorithms/`
- Review utility functions in `../utils/`
- Read the main README.md for algorithm theory

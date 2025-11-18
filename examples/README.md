# Examples Directory

This directory contains comprehensive examples demonstrating the usage of machine learning algorithms implemented in this repository.

## Running the Examples

### Prerequisites

Install the required dependencies:
```bash
pip install -r ../requirements.txt
```

### Running Examples

**From the root directory (recommended):**
```bash
python run_example.py 1  # Classification
python run_example.py 2  # Regression
python run_example.py 3  # Clustering
python run_example.py 4  # Dimensionality Reduction
python run_example.py 5  # Neural Networks
python run_example.py 6  # Model Selection
python run_example.py 7  # Complete Pipeline
```

**Or run the test suite:**
```bash
python test_implementations.py
```

### Example Files

All examples can be run using the `run_example.py` script from the root directory.

#### 1. Classification (`01_classification_example.py`)
Demonstrates supervised classification with multiple algorithms:
- Naive Bayes Classifier
- Logistic Regression
- K-Nearest Neighbors
- Decision Tree
- Random Forest

**Run:**
```bash
python run_example.py 1
```

#### 2. Regression (`02_regression_example.py`)
Demonstrates linear regression with different methods:
- Normal Equation (closed-form solution)
- Gradient Descent optimization
- Ridge Regression (L2 regularization)

**Run:**
```bash
python run_example.py 2
```

#### 3. Clustering (`03_clustering_example.py`)
Demonstrates unsupervised clustering algorithms:
- K-Means
- DBSCAN
- Hierarchical Clustering
- Elbow method for optimal K selection

**Run:**
```bash
python run_example.py 3
```

#### 4. Dimensionality Reduction (`04_dimensionality_reduction_example.py`)
Demonstrates dimensionality reduction techniques:
- Principal Component Analysis (PCA)
- t-SNE visualization
- Variance analysis
- Reconstruction error

**Run:**
```bash
python run_example.py 4
```

#### 5. Neural Networks (`05_neural_network_example.py`)
Demonstrates deep learning with feedforward neural networks:
- Multi-layer architecture
- Backpropagation training
- Multi-class classification
- Loss curve analysis

**Run:**
```bash
python run_example.py 5
```

#### 6. Model Selection (`06_model_selection_example.py`)
Demonstrates hyperparameter tuning and model selection:
- Cross-validation
- Grid Search
- Randomized Search
- Performance comparison

**Run:**
```bash
python run_example.py 6
```

#### 7. Complete Pipeline (`07_complete_pipeline_example.py`)
Demonstrates end-to-end machine learning workflow:
- Data generation and preprocessing
- Exploratory analysis with clustering
- Dimensionality reduction
- Multiple model training
- Cross-validation
- Hyperparameter tuning
- Final evaluation and metrics

**Run:**
```bash
python run_example.py 7
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

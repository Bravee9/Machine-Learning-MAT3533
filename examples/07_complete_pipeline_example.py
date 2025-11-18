"""
Complete ML Pipeline Example
Demonstrates end-to-end machine learning workflow
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath('..'))

from algorithms.supervised import LogisticRegression, RandomForestClassifier
from algorithms.unsupervised import PCA, KMeans
from algorithms.neural_networks import NeuralNetwork, DenseLayer, ActivationLayer
from utils.preprocessing import StandardScaler, OneHotEncoder, train_test_split
from utils.evaluation import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report
)
from utils.model_selection import cross_val_score, GridSearchCV

print("=" * 80)
print("COMPLETE MACHINE LEARNING PIPELINE")
print("=" * 80)

# ============================================================================
# 1. DATA GENERATION & PREPROCESSING
# ============================================================================
print("\n[STEP 1] Data Generation and Preprocessing")
print("-" * 80)

np.random.seed(42)

# Generate 4-class classification problem with 8 features
n_samples_per_class = 150
n_features = 8

# Create 4 distinct clusters
X_class0 = np.random.randn(n_samples_per_class, n_features) + np.array([2, 2, 0, 0, 0, 0, 0, 0])
X_class1 = np.random.randn(n_samples_per_class, n_features) + np.array([-2, 2, 0, 0, 0, 0, 0, 0])
X_class2 = np.random.randn(n_samples_per_class, n_features) + np.array([2, -2, 0, 0, 0, 0, 0, 0])
X_class3 = np.random.randn(n_samples_per_class, n_features) + np.array([-2, -2, 0, 0, 0, 0, 0, 0])

X = np.vstack([X_class0, X_class1, X_class2, X_class3])
y = np.concatenate([
    np.zeros(n_samples_per_class),
    np.ones(n_samples_per_class),
    np.ones(n_samples_per_class) * 2,
    np.ones(n_samples_per_class) * 3
]).astype(int)

print(f"Dataset created: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes")
print(f"Class distribution: {np.bincount(y)}")

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train set: {X_train.shape[0]} samples")
print(f"Test set:  {X_test.shape[0]} samples")

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("✓ Features standardized (mean=0, std=1)")

# ============================================================================
# 2. EXPLORATORY DATA ANALYSIS WITH CLUSTERING
# ============================================================================
print("\n[STEP 2] Exploratory Data Analysis - Clustering")
print("-" * 80)

# Apply K-Means to discover patterns
kmeans = KMeans(k=4, max_iters=100)
kmeans.fit(X_train_scaled)
cluster_labels = kmeans.labels

print(f"K-Means clustering:")
print(f"  Clusters found: {len(np.unique(cluster_labels))}")
print(f"  Cluster sizes: {np.bincount(cluster_labels)}")
print(f"  Inertia: {kmeans.inertia(X_train_scaled):.4f}")

# ============================================================================
# 3. DIMENSIONALITY REDUCTION
# ============================================================================
print("\n[STEP 3] Dimensionality Reduction - PCA")
print("-" * 80)

# Apply PCA
pca = PCA(n_components=4)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print(f"Original dimensions: {X_train_scaled.shape[1]}")
print(f"Reduced dimensions: {X_train_pca.shape[1]}")
print(f"Explained variance per component:")
for i, var in enumerate(pca.explained_variance_ratio, 1):
    print(f"  PC{i}: {var:.4f}")
print(f"Total variance explained: {np.sum(pca.explained_variance_ratio):.4f}")

# ============================================================================
# 4. MODEL TRAINING - MULTIPLE ALGORITHMS
# ============================================================================
print("\n[STEP 4] Model Training - Multiple Algorithms")
print("-" * 80)

models = {}

# 4.1 Logistic Regression
print("\n4.1 Logistic Regression")
lr = LogisticRegression(learning_rate=0.1, n_iterations=1000, regularization='l2', lambda_reg=0.01)
# Note: For multi-class, we'll train on binary for simplicity
y_train_binary = (y_train == 0).astype(int)
y_test_binary = (y_test == 0).astype(int)
lr.fit(X_train_scaled, y_train_binary)
y_pred_lr = lr.predict(X_test_scaled)
acc_lr = accuracy_score(y_test_binary, y_pred_lr)
print(f"  Accuracy: {acc_lr:.4f}")
models['Logistic Regression'] = acc_lr

# 4.2 Random Forest
print("\n4.2 Random Forest")
rf = RandomForestClassifier(n_trees=50, max_depth=8)
rf.fit(X_train_scaled, y_train)
y_pred_rf = rf.predict(X_test_scaled)
acc_rf = accuracy_score(y_test, y_pred_rf)
print(f"  Accuracy: {acc_rf:.4f}")
models['Random Forest'] = acc_rf

# 4.3 Neural Network
print("\n4.3 Neural Network")
nn = NeuralNetwork()
nn.add(DenseLayer(input_size=8, output_size=16))
nn.add(ActivationLayer('relu'))
nn.add(DenseLayer(input_size=16, output_size=8))
nn.add(ActivationLayer('relu'))
nn.add(DenseLayer(input_size=8, output_size=4))
nn.add(ActivationLayer('softmax'))

# One-hot encode labels for neural network
encoder = OneHotEncoder()
y_train_encoded = encoder.fit_transform(y_train)
y_test_encoded = encoder.transform(y_test)

nn.fit(X_train_scaled, y_train_encoded, epochs=100, learning_rate=0.1, 
       loss='cross_entropy', verbose=False)
y_pred_nn = nn.predict(X_test_scaled)
y_pred_nn_classes = np.argmax(y_pred_nn, axis=1)
acc_nn = accuracy_score(y_test, y_pred_nn_classes)
print(f"  Accuracy: {acc_nn:.4f}")
print(f"  Final loss: {nn.loss_history[-1]:.6f}")
models['Neural Network'] = acc_nn

# ============================================================================
# 5. MODEL SELECTION - CROSS-VALIDATION
# ============================================================================
print("\n[STEP 5] Model Selection - Cross-Validation")
print("-" * 80)

rf_cv = RandomForestClassifier(n_trees=30, max_depth=8)
cv_scores = cross_val_score(rf_cv, X_train_scaled, y_train, cv=5)
print(f"Random Forest 5-Fold CV scores: {cv_scores}")
print(f"Mean CV score: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")

# ============================================================================
# 6. HYPERPARAMETER TUNING
# ============================================================================
print("\n[STEP 6] Hyperparameter Tuning - Grid Search")
print("-" * 80)

param_grid = {
    'n_trees': [20, 30, 40],
    'max_depth': [5, 7, 10]
}

rf_base = RandomForestClassifier()
grid_search = GridSearchCV(rf_base, param_grid, cv=3, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)

print(f"Best parameters: {grid_search.best_params}")
print(f"Best CV score: {grid_search.best_score:.4f}")

y_pred_best = grid_search.predict(X_test_scaled)
acc_best = accuracy_score(y_test, y_pred_best)
print(f"Test accuracy with best params: {acc_best:.4f}")

# ============================================================================
# 7. FINAL EVALUATION
# ============================================================================
print("\n[STEP 7] Final Model Evaluation")
print("-" * 80)

# Use best Random Forest model
final_model = grid_search.best_model
y_pred_final = final_model.predict(X_test_scaled)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_final)
print("\nConfusion Matrix:")
print(cm)

# Detailed Metrics
print("\nClassification Metrics:")
report = classification_report(y_test, y_pred_final)
for key, value in report.items():
    if isinstance(value, dict):
        print(f"\nClass {key}:")
        for metric, score in value.items():
            print(f"  {metric}: {score:.4f}")
    else:
        print(f"\n{key}: {value:.4f}")

# ============================================================================
# 8. SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("PIPELINE SUMMARY")
print("=" * 80)

print("\nData Pipeline:")
print(f"  ✓ Generated {X.shape[0]} samples with {X.shape[1]} features")
print(f"  ✓ Standardized features")
print(f"  ✓ Split into train ({len(X_train)}) and test ({len(X_test)}) sets")

print("\nExploratory Analysis:")
print(f"  ✓ K-Means clustering revealed {len(np.unique(cluster_labels))} clusters")
print(f"  ✓ PCA reduced dimensions from {X.shape[1]} to 4 ({np.sum(pca.explained_variance_ratio):.2%} variance)")

print("\nModel Performance:")
for model_name, accuracy in sorted(models.items(), key=lambda x: x[1], reverse=True):
    print(f"  {model_name:25s}: {accuracy:.4f}")

print("\nBest Model (Random Forest after tuning):")
print(f"  Parameters: {grid_search.best_params}")
print(f"  Test Accuracy: {acc_best:.4f}")

print("\n" + "=" * 80)
print("COMPLETE ML PIPELINE EXECUTED SUCCESSFULLY!")
print("=" * 80)

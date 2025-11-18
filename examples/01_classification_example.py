"""
Example 1: Classification with Multiple Algorithms
Demonstrates supervised learning for classification tasks
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath('..'))

from algorithms.supervised import (
    NaiveBayesClassifier, 
    LogisticRegression, 
    DecisionTreeClassifier,
    KNearestNeighbors,
    RandomForestClassifier
)
from utils.preprocessing import StandardScaler, train_test_split
from utils.evaluation import accuracy_score, confusion_matrix, classification_report
from utils.visualization import plot_confusion_matrix, plot_decision_boundary

# Generate synthetic classification data
np.random.seed(42)

# Class 0: centered at (-2, -2)
X_class0 = np.random.randn(100, 2) + np.array([-2, -2])
y_class0 = np.zeros(100)

# Class 1: centered at (2, 2)
X_class1 = np.random.randn(100, 2) + np.array([2, 2])
y_class1 = np.ones(100)

# Combine data
X = np.vstack([X_class0, X_class1])
y = np.concatenate([y_class0, y_class1])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("=" * 70)
print("CLASSIFICATION EXAMPLE - Multiple Algorithms Comparison")
print("=" * 70)
print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Number of features: {X_train.shape[1]}")
print()

# Dictionary to store results
results = {}

# 1. Naive Bayes
print("\n1. Naive Bayes Classifier")
print("-" * 50)
nb = NaiveBayesClassifier()
nb.fit(X_train_scaled, y_train)
y_pred_nb = nb.predict(X_test_scaled)
acc_nb = accuracy_score(y_test, y_pred_nb)
print(f"Accuracy: {acc_nb:.4f}")
results['Naive Bayes'] = acc_nb

# 2. Logistic Regression
print("\n2. Logistic Regression")
print("-" * 50)
lr = LogisticRegression(learning_rate=0.1, n_iterations=1000)
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)
acc_lr = accuracy_score(y_test, y_pred_lr)
print(f"Accuracy: {acc_lr:.4f}")
results['Logistic Regression'] = acc_lr

# 3. K-Nearest Neighbors
print("\n3. K-Nearest Neighbors (k=5)")
print("-" * 50)
knn = KNearestNeighbors(k=5)
knn.fit(X_train_scaled, y_train)
y_pred_knn = knn.predict(X_test_scaled)
acc_knn = accuracy_score(y_test, y_pred_knn)
print(f"Accuracy: {acc_knn:.4f}")
results['KNN'] = acc_knn

# 4. Decision Tree
print("\n4. Decision Tree")
print("-" * 50)
dt = DecisionTreeClassifier(max_depth=5, min_samples_split=5)
dt.fit(X_train_scaled, y_train)
y_pred_dt = dt.predict(X_test_scaled)
acc_dt = accuracy_score(y_test, y_pred_dt)
print(f"Accuracy: {acc_dt:.4f}")
results['Decision Tree'] = acc_dt

# 5. Random Forest
print("\n5. Random Forest")
print("-" * 50)
rf = RandomForestClassifier(n_trees=50, max_depth=5)
rf.fit(X_train_scaled, y_train)
y_pred_rf = rf.predict(X_test_scaled)
acc_rf = accuracy_score(y_test, y_pred_rf)
print(f"Accuracy: {acc_rf:.4f}")
results['Random Forest'] = acc_rf

# Summary
print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)
for model_name, accuracy in sorted(results.items(), key=lambda x: x[1], reverse=True):
    print(f"{model_name:25s}: {accuracy:.4f}")

# Detailed metrics for best model
print("\n" + "=" * 70)
print("DETAILED METRICS - Logistic Regression")
print("=" * 70)
report = classification_report(y_test, y_pred_lr)
for key, value in report.items():
    if isinstance(value, dict):
        print(f"\nClass {key}:")
        for metric, score in value.items():
            print(f"  {metric}: {score:.4f}")
    else:
        print(f"\n{key}: {value:.4f}")

# Confusion matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred_lr)
print(cm)

print("\n" + "=" * 70)
print("Example completed successfully!")
print("=" * 70)

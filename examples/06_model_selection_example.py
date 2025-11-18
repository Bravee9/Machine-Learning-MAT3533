"""
Example 6: Model Selection and Hyperparameter Tuning
Demonstrates cross-validation and hyperparameter optimization
"""

import numpy as np
import sys
sys.path.append('..')

from algorithms.supervised import DecisionTreeClassifier, KNearestNeighbors
from utils.preprocessing import StandardScaler, train_test_split
from utils.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV

# Generate synthetic data
np.random.seed(42)

# Create classification problem
X_class0 = np.random.randn(100, 2) + np.array([-2, -2])
X_class1 = np.random.randn(100, 2) + np.array([2, 2])

X = np.vstack([X_class0, X_class1])
y = np.concatenate([np.zeros(100), np.ones(100)])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("=" * 70)
print("MODEL SELECTION AND HYPERPARAMETER TUNING")
print("=" * 70)
print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print()

# 1. Cross-Validation
print("\n1. Cross-Validation Evaluation")
print("-" * 50)

# Evaluate Decision Tree with cross-validation
dt = DecisionTreeClassifier(max_depth=5)
cv_scores_dt = cross_val_score(dt, X_train_scaled, y_train, cv=5)

print("Decision Tree (max_depth=5):")
print(f"  CV Scores: {cv_scores_dt}")
print(f"  Mean CV Score: {np.mean(cv_scores_dt):.4f} (+/- {np.std(cv_scores_dt):.4f})")

# Evaluate KNN with cross-validation
knn = KNearestNeighbors(k=5)
cv_scores_knn = cross_val_score(knn, X_train_scaled, y_train, cv=5)

print("\nK-Nearest Neighbors (k=5):")
print(f"  CV Scores: {cv_scores_knn}")
print(f"  Mean CV Score: {np.mean(cv_scores_knn):.4f} (+/- {np.std(cv_scores_knn):.4f})")

# 2. Grid Search
print("\n2. Grid Search for Decision Tree")
print("-" * 50)

param_grid = {
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 5, 10]
}

dt_base = DecisionTreeClassifier()
grid_search = GridSearchCV(dt_base, param_grid, cv=3, scoring='accuracy')

print("Parameter grid:")
for param, values in param_grid.items():
    print(f"  {param}: {values}")

print("\nSearching... (this may take a moment)")
grid_search.fit(X_train_scaled, y_train)

print(f"\nBest parameters: {grid_search.best_params}")
print(f"Best CV score: {grid_search.best_score:.4f}")

# Test best model
test_pred = grid_search.predict(X_test_scaled)
test_accuracy = np.mean(test_pred == y_test)
print(f"Test accuracy with best params: {test_accuracy:.4f}")

# 3. Randomized Search
print("\n3. Randomized Search for KNN")
print("-" * 50)

param_distributions = {
    'k': [3, 5, 7, 9, 11, 13, 15],
    'distance_metric': ['euclidean', 'manhattan']
}

knn_base = KNearestNeighbors()
random_search = RandomizedSearchCV(
    knn_base, 
    param_distributions, 
    n_iter=8, 
    cv=3, 
    scoring='accuracy',
    random_state=42
)

print("Parameter distributions:")
for param, values in param_distributions.items():
    print(f"  {param}: {values}")

print(f"\nNumber of iterations: 8")
print("Searching...")
random_search.fit(X_train_scaled, y_train)

print(f"\nBest parameters: {random_search.best_params}")
print(f"Best CV score: {random_search.best_score:.4f}")

# Test best model
test_pred = random_search.predict(X_test_scaled)
test_accuracy = np.mean(test_pred == y_test)
print(f"Test accuracy with best params: {test_accuracy:.4f}")

# Summary
print("\n" + "=" * 70)
print("SUMMARY - Model Selection Results")
print("=" * 70)
print(f"{'Method':<30} {'Best Params':<25} {'CV Score':<10}")
print("-" * 70)
print(f"{'Decision Tree (Grid)':<30} {str(grid_search.best_params):<25} {grid_search.best_score:.4f}")
print(f"{'KNN (Random)':<30} {str(random_search.best_params):<25} {random_search.best_score:.4f}")

print("\n" + "=" * 70)
print("Example completed successfully!")
print("=" * 70)

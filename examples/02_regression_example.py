"""
Example 2: Regression Analysis
Demonstrates linear regression and evaluation
"""

import numpy as np
import sys
sys.path.append('..')

from algorithms.supervised import LinearRegression
from utils.preprocessing import StandardScaler, train_test_split
from utils.evaluation import mean_squared_error, mean_absolute_error, r2_score

# Generate synthetic regression data
np.random.seed(42)
n_samples = 200

# True relationship: y = 3*x1 + 2*x2 + 1 + noise
X = np.random.randn(n_samples, 2)
true_weights = np.array([3, 2])
true_bias = 1
noise = np.random.randn(n_samples) * 0.5
y = X @ true_weights + true_bias + noise

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("=" * 70)
print("REGRESSION EXAMPLE - Linear Regression")
print("=" * 70)
print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Number of features: {X_train.shape[1]}")
print(f"\nTrue model: y = {true_weights[0]}*x1 + {true_weights[1]}*x2 + {true_bias}")
print()

# 1. Normal Equation
print("\n1. Linear Regression (Normal Equation)")
print("-" * 50)
lr_normal = LinearRegression(method='normal')
lr_normal.fit(X_train, y_train)
y_pred_normal = lr_normal.predict(X_test)

print(f"Learned weights: {lr_normal.weights}")
print(f"Learned bias: {lr_normal.bias:.4f}")
print(f"\nTest MSE: {mean_squared_error(y_test, y_pred_normal):.4f}")
print(f"Test MAE: {mean_absolute_error(y_test, y_pred_normal):.4f}")
print(f"Test R²: {r2_score(y_test, y_pred_normal):.4f}")

# 2. Gradient Descent
print("\n2. Linear Regression (Gradient Descent)")
print("-" * 50)
lr_gd = LinearRegression(method='gradient_descent', learning_rate=0.1, n_iterations=1000)
lr_gd.fit(X_train, y_train)
y_pred_gd = lr_gd.predict(X_test)

print(f"Learned weights: {lr_gd.weights}")
print(f"Learned bias: {lr_gd.bias:.4f}")
print(f"\nTest MSE: {mean_squared_error(y_test, y_pred_gd):.4f}")
print(f"Test MAE: {mean_absolute_error(y_test, y_pred_gd):.4f}")
print(f"Test R²: {r2_score(y_test, y_pred_gd):.4f}")

# 3. Ridge Regression (L2 regularization)
print("\n3. Ridge Regression (L2 Regularization)")
print("-" * 50)
lr_ridge = LinearRegression(method='normal', regularization='l2', lambda_reg=0.1)
lr_ridge.fit(X_train, y_train)
y_pred_ridge = lr_ridge.predict(X_test)

print(f"Learned weights: {lr_ridge.weights}")
print(f"Learned bias: {lr_ridge.bias:.4f}")
print(f"\nTest MSE: {mean_squared_error(y_test, y_pred_ridge):.4f}")
print(f"Test MAE: {mean_absolute_error(y_test, y_pred_ridge):.4f}")
print(f"Test R²: {r2_score(y_test, y_pred_ridge):.4f}")

# Comparison
print("\n" + "=" * 70)
print("COMPARISON")
print("=" * 70)
print(f"{'Method':<30} {'MSE':<12} {'MAE':<12} {'R²':<12}")
print("-" * 70)
print(f"{'Normal Equation':<30} {mean_squared_error(y_test, y_pred_normal):<12.4f} "
      f"{mean_absolute_error(y_test, y_pred_normal):<12.4f} {r2_score(y_test, y_pred_normal):<12.4f}")
print(f"{'Gradient Descent':<30} {mean_squared_error(y_test, y_pred_gd):<12.4f} "
      f"{mean_absolute_error(y_test, y_pred_gd):<12.4f} {r2_score(y_test, y_pred_gd):<12.4f}")
print(f"{'Ridge (L2)':<30} {mean_squared_error(y_test, y_pred_ridge):<12.4f} "
      f"{mean_absolute_error(y_test, y_pred_ridge):<12.4f} {r2_score(y_test, y_pred_ridge):<12.4f}")

print("\n" + "=" * 70)
print("Example completed successfully!")
print("=" * 70)

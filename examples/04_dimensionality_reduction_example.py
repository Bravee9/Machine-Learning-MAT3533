"""
Example 4: Dimensionality Reduction
Demonstrates PCA and t-SNE for visualization and feature reduction
"""

import numpy as np
import sys
sys.path.append('..')

from algorithms.unsupervised import PCA, TSNE
from utils.preprocessing import StandardScaler

# Generate high-dimensional data
np.random.seed(42)
n_samples = 150
n_features = 10

# Create data with some correlation structure
base = np.random.randn(n_samples, 3)
X = np.column_stack([
    base[:, 0] + np.random.randn(n_samples) * 0.1,
    base[:, 0] * 2 + np.random.randn(n_samples) * 0.1,
    base[:, 1] + np.random.randn(n_samples) * 0.1,
    base[:, 1] - base[:, 2] + np.random.randn(n_samples) * 0.1,
    base[:, 2] + np.random.randn(n_samples) * 0.1,
])

# Add random features
X = np.column_stack([X, np.random.randn(n_samples, n_features - 5)])

# Create labels for visualization
y = np.zeros(n_samples, dtype=int)
y[50:100] = 1
y[100:] = 2

# Standardize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("=" * 70)
print("DIMENSIONALITY REDUCTION EXAMPLE")
print("=" * 70)
print(f"Number of samples: {n_samples}")
print(f"Original number of features: {n_features}")
print(f"Number of classes: {len(np.unique(y))}")
print()

# 1. PCA
print("\n1. Principal Component Analysis (PCA)")
print("-" * 50)

# Full PCA
pca_full = PCA(n_components=n_features)
X_pca_full = pca_full.fit_transform(X_scaled)

print(f"Original shape: {X_scaled.shape}")
print(f"Transformed shape (all components): {X_pca_full.shape}")
print(f"\nExplained variance ratio:")
for i, var_ratio in enumerate(pca_full.explained_variance_ratio, 1):
    cumsum = np.sum(pca_full.explained_variance_ratio[:i])
    print(f"  PC{i}: {var_ratio:.4f} (cumulative: {cumsum:.4f})")

# Determine components for 95% variance
cumsum_variance = np.cumsum(pca_full.explained_variance_ratio)
n_components_95 = np.argmax(cumsum_variance >= 0.95) + 1
print(f"\nComponents needed for 95% variance: {n_components_95}")

# PCA to 2D for visualization
pca_2d = PCA(n_components=2)
X_pca_2d = pca_2d.fit_transform(X_scaled)
print(f"\n2D PCA shape: {X_pca_2d.shape}")
print(f"Variance explained by 2 components: {np.sum(pca_2d.explained_variance_ratio):.4f}")

# Reconstruction error
X_reconstructed = pca_2d.inverse_transform(X_pca_2d)
reconstruction_error = np.mean((X_scaled - X_reconstructed) ** 2)
print(f"Mean reconstruction error: {reconstruction_error:.6f}")

# 2. t-SNE (simplified, may take time)
print("\n2. t-SNE (t-Distributed Stochastic Neighbor Embedding)")
print("-" * 50)
print("Note: t-SNE is computationally intensive...")

tsne = TSNE(n_components=2, perplexity=30.0, learning_rate=200.0, n_iter=250)
X_tsne = tsne.fit_transform(X_scaled)

print(f"Original shape: {X_scaled.shape}")
print(f"t-SNE 2D shape: {X_tsne.shape}")

# Summary
print("\n" + "=" * 70)
print("DIMENSIONALITY REDUCTION SUMMARY")
print("=" * 70)
print(f"Method              Input Dim    Output Dim    Variance/Info")
print("-" * 70)
print(f"PCA (95% var)       {n_features:<12} {n_components_95:<13} 95% variance")
print(f"PCA (2D)            {n_features:<12} 2             {np.sum(pca_2d.explained_variance_ratio):.2%} variance")
print(f"t-SNE (2D)          {n_features:<12} 2             Local structure")

print("\n" + "=" * 70)
print("Example completed successfully!")
print("=" * 70)

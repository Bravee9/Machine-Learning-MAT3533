"""
Example 3: Clustering Analysis
Demonstrates unsupervised learning with K-Means, DBSCAN, and Hierarchical Clustering
"""

import numpy as np
import sys
sys.path.append('..')

from algorithms.unsupervised import KMeans, DBSCAN, HierarchicalClustering
from utils.preprocessing import StandardScaler

# Generate synthetic clustering data
np.random.seed(42)

# Create 3 clusters
cluster1 = np.random.randn(50, 2) + np.array([0, 0])
cluster2 = np.random.randn(50, 2) + np.array([5, 5])
cluster3 = np.random.randn(50, 2) + np.array([0, 5])

X = np.vstack([cluster1, cluster2, cluster3])

# Standardize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("=" * 70)
print("CLUSTERING EXAMPLE - Unsupervised Learning")
print("=" * 70)
print(f"Number of samples: {len(X)}")
print(f"Number of features: {X.shape[1]}")
print(f"True number of clusters: 3")
print()

# 1. K-Means Clustering
print("\n1. K-Means Clustering")
print("-" * 50)
kmeans = KMeans(k=3, max_iters=100)
kmeans.fit(X_scaled)
labels_kmeans = kmeans.labels

print(f"Converged in {kmeans.max_iters} iterations or less")
print(f"Final inertia: {kmeans.inertia(X_scaled):.4f}")
print(f"Cluster sizes: {np.bincount(labels_kmeans)}")
print(f"Centroids shape: {kmeans.centroids.shape}")

# 2. DBSCAN
print("\n2. DBSCAN Clustering")
print("-" * 50)
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan.fit(X_scaled)
labels_dbscan = dbscan.labels

n_clusters_dbscan = len(np.unique(labels_dbscan[labels_dbscan != -1]))
n_noise = np.sum(labels_dbscan == -1)

print(f"Number of clusters found: {n_clusters_dbscan}")
print(f"Number of noise points: {n_noise}")
if n_clusters_dbscan > 0:
    cluster_sizes = []
    for i in range(n_clusters_dbscan):
        cluster_sizes.append(np.sum(labels_dbscan == i))
    print(f"Cluster sizes: {cluster_sizes}")

# 3. Hierarchical Clustering
print("\n3. Hierarchical Clustering")
print("-" * 50)
hierarchical = HierarchicalClustering(n_clusters=3, linkage='average')
hierarchical.fit(X_scaled)
labels_hierarchical = hierarchical.labels

print(f"Number of clusters: 3")
print(f"Cluster sizes: {np.bincount(labels_hierarchical)}")

# Elbow method for K-Means
print("\n" + "=" * 70)
print("ELBOW METHOD - Finding Optimal K")
print("=" * 70)
inertias = []
K_range = range(1, 8)

for k in K_range:
    kmeans_temp = KMeans(k=k, max_iters=100)
    kmeans_temp.fit(X_scaled)
    inertias.append(kmeans_temp.inertia(X_scaled))

print("K\tInertia")
print("-" * 30)
for k, inertia in zip(K_range, inertias):
    print(f"{k}\t{inertia:.4f}")

# Comparison
print("\n" + "=" * 70)
print("CLUSTERING COMPARISON")
print("=" * 70)
print(f"Algorithm               Clusters    Unique Labels")
print("-" * 50)
print(f"K-Means                 3           {len(np.unique(labels_kmeans))}")
print(f"DBSCAN                  {n_clusters_dbscan}           {len(np.unique(labels_dbscan))}")
print(f"Hierarchical            3           {len(np.unique(labels_hierarchical))}")

print("\n" + "=" * 70)
print("Example completed successfully!")
print("=" * 70)

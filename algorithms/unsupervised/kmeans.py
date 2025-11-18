"""
K-Means Clustering
Centroid-based clustering algorithm
"""

import numpy as np


class KMeans:
    """
    K-Means Clustering Algorithm
    
    Partitions data into k clusters by minimizing within-cluster variance.
    Uses iterative expectation-maximization approach.
    
    Attributes:
        k (int): Number of clusters
        max_iters (int): Maximum number of iterations
        tol (float): Tolerance for convergence
        centroids (array): Cluster centroids
        labels (array): Cluster labels for training data
    """
    
    def __init__(self, k=3, max_iters=100, tol=1e-4):
        """
        Initialize K-Means
        
        Args:
            k (int): Number of clusters
            max_iters (int): Maximum iterations
            tol (float): Convergence tolerance
        """
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.centroids = None
        self.labels = None
        
    def fit(self, X):
        """
        Fit K-Means to data
        
        Args:
            X (array-like): Data to cluster, shape (n_samples, n_features)
            
        Returns:
            self: Fitted model
        """
        X = np.array(X)
        n_samples, n_features = X.shape
        
        # Initialize centroids randomly from data points
        random_indices = np.random.choice(n_samples, self.k, replace=False)
        self.centroids = X[random_indices]
        
        # Iterative optimization
        for _ in range(self.max_iters):
            # Assign samples to nearest centroid
            self.labels = self._assign_clusters(X)
            
            # Calculate new centroids
            new_centroids = self._compute_centroids(X, self.labels)
            
            # Check convergence
            if np.allclose(self.centroids, new_centroids, atol=self.tol):
                break
            
            self.centroids = new_centroids
        
        return self
    
    def _assign_clusters(self, X):
        """
        Assign each sample to nearest centroid
        
        Args:
            X: Data points
            
        Returns:
            array: Cluster labels
        """
        distances = np.zeros((X.shape[0], self.k))
        
        for i, centroid in enumerate(self.centroids):
            distances[:, i] = np.linalg.norm(X - centroid, axis=1)
        
        return np.argmin(distances, axis=1)
    
    def _compute_centroids(self, X, labels):
        """
        Compute new centroids as mean of assigned points
        
        Args:
            X: Data points
            labels: Current cluster assignments
            
        Returns:
            array: New centroids
        """
        centroids = np.zeros((self.k, X.shape[1]))
        
        for i in range(self.k):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                centroids[i] = np.mean(cluster_points, axis=0)
            else:
                # If cluster is empty, reinitialize randomly
                centroids[i] = X[np.random.choice(X.shape[0])]
        
        return centroids
    
    def predict(self, X):
        """
        Predict cluster labels for new data
        
        Args:
            X (array-like): Data to predict, shape (n_samples, n_features)
            
        Returns:
            array: Predicted cluster labels
        """
        X = np.array(X)
        return self._assign_clusters(X)
    
    def fit_predict(self, X):
        """
        Fit and predict in one step
        
        Args:
            X (array-like): Data to cluster
            
        Returns:
            array: Cluster labels
        """
        self.fit(X)
        return self.labels
    
    def inertia(self, X):
        """
        Calculate within-cluster sum of squares (inertia)
        
        Args:
            X (array-like): Data points
            
        Returns:
            float: Inertia value
        """
        X = np.array(X)
        labels = self.predict(X)
        inertia_value = 0
        
        for i in range(self.k):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                inertia_value += np.sum((cluster_points - self.centroids[i]) ** 2)
        
        return inertia_value

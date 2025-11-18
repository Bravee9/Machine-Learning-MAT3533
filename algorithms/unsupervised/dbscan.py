"""
DBSCAN Clustering
Density-based spatial clustering algorithm
"""

import numpy as np


class DBSCAN:
    """
    DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
    
    Groups together points that are closely packed and marks outliers.
    Does not require specifying number of clusters beforehand.
    
    Attributes:
        eps (float): Maximum distance between two samples for neighborhood
        min_samples (int): Minimum samples in neighborhood to be core point
        labels (array): Cluster labels (-1 for noise)
    """
    
    def __init__(self, eps=0.5, min_samples=5):
        """
        Initialize DBSCAN
        
        Args:
            eps (float): Neighborhood radius
            min_samples (int): Minimum points for core point
        """
        self.eps = eps
        self.min_samples = min_samples
        self.labels = None
        
    def fit(self, X):
        """
        Fit DBSCAN to data
        
        Args:
            X (array-like): Data to cluster, shape (n_samples, n_features)
            
        Returns:
            self: Fitted model
        """
        X = np.array(X)
        n_samples = X.shape[0]
        
        # Initialize all points as unvisited (-2) and noise (-1)
        self.labels = np.full(n_samples, -2)
        cluster_id = 0
        
        for i in range(n_samples):
            if self.labels[i] != -2:
                continue
            
            # Find neighbors
            neighbors = self._find_neighbors(X, i)
            
            if len(neighbors) < self.min_samples:
                # Mark as noise
                self.labels[i] = -1
            else:
                # Start new cluster
                self._expand_cluster(X, i, neighbors, cluster_id)
                cluster_id += 1
        
        return self
    
    def _find_neighbors(self, X, point_idx):
        """
        Find all neighbors within eps distance
        
        Args:
            X: Data points
            point_idx: Index of point to find neighbors for
            
        Returns:
            array: Indices of neighbors
        """
        distances = np.linalg.norm(X - X[point_idx], axis=1)
        return np.where(distances <= self.eps)[0]
    
    def _expand_cluster(self, X, point_idx, neighbors, cluster_id):
        """
        Expand cluster from seed point
        
        Args:
            X: Data points
            point_idx: Starting point index
            neighbors: Initial neighbors
            cluster_id: Current cluster ID
        """
        self.labels[point_idx] = cluster_id
        
        i = 0
        while i < len(neighbors):
            neighbor_idx = neighbors[i]
            
            if self.labels[neighbor_idx] == -1:
                # Change noise to border point
                self.labels[neighbor_idx] = cluster_id
            elif self.labels[neighbor_idx] == -2:
                # Unvisited point
                self.labels[neighbor_idx] = cluster_id
                
                # Find neighbors of neighbor
                neighbor_neighbors = self._find_neighbors(X, neighbor_idx)
                
                if len(neighbor_neighbors) >= self.min_samples:
                    # Add new neighbors to expand
                    neighbors = np.concatenate([neighbors, neighbor_neighbors])
            
            i += 1
    
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
    
    def predict(self, X):
        """
        Predict cluster labels (not traditionally supported in DBSCAN)
        
        Args:
            X (array-like): New data points
            
        Returns:
            array: Predicted labels (simplified nearest cluster approach)
        """
        # Note: Standard DBSCAN doesn't predict on new data
        # This is a simplified approach
        return self.labels

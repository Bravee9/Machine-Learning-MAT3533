"""
Hierarchical Clustering
Agglomerative clustering using distance linkage
"""

import numpy as np


class HierarchicalClustering:
    """
    Hierarchical Agglomerative Clustering
    
    Bottom-up approach that merges clusters based on linkage criterion.
    Supports multiple linkage methods.
    
    Attributes:
        n_clusters (int): Number of clusters to form
        linkage (str): Linkage criterion ('single', 'complete', 'average')
        labels (array): Cluster labels
    """
    
    def __init__(self, n_clusters=2, linkage='average'):
        """
        Initialize Hierarchical Clustering
        
        Args:
            n_clusters (int): Number of clusters
            linkage (str): Linkage method
        """
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.labels = None
        
    def fit(self, X):
        """
        Fit hierarchical clustering
        
        Args:
            X (array-like): Data to cluster, shape (n_samples, n_features)
            
        Returns:
            self: Fitted model
        """
        X = np.array(X)
        n_samples = X.shape[0]
        
        # Initialize each point as its own cluster
        clusters = [[i] for i in range(n_samples)]
        
        # Compute initial distance matrix
        distances = self._compute_distance_matrix(X)
        
        # Merge clusters until we have n_clusters
        while len(clusters) > self.n_clusters:
            # Find closest pair of clusters
            min_dist = float('inf')
            merge_i, merge_j = 0, 1
            
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    dist = self._cluster_distance(distances, clusters[i], clusters[j])
                    if dist < min_dist:
                        min_dist = dist
                        merge_i, merge_j = i, j
            
            # Merge clusters
            clusters[merge_i].extend(clusters[merge_j])
            del clusters[merge_j]
        
        # Assign labels
        self.labels = np.zeros(n_samples, dtype=int)
        for cluster_id, cluster in enumerate(clusters):
            for point_idx in cluster:
                self.labels[point_idx] = cluster_id
        
        return self
    
    def _compute_distance_matrix(self, X):
        """
        Compute pairwise distance matrix
        
        Args:
            X: Data points
            
        Returns:
            array: Distance matrix
        """
        n_samples = X.shape[0]
        distances = np.zeros((n_samples, n_samples))
        
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                dist = np.linalg.norm(X[i] - X[j])
                distances[i, j] = dist
                distances[j, i] = dist
        
        return distances
    
    def _cluster_distance(self, distances, cluster1, cluster2):
        """
        Calculate distance between two clusters
        
        Args:
            distances: Pairwise distance matrix
            cluster1: First cluster (list of indices)
            cluster2: Second cluster (list of indices)
            
        Returns:
            float: Distance between clusters
        """
        dists = []
        for i in cluster1:
            for j in cluster2:
                dists.append(distances[i, j])
        
        if self.linkage == 'single':
            return min(dists)
        elif self.linkage == 'complete':
            return max(dists)
        else:  # average
            return np.mean(dists)
    
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
        Return stored labels
        
        Args:
            X: Data points (not used)
            
        Returns:
            array: Cluster labels
        """
        return self.labels

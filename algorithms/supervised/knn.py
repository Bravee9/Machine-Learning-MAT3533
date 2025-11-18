"""
K-Nearest Neighbors
Instance-based learning algorithm
"""

import numpy as np
from collections import Counter


class KNearestNeighbors:
    """
    K-Nearest Neighbors Classifier
    
    Non-parametric algorithm that classifies based on k nearest training examples.
    Supports multiple distance metrics.
    
    Attributes:
        k (int): Number of neighbors to consider
        distance_metric (str): Distance metric ('euclidean', 'manhattan', 'minkowski')
        p (int): Power parameter for Minkowski distance
        X_train: Training features
        y_train: Training labels
    """
    
    def __init__(self, k=3, distance_metric='euclidean', p=2):
        """
        Initialize KNN
        
        Args:
            k (int): Number of neighbors
            distance_metric (str): Distance metric to use
            p (int): Power for Minkowski distance
        """
        self.k = k
        self.distance_metric = distance_metric
        self.p = p
        self.X_train = None
        self.y_train = None
        
    def fit(self, X, y):
        """
        Store training data
        
        Args:
            X (array-like): Training features, shape (n_samples, n_features)
            y (array-like): Training labels, shape (n_samples,)
            
        Returns:
            self: Fitted classifier
        """
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        return self
    
    def _calculate_distance(self, x1, x2):
        """
        Calculate distance between two points
        
        Args:
            x1: First point
            x2: Second point
            
        Returns:
            float: Distance
        """
        if self.distance_metric == 'euclidean':
            return np.sqrt(np.sum((x1 - x2) ** 2))
        elif self.distance_metric == 'manhattan':
            return np.sum(np.abs(x1 - x2))
        elif self.distance_metric == 'minkowski':
            return np.power(np.sum(np.abs(x1 - x2) ** self.p), 1 / self.p)
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
    
    def predict(self, X):
        """
        Predict class labels
        
        Args:
            X (array-like): Test features, shape (n_samples, n_features)
            
        Returns:
            array: Predicted labels
        """
        X = np.array(X)
        predictions = [self._predict_single(x) for x in X]
        return np.array(predictions)
    
    def _predict_single(self, x):
        """
        Predict label for single sample
        
        Args:
            x: Single sample
            
        Returns:
            Predicted label
        """
        # Calculate distances to all training samples
        distances = [self._calculate_distance(x, x_train) for x_train in self.X_train]
        
        # Get k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = self.y_train[k_indices]
        
        # Majority vote
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
    
    def score(self, X, y):
        """
        Calculate accuracy score
        
        Args:
            X (array-like): Test features
            y (array-like): True labels
            
        Returns:
            float: Accuracy score
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)

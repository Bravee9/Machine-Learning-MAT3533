"""
Random Forest Classifier
Ensemble of decision trees using bagging
"""

import numpy as np
from .decision_tree import DecisionTreeClassifier
from collections import Counter


class RandomForestClassifier:
    """
    Random Forest Classifier
    
    Ensemble learning method using multiple decision trees with bootstrap sampling.
    Combines predictions through majority voting.
    
    Attributes:
        n_trees (int): Number of trees in forest
        max_depth (int): Maximum depth of each tree
        min_samples_split (int): Minimum samples to split
        max_features (int): Maximum features to consider for splitting
        trees (list): List of decision trees
    """
    
    def __init__(self, n_trees=100, max_depth=10, min_samples_split=2, max_features=None):
        """
        Initialize Random Forest
        
        Args:
            n_trees (int): Number of trees
            max_depth (int): Maximum tree depth
            min_samples_split (int): Minimum samples to split
            max_features (int): Maximum features for splitting (None = sqrt(n_features))
        """
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []
        
    def fit(self, X, y):
        """
        Build the random forest
        
        Args:
            X (array-like): Training features, shape (n_samples, n_features)
            y (array-like): Training labels, shape (n_samples,)
            
        Returns:
            self: Fitted classifier
        """
        X = np.array(X)
        y = np.array(y)
        
        self.trees = []
        
        for _ in range(self.n_trees):
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split
            )
            
            # Bootstrap sampling
            X_sample, y_sample = self._bootstrap_sample(X, y)
            
            # Train tree
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
        
        return self
    
    def _bootstrap_sample(self, X, y):
        """
        Create bootstrap sample
        
        Args:
            X: Features
            y: Labels
            
        Returns:
            tuple: Bootstrap sample (X_sample, y_sample)
        """
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]
    
    def predict(self, X):
        """
        Predict class labels
        
        Args:
            X (array-like): Test features, shape (n_samples, n_features)
            
        Returns:
            array: Predicted labels
        """
        X = np.array(X)
        
        # Collect predictions from all trees
        tree_predictions = np.array([tree.predict(X) for tree in self.trees])
        
        # Majority vote for each sample
        predictions = []
        for i in range(X.shape[0]):
            votes = tree_predictions[:, i]
            most_common = Counter(votes).most_common(1)[0][0]
            predictions.append(most_common)
        
        return np.array(predictions)
    
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

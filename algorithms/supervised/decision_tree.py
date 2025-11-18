"""
Decision Tree Classifier
Tree-based classification using information gain and Gini impurity
"""

import numpy as np
from collections import Counter


class Node:
    """
    Node in decision tree
    
    Attributes:
        feature (int): Feature index for splitting
        threshold (float): Threshold value for splitting
        left (Node): Left child node
        right (Node): Right child node
        value: Predicted class for leaf nodes
    """
    
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        
    def is_leaf_node(self):
        return self.value is not None


class DecisionTreeClassifier:
    """
    Decision Tree Classifier
    
    Uses recursive binary splitting based on information gain.
    Implements pre-pruning through max_depth and min_samples_split.
    
    Attributes:
        max_depth (int): Maximum depth of tree
        min_samples_split (int): Minimum samples required to split
        criterion (str): Split criterion ('gini' or 'entropy')
        root (Node): Root node of the tree
    """
    
    def __init__(self, max_depth=10, min_samples_split=2, criterion='gini'):
        """
        Initialize Decision Tree
        
        Args:
            max_depth (int): Maximum tree depth
            min_samples_split (int): Minimum samples to split a node
            criterion (str): Splitting criterion ('gini' or 'entropy')
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.root = None
        
    def fit(self, X, y):
        """
        Build the decision tree
        
        Args:
            X (array-like): Training features, shape (n_samples, n_features)
            y (array-like): Training labels, shape (n_samples,)
            
        Returns:
            self: Fitted classifier
        """
        X = np.array(X)
        y = np.array(y)
        self.root = self._grow_tree(X, y)
        return self
    
    def _grow_tree(self, X, y, depth=0):
        """
        Recursively grow the decision tree
        
        Args:
            X: Features for current node
            y: Labels for current node
            depth: Current depth
            
        Returns:
            Node: Root of subtree
        """
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))
        
        # Stopping criteria
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
        
        # Find best split
        best_feature, best_threshold = self._best_split(X, y)
        
        # Create child splits
        left_idxs = X[:, best_feature] <= best_threshold
        right_idxs = X[:, best_feature] > best_threshold
        
        # Grow children
        left = self._grow_tree(X[left_idxs], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs], y[right_idxs], depth + 1)
        
        return Node(best_feature, best_threshold, left, right)
    
    def _best_split(self, X, y):
        """
        Find the best feature and threshold for splitting
        
        Args:
            X: Features
            y: Labels
            
        Returns:
            tuple: (best_feature, best_threshold)
        """
        best_gain = -1
        best_feature, best_threshold = None, None
        
        for feature_idx in range(X.shape[1]):
            thresholds = np.unique(X[:, feature_idx])
            
            for threshold in thresholds:
                gain = self._information_gain(X[:, feature_idx], y, threshold)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def _information_gain(self, X_column, y, threshold):
        """
        Calculate information gain from a split
        
        Args:
            X_column: Feature column
            y: Labels
            threshold: Split threshold
            
        Returns:
            float: Information gain
        """
        # Parent impurity
        parent_impurity = self._impurity(y)
        
        # Split
        left_idxs = X_column <= threshold
        right_idxs = X_column > threshold
        
        if len(y[left_idxs]) == 0 or len(y[right_idxs]) == 0:
            return 0
        
        # Weighted child impurity
        n = len(y)
        n_left, n_right = len(y[left_idxs]), len(y[right_idxs])
        impurity_left = self._impurity(y[left_idxs])
        impurity_right = self._impurity(y[right_idxs])
        child_impurity = (n_left / n) * impurity_left + (n_right / n) * impurity_right
        
        # Information gain
        return parent_impurity - child_impurity
    
    def _impurity(self, y):
        """
        Calculate impurity of labels
        
        Args:
            y: Labels
            
        Returns:
            float: Impurity measure
        """
        if self.criterion == 'gini':
            return self._gini(y)
        else:
            return self._entropy(y)
    
    def _gini(self, y):
        """
        Calculate Gini impurity
        
        Args:
            y: Labels
            
        Returns:
            float: Gini impurity
        """
        counter = Counter(y)
        impurity = 1.0
        for count in counter.values():
            p = count / len(y)
            impurity -= p ** 2
        return impurity
    
    def _entropy(self, y):
        """
        Calculate entropy
        
        Args:
            y: Labels
            
        Returns:
            float: Entropy
        """
        counter = Counter(y)
        entropy = 0.0
        for count in counter.values():
            p = count / len(y)
            if p > 0:
                entropy -= p * np.log2(p)
        return entropy
    
    def _most_common_label(self, y):
        """
        Get most common label
        
        Args:
            y: Labels
            
        Returns:
            Most common label
        """
        counter = Counter(y)
        return counter.most_common(1)[0][0]
    
    def predict(self, X):
        """
        Predict class labels
        
        Args:
            X (array-like): Test features, shape (n_samples, n_features)
            
        Returns:
            array: Predicted labels
        """
        X = np.array(X)
        return np.array([self._traverse_tree(x, self.root) for x in X])
    
    def _traverse_tree(self, x, node):
        """
        Traverse tree to make prediction
        
        Args:
            x: Single sample
            node: Current node
            
        Returns:
            Predicted label
        """
        if node.is_leaf_node():
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
    
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

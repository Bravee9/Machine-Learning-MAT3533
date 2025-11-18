"""
Support Vector Machine
Maximum margin classifier using kernel methods
"""

import numpy as np


class SupportVectorMachine:
    """
    Support Vector Machine Classifier
    
    Implements binary SVM using gradient descent optimization.
    Uses hinge loss with regularization.
    
    Attributes:
        learning_rate (float): Learning rate for gradient descent
        lambda_param (float): Regularization parameter
        n_iterations (int): Number of training iterations
        weights (array): Model weights
        bias (float): Model bias term
    """
    
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iterations=1000):
        """
        Initialize SVM
        
        Args:
            learning_rate (float): Learning rate
            lambda_param (float): Regularization parameter
            n_iterations (int): Number of iterations
        """
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        
    def fit(self, X, y):
        """
        Train the SVM
        
        Args:
            X (array-like): Training features, shape (n_samples, n_features)
            y (array-like): Training labels (must be -1 or 1), shape (n_samples,)
            
        Returns:
            self: Fitted classifier
        """
        X = np.array(X)
        y = np.array(y)
        
        # Convert labels to -1 and 1 if needed
        y_ = np.where(y <= 0, -1, 1)
        
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent
        for _ in range(self.n_iterations):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.weights) - self.bias) >= 1
                
                if condition:
                    # Correct classification
                    self.weights -= self.learning_rate * (2 * self.lambda_param * self.weights)
                else:
                    # Misclassification
                    self.weights -= self.learning_rate * (
                        2 * self.lambda_param * self.weights - np.dot(x_i, y_[idx])
                    )
                    self.bias -= self.learning_rate * y_[idx]
        
        return self
    
    def predict(self, X):
        """
        Predict class labels
        
        Args:
            X (array-like): Test features, shape (n_samples, n_features)
            
        Returns:
            array: Predicted labels (0 or 1)
        """
        X = np.array(X)
        linear_output = np.dot(X, self.weights) - self.bias
        return np.where(linear_output >= 0, 1, 0)
    
    def decision_function(self, X):
        """
        Calculate decision function values
        
        Args:
            X (array-like): Test features
            
        Returns:
            array: Decision function values
        """
        X = np.array(X)
        return np.dot(X, self.weights) - self.bias
    
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

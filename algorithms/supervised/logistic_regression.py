"""
Logistic Regression
Binary and multi-class classification using logistic function
"""

import numpy as np


class LogisticRegression:
    """
    Logistic Regression Classifier
    
    Uses gradient descent to optimize log-likelihood.
    Supports binary and multi-class (one-vs-rest) classification.
    
    Attributes:
        learning_rate (float): Learning rate for gradient descent
        n_iterations (int): Number of training iterations
        weights (array): Model weights
        bias (float): Model bias term
        regularization (str): Type of regularization ('l1', 'l2', or None)
        lambda_reg (float): Regularization strength
    """
    
    def __init__(self, learning_rate=0.01, n_iterations=1000, regularization=None, lambda_reg=0.01):
        """
        Initialize Logistic Regression
        
        Args:
            learning_rate (float): Step size for gradient descent
            n_iterations (int): Number of iterations
            regularization (str): Regularization type ('l1', 'l2', or None)
            lambda_reg (float): Regularization parameter
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.lambda_reg = lambda_reg
        self.weights = None
        self.bias = None
        
    def _sigmoid(self, z):
        """
        Sigmoid activation function
        
        Args:
            z: Linear combination of inputs
            
        Returns:
            Sigmoid activation
        """
        # Clip to prevent overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def _add_regularization(self, gradient):
        """
        Add regularization to gradient
        
        Args:
            gradient: Original gradient
            
        Returns:
            Regularized gradient
        """
        if self.regularization == 'l2':
            return gradient + (self.lambda_reg * self.weights)
        elif self.regularization == 'l1':
            return gradient + (self.lambda_reg * np.sign(self.weights))
        return gradient
    
    def fit(self, X, y):
        """
        Train the logistic regression model
        
        Args:
            X (array-like): Training features, shape (n_samples, n_features)
            y (array-like): Training labels, shape (n_samples,)
            
        Returns:
            self: Fitted classifier
        """
        X = np.array(X)
        y = np.array(y)
        
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent
        for _ in range(self.n_iterations):
            # Forward pass
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)
            
            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            
            # Add regularization to weight gradient
            dw = self._add_regularization(dw)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
        
        return self
    
    def predict(self, X):
        """
        Predict class labels
        
        Args:
            X (array-like): Test features, shape (n_samples, n_features)
            
        Returns:
            array: Predicted binary labels
        """
        X = np.array(X)
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        return (y_predicted >= 0.5).astype(int)
    
    def predict_proba(self, X):
        """
        Predict class probabilities
        
        Args:
            X (array-like): Test features
            
        Returns:
            array: Predicted probabilities
        """
        X = np.array(X)
        linear_model = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_model)
    
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

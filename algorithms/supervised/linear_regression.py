"""
Linear Regression
Regression using ordinary least squares and gradient descent
"""

import numpy as np


class LinearRegression:
    """
    Linear Regression Model
    
    Implements both closed-form (normal equation) and gradient descent solutions.
    Supports Ridge (L2) and Lasso (L1) regularization.
    
    Attributes:
        method (str): Optimization method ('normal' or 'gradient_descent')
        learning_rate (float): Learning rate for gradient descent
        n_iterations (int): Number of iterations for gradient descent
        regularization (str): Regularization type ('l1', 'l2', or None)
        lambda_reg (float): Regularization strength
        weights (array): Model weights
        bias (float): Model bias term
    """
    
    def __init__(self, method='normal', learning_rate=0.01, n_iterations=1000, 
                 regularization=None, lambda_reg=0.01):
        """
        Initialize Linear Regression
        
        Args:
            method (str): 'normal' for closed-form or 'gradient_descent'
            learning_rate (float): Learning rate for gradient descent
            n_iterations (int): Number of iterations for gradient descent
            regularization (str): 'l1', 'l2', or None
            lambda_reg (float): Regularization parameter
        """
        self.method = method
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.lambda_reg = lambda_reg
        self.weights = None
        self.bias = None
        
    def fit(self, X, y):
        """
        Train the linear regression model
        
        Args:
            X (array-like): Training features, shape (n_samples, n_features)
            y (array-like): Training targets, shape (n_samples,)
            
        Returns:
            self: Fitted model
        """
        X = np.array(X)
        y = np.array(y)
        
        if self.method == 'normal':
            self._fit_normal_equation(X, y)
        else:
            self._fit_gradient_descent(X, y)
        
        return self
    
    def _fit_normal_equation(self, X, y):
        """
        Fit using normal equation (closed-form solution)
        
        Args:
            X: Training features
            y: Training targets
        """
        n_samples, n_features = X.shape
        
        # Add bias term
        X_b = np.c_[np.ones((n_samples, 1)), X]
        
        if self.regularization == 'l2':
            # Ridge regression
            identity = np.eye(n_features + 1)
            identity[0, 0] = 0  # Don't regularize bias
            theta = np.linalg.inv(X_b.T @ X_b + self.lambda_reg * identity) @ X_b.T @ y
        else:
            # Ordinary least squares
            theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
        
        self.bias = theta[0]
        self.weights = theta[1:]
    
    def _fit_gradient_descent(self, X, y):
        """
        Fit using gradient descent
        
        Args:
            X: Training features
            y: Training targets
        """
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent
        for _ in range(self.n_iterations):
            # Predictions
            y_predicted = np.dot(X, self.weights) + self.bias
            
            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            
            # Add regularization
            if self.regularization == 'l2':
                dw += (self.lambda_reg / n_samples) * self.weights
            elif self.regularization == 'l1':
                dw += (self.lambda_reg / n_samples) * np.sign(self.weights)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def predict(self, X):
        """
        Make predictions
        
        Args:
            X (array-like): Test features, shape (n_samples, n_features)
            
        Returns:
            array: Predicted values
        """
        X = np.array(X)
        return np.dot(X, self.weights) + self.bias
    
    def score(self, X, y):
        """
        Calculate R² score
        
        Args:
            X (array-like): Test features
            y (array-like): True values
            
        Returns:
            float: R² score
        """
        y_pred = self.predict(X)
        ss_total = np.sum((y - np.mean(y)) ** 2)
        ss_residual = np.sum((y - y_pred) ** 2)
        return 1 - (ss_residual / ss_total)
    
    def mse(self, X, y):
        """
        Calculate mean squared error
        
        Args:
            X (array-like): Test features
            y (array-like): True values
            
        Returns:
            float: Mean squared error
        """
        y_pred = self.predict(X)
        return np.mean((y - y_pred) ** 2)

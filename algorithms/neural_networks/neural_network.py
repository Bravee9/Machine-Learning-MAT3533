"""
Neural Network
Feedforward neural network with backpropagation
"""

import numpy as np


class NeuralNetwork:
    """
    Feedforward Neural Network
    
    Multi-layer perceptron with customizable architecture.
    Uses backpropagation for training.
    
    Attributes:
        layers (list): List of network layers
        loss_history (list): Training loss history
    """
    
    def __init__(self):
        """
        Initialize neural network
        """
        self.layers = []
        self.loss_history = []
        
    def add(self, layer):
        """
        Add layer to network
        
        Args:
            layer: Layer to add (DenseLayer or ActivationLayer)
        """
        self.layers.append(layer)
        
    def forward(self, X):
        """
        Forward pass through network
        
        Args:
            X: Input data
            
        Returns:
            Network output
        """
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def backward(self, loss_gradient, learning_rate):
        """
        Backward pass through network
        
        Args:
            loss_gradient: Gradient of loss w.r.t. output
            learning_rate: Learning rate
        """
        gradient = loss_gradient
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient, learning_rate)
    
    def mse_loss(self, y_true, y_pred):
        """
        Mean squared error loss
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            tuple: (loss, gradient)
        """
        loss = np.mean((y_true - y_pred) ** 2)
        gradient = 2 * (y_pred - y_true) / y_true.shape[0]
        return loss, gradient
    
    def cross_entropy_loss(self, y_true, y_pred):
        """
        Cross-entropy loss
        
        Args:
            y_true: True labels (one-hot encoded)
            y_pred: Predicted probabilities
            
        Returns:
            tuple: (loss, gradient)
        """
        # Clip predictions to prevent log(0)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
        gradient = (y_pred - y_true) / y_true.shape[0]
        return loss, gradient
    
    def fit(self, X, y, epochs=100, learning_rate=0.01, loss='mse', verbose=False):
        """
        Train the neural network
        
        Args:
            X (array-like): Training features
            y (array-like): Training labels
            epochs (int): Number of training epochs
            learning_rate (float): Learning rate
            loss (str): Loss function ('mse' or 'cross_entropy')
            verbose (bool): Print training progress
            
        Returns:
            self: Fitted network
        """
        X = np.array(X)
        y = np.array(y)
        
        # Select loss function
        if loss == 'mse':
            loss_fn = self.mse_loss
        else:
            loss_fn = self.cross_entropy_loss
        
        self.loss_history = []
        
        for epoch in range(epochs):
            # Forward pass
            output = self.forward(X)
            
            # Compute loss
            loss_value, loss_gradient = loss_fn(y, output)
            self.loss_history.append(loss_value)
            
            # Backward pass
            self.backward(loss_gradient, learning_rate)
            
            if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch}/{epochs}, Loss: {loss_value:.6f}")
        
        return self
    
    def predict(self, X):
        """
        Make predictions
        
        Args:
            X (array-like): Input data
            
        Returns:
            array: Predictions
        """
        X = np.array(X)
        return self.forward(X)
    
    def score(self, X, y):
        """
        Calculate accuracy (for classification)
        
        Args:
            X (array-like): Test features
            y (array-like): True labels
            
        Returns:
            float: Accuracy score
        """
        predictions = self.predict(X)
        
        # For multi-class classification
        if len(y.shape) > 1 and y.shape[1] > 1:
            y_pred_classes = np.argmax(predictions, axis=1)
            y_true_classes = np.argmax(y, axis=1)
        else:
            y_pred_classes = (predictions > 0.5).astype(int).flatten()
            y_true_classes = y.flatten()
        
        return np.mean(y_pred_classes == y_true_classes)

"""
Neural Network Layers
Building blocks for neural networks
"""

import numpy as np
from .activations import ACTIVATIONS


class DenseLayer:
    """
    Fully connected (dense) layer
    
    Attributes:
        input_size (int): Number of input features
        output_size (int): Number of output units
        weights (array): Layer weights
        bias (array): Layer bias
        input (array): Cached input for backprop
        output (array): Cached output for backprop
    """
    
    def __init__(self, input_size, output_size):
        """
        Initialize dense layer
        
        Args:
            input_size (int): Input dimension
            output_size (int): Output dimension
        """
        self.input_size = input_size
        self.output_size = output_size
        
        # Xavier initialization
        limit = np.sqrt(6 / (input_size + output_size))
        self.weights = np.random.uniform(-limit, limit, (input_size, output_size))
        self.bias = np.zeros((1, output_size))
        
        self.input = None
        self.output = None
        
    def forward(self, input_data):
        """
        Forward pass
        
        Args:
            input_data: Input to layer
            
        Returns:
            Output of layer
        """
        self.input = input_data
        self.output = np.dot(input_data, self.weights) + self.bias
        return self.output
    
    def backward(self, output_gradient, learning_rate):
        """
        Backward pass
        
        Args:
            output_gradient: Gradient from next layer
            learning_rate: Learning rate
            
        Returns:
            Gradient to pass to previous layer
        """
        # Gradient w.r.t. weights
        weights_gradient = np.dot(self.input.T, output_gradient)
        
        # Gradient w.r.t. bias
        bias_gradient = np.sum(output_gradient, axis=0, keepdims=True)
        
        # Gradient w.r.t. input
        input_gradient = np.dot(output_gradient, self.weights.T)
        
        # Update parameters
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * bias_gradient
        
        return input_gradient


class ActivationLayer:
    """
    Activation layer
    
    Attributes:
        activation_name (str): Name of activation function
        activation (function): Activation function
        activation_derivative (function): Derivative of activation
        input (array): Cached input
        output (array): Cached output
    """
    
    def __init__(self, activation_name):
        """
        Initialize activation layer
        
        Args:
            activation_name (str): Name of activation ('sigmoid', 'relu', 'tanh', 'softmax')
        """
        self.activation_name = activation_name
        self.activation, self.activation_derivative = ACTIVATIONS[activation_name]
        self.input = None
        self.output = None
        
    def forward(self, input_data):
        """
        Forward pass
        
        Args:
            input_data: Input to layer
            
        Returns:
            Activated output
        """
        self.input = input_data
        self.output = self.activation(input_data)
        return self.output
    
    def backward(self, output_gradient, learning_rate):
        """
        Backward pass
        
        Args:
            output_gradient: Gradient from next layer
            learning_rate: Not used in activation layer
            
        Returns:
            Gradient to pass to previous layer
        """
        return output_gradient * self.activation_derivative(self.output)

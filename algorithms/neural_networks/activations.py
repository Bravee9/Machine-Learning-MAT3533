"""
Activation Functions
Common activation functions and their derivatives
"""

import numpy as np


def sigmoid(x):
    """
    Sigmoid activation function
    
    Args:
        x: Input
        
    Returns:
        Sigmoid output
    """
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def sigmoid_derivative(x):
    """
    Derivative of sigmoid
    
    Args:
        x: Input (output of sigmoid)
        
    Returns:
        Derivative
    """
    return x * (1 - x)


def relu(x):
    """
    ReLU activation function
    
    Args:
        x: Input
        
    Returns:
        ReLU output
    """
    return np.maximum(0, x)


def relu_derivative(x):
    """
    Derivative of ReLU
    
    Args:
        x: Input
        
    Returns:
        Derivative
    """
    return (x > 0).astype(float)


def tanh(x):
    """
    Hyperbolic tangent activation
    
    Args:
        x: Input
        
    Returns:
        tanh output
    """
    return np.tanh(x)


def tanh_derivative(x):
    """
    Derivative of tanh
    
    Args:
        x: Input (output of tanh)
        
    Returns:
        Derivative
    """
    return 1 - x ** 2


def softmax(x):
    """
    Softmax activation for multi-class classification
    
    Args:
        x: Input
        
    Returns:
        Softmax probabilities
    """
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def softmax_derivative(x):
    """
    Derivative of softmax (simplified)
    
    Args:
        x: Input
        
    Returns:
        Derivative
    """
    return x * (1 - x)


# Dictionary mapping activation names to functions
ACTIVATIONS = {
    'sigmoid': (sigmoid, sigmoid_derivative),
    'relu': (relu, relu_derivative),
    'tanh': (tanh, tanh_derivative),
    'softmax': (softmax, softmax_derivative)
}

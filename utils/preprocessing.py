"""
Data Preprocessing Utilities
Feature scaling, encoding, and transformation
"""

import numpy as np


class StandardScaler:
    """
    Standardize features by removing mean and scaling to unit variance
    
    z = (x - μ) / σ
    
    Attributes:
        mean (array): Mean of training data
        std (array): Standard deviation of training data
    """
    
    def __init__(self):
        self.mean = None
        self.std = None
        
    def fit(self, X):
        """
        Compute mean and std for scaling
        
        Args:
            X (array-like): Training data
            
        Returns:
            self: Fitted scaler
        """
        X = np.array(X)
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        # Avoid division by zero
        self.std[self.std == 0] = 1
        return self
    
    def transform(self, X):
        """
        Standardize data
        
        Args:
            X (array-like): Data to transform
            
        Returns:
            array: Standardized data
        """
        X = np.array(X)
        return (X - self.mean) / self.std
    
    def fit_transform(self, X):
        """
        Fit and transform in one step
        
        Args:
            X (array-like): Data
            
        Returns:
            array: Standardized data
        """
        self.fit(X)
        return self.transform(X)


class MinMaxScaler:
    """
    Scale features to a given range [min, max]
    
    X_scaled = (X - X_min) / (X_max - X_min)
    
    Attributes:
        min (array): Minimum values of training data
        max (array): Maximum values of training data
        feature_range (tuple): Desired range of transformed data
    """
    
    def __init__(self, feature_range=(0, 1)):
        """
        Initialize MinMaxScaler
        
        Args:
            feature_range (tuple): Desired range (min, max)
        """
        self.min = None
        self.max = None
        self.feature_range = feature_range
        
    def fit(self, X):
        """
        Compute min and max for scaling
        
        Args:
            X (array-like): Training data
            
        Returns:
            self: Fitted scaler
        """
        X = np.array(X)
        self.min = np.min(X, axis=0)
        self.max = np.max(X, axis=0)
        # Avoid division by zero
        self.max = np.where(self.max == self.min, self.min + 1, self.max)
        return self
    
    def transform(self, X):
        """
        Scale data to feature_range
        
        Args:
            X (array-like): Data to transform
            
        Returns:
            array: Scaled data
        """
        X = np.array(X)
        X_std = (X - self.min) / (self.max - self.min)
        return X_std * (self.feature_range[1] - self.feature_range[0]) + self.feature_range[0]
    
    def fit_transform(self, X):
        """
        Fit and transform in one step
        
        Args:
            X (array-like): Data
            
        Returns:
            array: Scaled data
        """
        self.fit(X)
        return self.transform(X)


class LabelEncoder:
    """
    Encode categorical labels as integers
    
    Attributes:
        classes (array): Unique class labels
        class_to_index (dict): Mapping from class to index
    """
    
    def __init__(self):
        self.classes = None
        self.class_to_index = {}
        
    def fit(self, y):
        """
        Fit label encoder
        
        Args:
            y (array-like): Labels
            
        Returns:
            self: Fitted encoder
        """
        self.classes = np.unique(y)
        self.class_to_index = {cls: idx for idx, cls in enumerate(self.classes)}
        return self
    
    def transform(self, y):
        """
        Transform labels to integers
        
        Args:
            y (array-like): Labels to transform
            
        Returns:
            array: Encoded labels
        """
        return np.array([self.class_to_index[label] for label in y])
    
    def fit_transform(self, y):
        """
        Fit and transform in one step
        
        Args:
            y (array-like): Labels
            
        Returns:
            array: Encoded labels
        """
        self.fit(y)
        return self.transform(y)
    
    def inverse_transform(self, y):
        """
        Transform integers back to original labels
        
        Args:
            y (array-like): Encoded labels
            
        Returns:
            array: Original labels
        """
        return np.array([self.classes[idx] for idx in y])


class OneHotEncoder:
    """
    One-hot encode categorical features
    
    Attributes:
        n_classes (int): Number of classes
    """
    
    def __init__(self):
        self.n_classes = None
        
    def fit(self, y):
        """
        Fit encoder
        
        Args:
            y (array-like): Labels
            
        Returns:
            self: Fitted encoder
        """
        self.n_classes = len(np.unique(y))
        return self
    
    def transform(self, y):
        """
        Transform labels to one-hot encoding
        
        Args:
            y (array-like): Labels
            
        Returns:
            array: One-hot encoded labels
        """
        y = np.array(y)
        one_hot = np.zeros((len(y), self.n_classes))
        one_hot[np.arange(len(y)), y] = 1
        return one_hot
    
    def fit_transform(self, y):
        """
        Fit and transform in one step
        
        Args:
            y (array-like): Labels
            
        Returns:
            array: One-hot encoded labels
        """
        self.fit(y)
        return self.transform(y)


def train_test_split(X, y, test_size=0.2, random_state=None):
    """
    Split data into train and test sets
    
    Args:
        X (array-like): Features
        y (array-like): Labels
        test_size (float): Proportion of test set
        random_state (int): Random seed
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    X = np.array(X)
    y = np.array(y)
    
    n_samples = len(X)
    n_test = int(n_samples * test_size)
    
    # Shuffle indices
    indices = np.random.permutation(n_samples)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

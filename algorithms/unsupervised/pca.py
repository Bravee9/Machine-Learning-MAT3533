"""
Principal Component Analysis (PCA)
Linear dimensionality reduction using eigendecomposition
"""

import numpy as np


class PCA:
    """
    Principal Component Analysis
    
    Reduces dimensionality by projecting data onto principal components.
    Uses eigendecomposition of covariance matrix.
    
    Attributes:
        n_components (int): Number of components to keep
        components (array): Principal components
        mean (array): Mean of training data
        explained_variance (array): Variance explained by each component
        explained_variance_ratio (array): Proportion of variance explained
    """
    
    def __init__(self, n_components=2):
        """
        Initialize PCA
        
        Args:
            n_components (int): Number of principal components
        """
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.explained_variance = None
        self.explained_variance_ratio = None
        
    def fit(self, X):
        """
        Fit PCA to data
        
        Args:
            X (array-like): Data, shape (n_samples, n_features)
            
        Returns:
            self: Fitted model
        """
        X = np.array(X)
        
        # Center the data
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        
        # Compute covariance matrix
        cov_matrix = np.cov(X_centered.T)
        
        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        
        # Sort eigenvectors by eigenvalues in descending order
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Store principal components
        self.components = eigenvectors[:, :self.n_components].T
        
        # Calculate explained variance
        self.explained_variance = eigenvalues[:self.n_components]
        total_var = np.sum(eigenvalues)
        self.explained_variance_ratio = self.explained_variance / total_var
        
        return self
    
    def transform(self, X):
        """
        Project data onto principal components
        
        Args:
            X (array-like): Data to transform
            
        Returns:
            array: Transformed data
        """
        X = np.array(X)
        X_centered = X - self.mean
        return np.dot(X_centered, self.components.T)
    
    def fit_transform(self, X):
        """
        Fit and transform in one step
        
        Args:
            X (array-like): Data
            
        Returns:
            array: Transformed data
        """
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X_transformed):
        """
        Transform data back to original space
        
        Args:
            X_transformed: Transformed data
            
        Returns:
            array: Reconstructed data
        """
        return np.dot(X_transformed, self.components) + self.mean

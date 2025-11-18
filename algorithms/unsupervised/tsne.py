"""
t-SNE (t-Distributed Stochastic Neighbor Embedding)
Nonlinear dimensionality reduction for visualization
"""

import numpy as np


class TSNE:
    """
    t-SNE Algorithm
    
    Reduces dimensionality while preserving local structure.
    Useful for visualization of high-dimensional data.
    
    Attributes:
        n_components (int): Target dimensionality (typically 2 or 3)
        perplexity (float): Related to number of nearest neighbors
        learning_rate (float): Learning rate for optimization
        n_iter (int): Number of iterations
        embedding (array): Low-dimensional embedding
    """
    
    def __init__(self, n_components=2, perplexity=30.0, learning_rate=200.0, n_iter=1000):
        """
        Initialize t-SNE
        
        Args:
            n_components (int): Target dimensions
            perplexity (float): Perplexity parameter
            learning_rate (float): Learning rate
            n_iter (int): Number of iterations
        """
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.embedding = None
        
    def fit_transform(self, X):
        """
        Fit t-SNE and return embedding
        
        Args:
            X (array-like): Data, shape (n_samples, n_features)
            
        Returns:
            array: Low-dimensional embedding
        """
        X = np.array(X)
        n_samples = X.shape[0]
        
        # Compute pairwise distances
        distances = self._compute_pairwise_distances(X)
        
        # Compute high-dimensional probabilities
        P = self._compute_joint_probabilities(distances)
        
        # Initialize low-dimensional embedding randomly
        self.embedding = np.random.randn(n_samples, self.n_components) * 1e-4
        
        # Gradient descent optimization
        for iteration in range(self.n_iter):
            # Compute low-dimensional probabilities
            Q = self._compute_low_dim_probabilities(self.embedding)
            
            # Compute gradient
            gradient = self._compute_gradient(P, Q, self.embedding)
            
            # Update embedding
            self.embedding -= self.learning_rate * gradient
            
            # Early exaggeration for first iterations
            if iteration < 250:
                P_adjusted = P * 4
            else:
                P_adjusted = P
        
        return self.embedding
    
    def _compute_pairwise_distances(self, X):
        """
        Compute pairwise Euclidean distances
        
        Args:
            X: Data points
            
        Returns:
            array: Distance matrix
        """
        n_samples = X.shape[0]
        distances = np.zeros((n_samples, n_samples))
        
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                dist = np.linalg.norm(X[i] - X[j])
                distances[i, j] = dist
                distances[j, i] = dist
        
        return distances
    
    def _compute_joint_probabilities(self, distances):
        """
        Compute joint probabilities in high-dimensional space
        
        Args:
            distances: Pairwise distances
            
        Returns:
            array: Joint probability matrix
        """
        n_samples = distances.shape[0]
        P = np.zeros((n_samples, n_samples))
        
        # Simplified version using fixed perplexity
        beta = 1.0  # Precision parameter
        
        for i in range(n_samples):
            # Compute conditional probabilities
            diff = distances[i] ** 2
            diff[i] = 0
            P[i] = np.exp(-diff * beta)
            P[i] = P[i] / np.sum(P[i])
        
        # Symmetrize
        P = (P + P.T) / (2 * n_samples)
        P = np.maximum(P, 1e-12)
        
        return P
    
    def _compute_low_dim_probabilities(self, Y):
        """
        Compute joint probabilities in low-dimensional space
        
        Args:
            Y: Low-dimensional embedding
            
        Returns:
            array: Joint probability matrix
        """
        n_samples = Y.shape[0]
        distances = np.zeros((n_samples, n_samples))
        
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                dist = np.sum((Y[i] - Y[j]) ** 2)
                distances[i, j] = dist
                distances[j, i] = dist
        
        # Student t-distribution with df=1
        Q = 1 / (1 + distances)
        np.fill_diagonal(Q, 0)
        Q = Q / np.sum(Q)
        Q = np.maximum(Q, 1e-12)
        
        return Q
    
    def _compute_gradient(self, P, Q, Y):
        """
        Compute gradient of KL divergence
        
        Args:
            P: High-dimensional probabilities
            Q: Low-dimensional probabilities
            Y: Current embedding
            
        Returns:
            array: Gradient
        """
        n_samples = Y.shape[0]
        gradient = np.zeros_like(Y)
        
        PQ_diff = P - Q
        
        for i in range(n_samples):
            diff = Y[i] - Y
            distances = np.sum(diff ** 2, axis=1)
            weights = PQ_diff[i] * (1 / (1 + distances))
            gradient[i] = 4 * np.sum(weights[:, np.newaxis] * diff, axis=0)
        
        return gradient

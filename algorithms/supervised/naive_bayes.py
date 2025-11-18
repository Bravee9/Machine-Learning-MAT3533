"""
Naive Bayes Classifier
Probability-based classification algorithm using Bayes' theorem
"""

import numpy as np
from collections import defaultdict


class NaiveBayesClassifier:
    """
    Gaussian Naive Bayes Classifier
    
    Assumes features follow Gaussian distribution and are conditionally independent.
    Uses maximum likelihood estimation for parameters.
    
    Attributes:
        classes (array): Unique class labels
        class_priors (dict): Prior probabilities P(Y=c)
        means (dict): Mean values for each feature per class
        variances (dict): Variance values for each feature per class
    """
    
    def __init__(self):
        self.classes = None
        self.class_priors = {}
        self.means = {}
        self.variances = {}
        
    def fit(self, X, y):
        """
        Train the Naive Bayes classifier
        
        Args:
            X (array-like): Training features, shape (n_samples, n_features)
            y (array-like): Training labels, shape (n_samples,)
        
        Returns:
            self: Fitted classifier
        """
        X = np.array(X)
        y = np.array(y)
        
        self.classes = np.unique(y)
        n_samples = len(y)
        
        # Calculate class priors and feature statistics
        for c in self.classes:
            X_c = X[y == c]
            self.class_priors[c] = len(X_c) / n_samples
            self.means[c] = np.mean(X_c, axis=0)
            self.variances[c] = np.var(X_c, axis=0) + 1e-9  # Add small constant for numerical stability
        
        return self
    
    def _calculate_likelihood(self, x, mean, variance):
        """
        Calculate Gaussian likelihood P(x|y)
        
        Args:
            x: Feature value
            mean: Mean of the distribution
            variance: Variance of the distribution
            
        Returns:
            Likelihood probability
        """
        eps = 1e-9
        exponent = np.exp(-((x - mean) ** 2) / (2 * variance + eps))
        return (1 / np.sqrt(2 * np.pi * variance + eps)) * exponent
    
    def _calculate_posterior(self, x):
        """
        Calculate posterior probabilities for all classes
        
        Args:
            x: Sample features
            
        Returns:
            Dictionary of posterior probabilities for each class
        """
        posteriors = {}
        
        for c in self.classes:
            # Start with log prior
            posterior = np.log(self.class_priors[c])
            
            # Add log likelihoods
            likelihood = self._calculate_likelihood(x, self.means[c], self.variances[c])
            posterior += np.sum(np.log(likelihood + 1e-9))
            
            posteriors[c] = posterior
        
        return posteriors
    
    def predict(self, X):
        """
        Predict class labels for samples
        
        Args:
            X (array-like): Test features, shape (n_samples, n_features)
            
        Returns:
            array: Predicted class labels
        """
        X = np.array(X)
        predictions = []
        
        for x in X:
            posteriors = self._calculate_posterior(x)
            predictions.append(max(posteriors, key=posteriors.get))
        
        return np.array(predictions)
    
    def predict_proba(self, X):
        """
        Predict class probabilities for samples
        
        Args:
            X (array-like): Test features, shape (n_samples, n_features)
            
        Returns:
            array: Class probabilities, shape (n_samples, n_classes)
        """
        X = np.array(X)
        probas = []
        
        for x in X:
            posteriors = self._calculate_posterior(x)
            # Convert log probabilities to probabilities
            max_log = max(posteriors.values())
            exp_posteriors = {c: np.exp(posteriors[c] - max_log) for c in self.classes}
            total = sum(exp_posteriors.values())
            probas.append([exp_posteriors[c] / total for c in self.classes])
        
        return np.array(probas)
    
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

"""
Model Selection Utilities
Cross-validation and hyperparameter optimization
"""

import numpy as np
from .evaluation import accuracy_score, mean_squared_error


class KFold:
    """
    K-Fold cross-validator
    
    Provides train/test indices to split data in train/test sets.
    
    Attributes:
        n_splits (int): Number of folds
        shuffle (bool): Whether to shuffle data
        random_state (int): Random seed
    """
    
    def __init__(self, n_splits=5, shuffle=True, random_state=None):
        """
        Initialize KFold
        
        Args:
            n_splits (int): Number of folds
            shuffle (bool): Shuffle data before splitting
            random_state (int): Random seed
        """
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        
    def split(self, X, y=None):
        """
        Generate indices to split data into training and test set
        
        Args:
            X (array-like): Data to split
            y (array-like): Target variable (optional)
            
        Yields:
            tuple: (train_indices, test_indices)
        """
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        if self.shuffle:
            if self.random_state is not None:
                np.random.seed(self.random_state)
            np.random.shuffle(indices)
        
        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        fold_sizes[:n_samples % self.n_splits] += 1
        
        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            test_indices = indices[start:stop]
            train_indices = np.concatenate([indices[:start], indices[stop:]])
            yield train_indices, test_indices
            current = stop


def cross_val_score(model, X, y, cv=5, scoring='accuracy'):
    """
    Evaluate model using cross-validation
    
    Args:
        model: Model with fit and predict methods
        X (array-like): Features
        y (array-like): Labels
        cv (int): Number of folds
        scoring (str): Scoring metric ('accuracy' or 'mse')
        
    Returns:
        array: Cross-validation scores
    """
    X = np.array(X)
    y = np.array(y)
    
    kfold = KFold(n_splits=cv)
    scores = []
    
    for train_idx, test_idx in kfold.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        if scoring == 'accuracy':
            score = accuracy_score(y_test, y_pred)
        elif scoring == 'mse':
            score = -mean_squared_error(y_test, y_pred)  # Negative for consistency
        else:
            score = model.score(X_test, y_test)
        
        scores.append(score)
    
    return np.array(scores)


class GridSearchCV:
    """
    Exhaustive search over specified parameter values
    
    Attributes:
        model: Base model to optimize
        param_grid (dict): Parameter grid
        cv (int): Number of cross-validation folds
        scoring (str): Scoring metric
        best_params (dict): Best parameters found
        best_score (float): Best score achieved
        best_model: Best fitted model
    """
    
    def __init__(self, model, param_grid, cv=5, scoring='accuracy'):
        """
        Initialize GridSearchCV
        
        Args:
            model: Model to optimize
            param_grid (dict): Grid of parameters to search
            cv (int): Number of folds
            scoring (str): Scoring metric
        """
        self.model = model
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring
        self.best_params = None
        self.best_score = -np.inf
        self.best_model = None
        
    def fit(self, X, y):
        """
        Fit GridSearchCV
        
        Args:
            X (array-like): Training features
            y (array-like): Training labels
            
        Returns:
            self: Fitted grid search
        """
        # Generate all parameter combinations
        param_combinations = self._generate_param_combinations()
        
        for params in param_combinations:
            # Create model with current parameters
            model = self._create_model_with_params(params)
            
            # Evaluate with cross-validation
            scores = cross_val_score(model, X, y, cv=self.cv, scoring=self.scoring)
            mean_score = np.mean(scores)
            
            # Update best parameters
            if mean_score > self.best_score:
                self.best_score = mean_score
                self.best_params = params
                self.best_model = model
        
        # Fit best model on full dataset
        self.best_model.fit(X, y)
        
        return self
    
    def _generate_param_combinations(self):
        """
        Generate all combinations of parameters
        
        Returns:
            list: List of parameter dictionaries
        """
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        
        combinations = []
        self._recursive_combinations(keys, values, 0, {}, combinations)
        
        return combinations
    
    def _recursive_combinations(self, keys, values, idx, current, combinations):
        """
        Recursively generate parameter combinations
        
        Args:
            keys: Parameter names
            values: Parameter values
            idx: Current index
            current: Current combination
            combinations: List to store combinations
        """
        if idx == len(keys):
            combinations.append(current.copy())
            return
        
        for value in values[idx]:
            current[keys[idx]] = value
            self._recursive_combinations(keys, values, idx + 1, current, combinations)
    
    def _create_model_with_params(self, params):
        """
        Create model instance with given parameters
        
        Args:
            params (dict): Parameters
            
        Returns:
            Model instance
        """
        # Import the model class
        model_class = type(self.model)
        return model_class(**params)
    
    def predict(self, X):
        """
        Predict using best model
        
        Args:
            X (array-like): Test features
            
        Returns:
            array: Predictions
        """
        return self.best_model.predict(X)


class RandomizedSearchCV:
    """
    Randomized search over parameter distributions
    
    Similar to GridSearchCV but samples a fixed number of parameter settings.
    
    Attributes:
        model: Base model to optimize
        param_distributions (dict): Parameter distributions
        n_iter (int): Number of parameter settings to sample
        cv (int): Number of cross-validation folds
        scoring (str): Scoring metric
        best_params (dict): Best parameters found
        best_score (float): Best score achieved
        best_model: Best fitted model
    """
    
    def __init__(self, model, param_distributions, n_iter=10, cv=5, scoring='accuracy', random_state=None):
        """
        Initialize RandomizedSearchCV
        
        Args:
            model: Model to optimize
            param_distributions (dict): Parameter distributions
            n_iter (int): Number of iterations
            cv (int): Number of folds
            scoring (str): Scoring metric
            random_state (int): Random seed
        """
        self.model = model
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.cv = cv
        self.scoring = scoring
        self.random_state = random_state
        self.best_params = None
        self.best_score = -np.inf
        self.best_model = None
        
    def fit(self, X, y):
        """
        Fit RandomizedSearchCV
        
        Args:
            X (array-like): Training features
            y (array-like): Training labels
            
        Returns:
            self: Fitted randomized search
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        for _ in range(self.n_iter):
            # Sample parameters
            params = self._sample_parameters()
            
            # Create model with sampled parameters
            model = self._create_model_with_params(params)
            
            # Evaluate with cross-validation
            scores = cross_val_score(model, X, y, cv=self.cv, scoring=self.scoring)
            mean_score = np.mean(scores)
            
            # Update best parameters
            if mean_score > self.best_score:
                self.best_score = mean_score
                self.best_params = params
                self.best_model = model
        
        # Fit best model on full dataset
        self.best_model.fit(X, y)
        
        return self
    
    def _sample_parameters(self):
        """
        Sample parameters from distributions
        
        Returns:
            dict: Sampled parameters
        """
        params = {}
        for key, values in self.param_distributions.items():
            if isinstance(values, list):
                params[key] = np.random.choice(values)
            else:
                params[key] = values
        return params
    
    def _create_model_with_params(self, params):
        """
        Create model instance with given parameters
        
        Args:
            params (dict): Parameters
            
        Returns:
            Model instance
        """
        model_class = type(self.model)
        return model_class(**params)
    
    def predict(self, X):
        """
        Predict using best model
        
        Args:
            X (array-like): Test features
            
        Returns:
            array: Predictions
        """
        return self.best_model.predict(X)

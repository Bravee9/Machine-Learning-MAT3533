"""
Supervised Learning Algorithms
Implementations from theoretical foundations
"""

from .naive_bayes import NaiveBayesClassifier
from .logistic_regression import LogisticRegression
from .decision_tree import DecisionTreeClassifier
from .random_forest import RandomForestClassifier
from .svm import SupportVectorMachine
from .linear_regression import LinearRegression
from .knn import KNearestNeighbors

__all__ = [
    'NaiveBayesClassifier',
    'LogisticRegression',
    'DecisionTreeClassifier',
    'RandomForestClassifier',
    'SupportVectorMachine',
    'LinearRegression',
    'KNearestNeighbors'
]

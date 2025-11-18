"""
Model Evaluation Metrics
Performance metrics for classification and regression
"""

import numpy as np


def accuracy_score(y_true, y_pred):
    """
    Calculate accuracy
    
    Args:
        y_true (array-like): True labels
        y_pred (array-like): Predicted labels
        
    Returns:
        float: Accuracy score
    """
    return np.mean(y_true == y_pred)


def precision_score(y_true, y_pred, average='binary'):
    """
    Calculate precision
    
    Precision = TP / (TP + FP)
    
    Args:
        y_true (array-like): True labels
        y_pred (array-like): Predicted labels
        average (str): Averaging method ('binary' or 'macro')
        
    Returns:
        float: Precision score
    """
    if average == 'binary':
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0
    else:
        # Macro-averaging for multi-class
        classes = np.unique(y_true)
        precisions = []
        for c in classes:
            tp = np.sum((y_true == c) & (y_pred == c))
            fp = np.sum((y_true != c) & (y_pred == c))
            precisions.append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
        return np.mean(precisions)


def recall_score(y_true, y_pred, average='binary'):
    """
    Calculate recall (sensitivity)
    
    Recall = TP / (TP + FN)
    
    Args:
        y_true (array-like): True labels
        y_pred (array-like): Predicted labels
        average (str): Averaging method ('binary' or 'macro')
        
    Returns:
        float: Recall score
    """
    if average == 'binary':
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0
    else:
        # Macro-averaging for multi-class
        classes = np.unique(y_true)
        recalls = []
        for c in classes:
            tp = np.sum((y_true == c) & (y_pred == c))
            fn = np.sum((y_true == c) & (y_pred != c))
            recalls.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
        return np.mean(recalls)


def f1_score(y_true, y_pred, average='binary'):
    """
    Calculate F1 score
    
    F1 = 2 * (precision * recall) / (precision + recall)
    
    Args:
        y_true (array-like): True labels
        y_pred (array-like): Predicted labels
        average (str): Averaging method ('binary' or 'macro')
        
    Returns:
        float: F1 score
    """
    precision = precision_score(y_true, y_pred, average)
    recall = recall_score(y_true, y_pred, average)
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * (precision * recall) / (precision + recall)


def confusion_matrix(y_true, y_pred):
    """
    Calculate confusion matrix
    
    Args:
        y_true (array-like): True labels
        y_pred (array-like): Predicted labels
        
    Returns:
        array: Confusion matrix
    """
    classes = np.unique(np.concatenate([y_true, y_pred]))
    n_classes = len(classes)
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    
    class_to_idx = {c: i for i, c in enumerate(classes)}
    
    for true, pred in zip(y_true, y_pred):
        matrix[class_to_idx[true], class_to_idx[pred]] += 1
    
    return matrix


def mean_squared_error(y_true, y_pred):
    """
    Calculate mean squared error
    
    Args:
        y_true (array-like): True values
        y_pred (array-like): Predicted values
        
    Returns:
        float: MSE
    """
    return np.mean((y_true - y_pred) ** 2)


def mean_absolute_error(y_true, y_pred):
    """
    Calculate mean absolute error
    
    Args:
        y_true (array-like): True values
        y_pred (array-like): Predicted values
        
    Returns:
        float: MAE
    """
    return np.mean(np.abs(y_true - y_pred))


def r2_score(y_true, y_pred):
    """
    Calculate R² (coefficient of determination)
    
    Args:
        y_true (array-like): True values
        y_pred (array-like): Predicted values
        
    Returns:
        float: R² score
    """
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    
    if ss_total == 0:
        return 0.0
    
    return 1 - (ss_residual / ss_total)


def classification_report(y_true, y_pred):
    """
    Generate classification report
    
    Args:
        y_true (array-like): True labels
        y_pred (array-like): Predicted labels
        
    Returns:
        dict: Report with metrics for each class
    """
    classes = np.unique(y_true)
    report = {}
    
    for c in classes:
        y_true_binary = (y_true == c).astype(int)
        y_pred_binary = (y_pred == c).astype(int)
        
        report[c] = {
            'precision': precision_score(y_true_binary, y_pred_binary),
            'recall': recall_score(y_true_binary, y_pred_binary),
            'f1-score': f1_score(y_true_binary, y_pred_binary)
        }
    
    report['accuracy'] = accuracy_score(y_true, y_pred)
    
    return report

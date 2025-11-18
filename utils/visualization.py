"""
Visualization Utilities
Plotting functions for data exploration and model interpretation
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_confusion_matrix(confusion_matrix, class_names=None, title='Confusion Matrix', cmap='Blues'):
    """
    Plot confusion matrix as heatmap
    
    Args:
        confusion_matrix (array): Confusion matrix
        class_names (list): Names of classes
        title (str): Plot title
        cmap (str): Color map
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap=cmap, 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()


def plot_decision_boundary(model, X, y, resolution=0.02, title='Decision Boundary'):
    """
    Plot decision boundary for 2D classification
    
    Args:
        model: Trained classifier with predict method
        X (array): Features (must be 2D)
        y (array): Labels
        resolution (float): Grid resolution
        title (str): Plot title
    """
    if X.shape[1] != 2:
        raise ValueError("X must have exactly 2 features for decision boundary plot")
    
    # Create color maps
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = plt.cm.RdYlBu
    
    # Plot decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    
    Z = model.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    
    plt.figure(figsize=(10, 8))
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # Plot samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                   alpha=0.8, c=colors[idx],
                   marker=markers[idx], label=cl, edgecolor='black')
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()


def plot_learning_curve(train_scores, val_scores, title='Learning Curve'):
    """
    Plot learning curve showing training and validation scores
    
    Args:
        train_scores (array): Training scores over epochs
        val_scores (array): Validation scores over epochs
        title (str): Plot title
    """
    epochs = np.arange(1, len(train_scores) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_scores, 'o-', label='Training Score')
    plt.plot(epochs, val_scores, 's-', label='Validation Score')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_loss_curve(loss_history, title='Loss Curve'):
    """
    Plot training loss over epochs
    
    Args:
        loss_history (array): Loss values over epochs
        title (str): Plot title
    """
    epochs = np.arange(1, len(loss_history) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, loss_history, 'b-', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_feature_importance(feature_names, importance_scores, title='Feature Importance'):
    """
    Plot feature importance as bar chart
    
    Args:
        feature_names (list): Names of features
        importance_scores (array): Importance scores
        title (str): Plot title
    """
    indices = np.argsort(importance_scores)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(importance_scores)), importance_scores[indices])
    plt.xticks(range(len(importance_scores)), 
               [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_correlation_matrix(X, feature_names=None, title='Correlation Matrix'):
    """
    Plot correlation matrix as heatmap
    
    Args:
        X (array): Feature matrix
        feature_names (list): Names of features
        title (str): Plot title
    """
    correlation = np.corrcoef(X.T)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0,
                xticklabels=feature_names, yticklabels=feature_names,
                vmin=-1, vmax=1)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_clusters(X, labels, centroids=None, title='Cluster Visualization'):
    """
    Plot clustering results (for 2D data)
    
    Args:
        X (array): Features (must be 2D)
        labels (array): Cluster labels
        centroids (array): Cluster centroids (optional)
        title (str): Plot title
    """
    if X.shape[1] != 2:
        raise ValueError("X must have exactly 2 features for cluster plot")
    
    plt.figure(figsize=(10, 8))
    
    # Plot points
    scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', 
                         alpha=0.6, edgecolor='k')
    
    # Plot centroids if provided
    if centroids is not None:
        plt.scatter(centroids[:, 0], centroids[:, 1], 
                   c='red', marker='X', s=200, edgecolor='black', 
                   linewidth=2, label='Centroids')
        plt.legend()
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.colorbar(scatter, label='Cluster')
    plt.tight_layout()
    plt.show()


def plot_pca_variance(explained_variance_ratio, title='PCA Explained Variance'):
    """
    Plot explained variance ratio for PCA components
    
    Args:
        explained_variance_ratio (array): Variance ratio for each component
        title (str): Plot title
    """
    cumulative_variance = np.cumsum(explained_variance_ratio)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Individual variance
    ax1.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio)
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Explained Variance Ratio')
    ax1.set_title('Individual Explained Variance')
    ax1.grid(True, alpha=0.3)
    
    # Cumulative variance
    ax2.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'o-')
    ax2.axhline(y=0.95, color='r', linestyle='--', label='95% threshold')
    ax2.set_xlabel('Number of Components')
    ax2.set_ylabel('Cumulative Explained Variance')
    ax2.set_title('Cumulative Explained Variance')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_roc_curve(y_true, y_scores, title='ROC Curve'):
    """
    Plot ROC curve
    
    Args:
        y_true (array): True binary labels
        y_scores (array): Predicted probabilities
        title (str): Plot title
    """
    # Sort by scores
    indices = np.argsort(y_scores)[::-1]
    y_true_sorted = y_true[indices]
    
    # Calculate TPR and FPR at different thresholds
    tpr = []
    fpr = []
    
    n_positive = np.sum(y_true == 1)
    n_negative = np.sum(y_true == 0)
    
    tp = 0
    fp = 0
    
    for label in y_true_sorted:
        if label == 1:
            tp += 1
        else:
            fp += 1
        tpr.append(tp / n_positive if n_positive > 0 else 0)
        fpr.append(fp / n_negative if n_negative > 0 else 0)
    
    # Calculate AUC
    auc = np.trapz(tpr, fpr)
    
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], 'r--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_data_distribution(X, feature_names=None, title='Feature Distributions'):
    """
    Plot distribution of features
    
    Args:
        X (array): Feature matrix
        feature_names (list): Names of features
        title (str): Plot title
    """
    n_features = X.shape[1]
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten() if n_features > 1 else [axes]
    
    for i in range(n_features):
        axes[i].hist(X[:, i], bins=30, edgecolor='black', alpha=0.7)
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Frequency')
        if feature_names:
            axes[i].set_title(feature_names[i])
        else:
            axes[i].set_title(f'Feature {i+1}')
        axes[i].grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_features, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

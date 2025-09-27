# backend/ml_models/evaluation/visualization.py

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix
from typing import List, Optional
import io


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], title: str = 'Confusion Matrix') -> bytes:
    """
    Generate and return a heatmap visualization of a confusion matrix.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return buf.read()


def plot_roc_curve(y_true: np.ndarray, y_proba: np.ndarray, title: str = 'ROC Curve') -> bytes:
    """
    Generate and return a plot of the Receiver Operating Characteristic curve.
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return buf.read()


def plot_feature_importance(importances: np.ndarray, feature_names: List[str],
                          title: str = 'Feature Importance', top_k: int = 20) -> bytes:
    """
    Create a bar chart showing the most important features.
    """
    # Sort features by importance
    indices = np.argsort(importances)[::-1][:top_k]

    plt.figure(figsize=(10, 8))
    plt.title(title)
    plt.bar(range(len(indices)), importances[indices])
    plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45)
    plt.xlabel('Features')
    plt.ylabel('Importance')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return buf.read()
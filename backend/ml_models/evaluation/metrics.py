# backend/ml_models/evaluation/metrics.py

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    matthews_corrcoef,
    cohen_kappa_score
)
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)


def compute_classification_metrics(y_true: np.ndarray,
                                 y_pred: np.ndarray,
                                 y_proba: Optional[np.ndarray] = None,
                                 class_names: Optional[List[str]] = None,
                                 average: str = 'weighted') -> Dict[str, Any]:
    """
    Calculates a comprehensive set of classification metrics.

    This function provides a standardized way to evaluate classification models
    across the project, ensuring consistent metric calculation and naming.

    Args:
        y_true: True class labels
        y_pred: Predicted class labels
        y_proba: Predicted probabilities (for probabilistic metrics)
        class_names: Optional list of class names for reporting
        average: Averaging strategy for multi-class metrics

    Returns:
        Dictionary containing all computed metrics
    """
    metrics = {}

    try:
        # Basic classification metrics
        metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
        metrics['precision'] = float(precision_score(y_true, y_pred, average=average, zero_division=0))
        metrics['recall'] = float(recall_score(y_true, y_pred, average=average, zero_division=0))
        metrics['f1'] = float(f1_score(y_true, y_pred, average=average, zero_division=0))

        # Additional metrics
        metrics['matthews_corrcoef'] = float(matthews_corrcoef(y_true, y_pred))
        metrics['cohen_kappa'] = float(cohen_kappa_score(y_true, y_pred))

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()

        # Per-class metrics
        if class_names is None:
            class_names = [f"class_{i}" for i in range(len(np.unique(y_true)))]

        per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
        per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
        per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

        metrics['per_class_metrics'] = {
            class_names[i]: {
                'precision': float(per_class_precision[i]),
                'recall': float(per_class_recall[i]),
                'f1': float(per_class_f1[i])
            }
            for i in range(len(class_names))
        }

        # Probabilistic metrics (if probabilities are provided)
        if y_proba is not None:
            # Handle binary vs multi-class classification
            if y_proba.shape[1] == 2:
                # Binary classification
                positive_proba = y_proba[:, 1]
                metrics['auc_roc'] = float(roc_auc_score(y_true, positive_proba))
                metrics['average_precision'] = float(average_precision_score(y_true, positive_proba))

                # ROC curve data
                fpr, tpr, roc_thresholds = roc_curve(y_true, positive_proba)
                metrics['roc_curve'] = {
                    'fpr': fpr.tolist(),
                    'tpr': tpr.tolist(),
                    'thresholds': roc_thresholds.tolist()
                }

                # Precision-Recall curve data
                precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, positive_proba)
                metrics['precision_recall_curve'] = {
                    'precision': precision_curve.tolist(),
                    'recall': recall_curve.tolist(),
                    'thresholds': pr_thresholds.tolist()
                }

            else:
                # Multi-class classification
                try:
                    metrics['auc_roc_ovr'] = float(roc_auc_score(y_true, y_proba, multi_class='ovr', average=average))
                    metrics['auc_roc_ovo'] = float(roc_auc_score(y_true, y_proba, multi_class='ovo', average=average))
                except ValueError as e:
                    logger.warning(f"Could not compute multi-class AUC: {e}")

        # Classification report as string
        report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
        metrics['classification_report'] = report

        # Sample-level statistics
        metrics['total_samples'] = int(len(y_true))
        metrics['class_distribution'] = {
            class_names[i]: int(np.sum(y_true == i))
            for i in range(len(class_names))
        }

        logger.info(f"Computed classification metrics. Accuracy: {metrics['accuracy']:.4f}")

    except Exception as e:
        logger.error(f"Error computing classification metrics: {e}")
        raise

    return metrics


def compute_binary_classification_metrics(y_true: np.ndarray,
                                        y_pred: np.ndarray,
                                        y_proba: np.ndarray,
                                        threshold: float = 0.5) -> Dict[str, Any]:
    """
    Compute metrics specifically for binary classification tasks.

    Args:
        y_true: True binary labels (0 or 1)
        y_pred: Predicted binary labels
        y_proba: Predicted probabilities for positive class
        threshold: Classification threshold

    Returns:
        Dictionary containing binary classification metrics
    """
    # Ensure binary format
    if y_proba.ndim == 2:
        y_proba = y_proba[:, 1]  # Take positive class probabilities

    # Apply threshold to get predictions if needed
    y_pred_threshold = (y_proba >= threshold).astype(int)

    metrics = {
        'threshold': threshold,
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'accuracy_at_threshold': float(accuracy_score(y_true, y_pred_threshold)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        'auc_roc': float(roc_auc_score(y_true, y_proba)),
        'average_precision': float(average_precision_score(y_true, y_proba))
    }

    # True/False Positives and Negatives
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics.update({
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'specificity': float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
            'sensitivity': float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
            'positive_predictive_value': float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
            'negative_predictive_value': float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
        })

    return metrics


def compute_threshold_metrics(y_true: np.ndarray,
                            y_proba: np.ndarray,
                            thresholds: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    Compute metrics across different classification thresholds.

    Args:
        y_true: True binary labels
        y_proba: Predicted probabilities for positive class
        thresholds: Array of thresholds to evaluate

    Returns:
        Dictionary containing metrics for each threshold
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 1.0, 0.1)

    threshold_metrics = []

    for threshold in thresholds:
        y_pred_thresh = (y_proba >= threshold).astype(int)

        metrics = {
            'threshold': float(threshold),
            'accuracy': float(accuracy_score(y_true, y_pred_thresh)),
            'precision': float(precision_score(y_true, y_pred_thresh, zero_division=0)),
            'recall': float(recall_score(y_true, y_pred_thresh, zero_division=0)),
            'f1': float(f1_score(y_true, y_pred_thresh, zero_division=0))
        }

        # Calculate specificity and sensitivity
        cm = confusion_matrix(y_true, y_pred_thresh)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics['specificity'] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
            metrics['sensitivity'] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

        threshold_metrics.append(metrics)

    return {
        'threshold_metrics': threshold_metrics,
        'optimal_threshold_f1': _find_optimal_threshold(threshold_metrics, 'f1'),
        'optimal_threshold_accuracy': _find_optimal_threshold(threshold_metrics, 'accuracy')
    }


def _find_optimal_threshold(threshold_metrics: List[Dict], metric: str) -> Dict[str, float]:
    """Find the threshold that maximizes a given metric."""
    best_metric = max(threshold_metrics, key=lambda x: x[metric])
    return {
        'threshold': best_metric['threshold'],
        'value': best_metric[metric]
    }


def compute_model_comparison_metrics(models_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compare metrics across multiple models.

    Args:
        models_results: Dictionary mapping model names to their metric dictionaries

    Returns:
        Dictionary containing comparison statistics
    """
    if not models_results:
        return {}

    comparison = {
        'model_count': len(models_results),
        'models': list(models_results.keys())
    }

    # Extract common metrics
    common_metrics = ['accuracy', 'precision', 'recall', 'f1']
    if 'auc_roc' in next(iter(models_results.values())):
        common_metrics.append('auc_roc')

    for metric in common_metrics:
        values = []
        model_values = {}

        for model_name, results in models_results.items():
            if metric in results:
                value = results[metric]
                values.append(value)
                model_values[model_name] = value

        if values:
            comparison[f'{metric}_comparison'] = {
                'values': model_values,
                'best_model': max(model_values.items(), key=lambda x: x[1])[0],
                'worst_model': min(model_values.items(), key=lambda x: x[1])[0],
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'range': float(np.max(values) - np.min(values))
            }

    return comparison


def compute_calibration_metrics(y_true: np.ndarray,
                               y_proba: np.ndarray,
                               n_bins: int = 10) -> Dict[str, Any]:
    """
    Compute calibration metrics for probabilistic predictions.

    Args:
        y_true: True binary labels
        y_proba: Predicted probabilities
        n_bins: Number of bins for calibration curve

    Returns:
        Dictionary containing calibration metrics
    """
    from sklearn.calibration import calibration_curve

    # Ensure binary probabilities
    if y_proba.ndim == 2:
        y_proba = y_proba[:, 1]

    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_proba, n_bins=n_bins
    )

    # Brier score (lower is better)
    brier_score = np.mean((y_proba - y_true) ** 2)

    # Expected Calibration Error
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_proba > bin_lower) & (y_proba <= bin_upper)
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_proba[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return {
        'brier_score': float(brier_score),
        'expected_calibration_error': float(ece),
        'calibration_curve': {
            'fraction_of_positives': fraction_of_positives.tolist(),
            'mean_predicted_value': mean_predicted_value.tolist()
        },
        'n_bins': n_bins
    }


def compute_fairness_metrics(y_true: np.ndarray,
                           y_pred: np.ndarray,
                           y_proba: np.ndarray,
                           sensitive_features: np.ndarray,
                           sensitive_feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Compute fairness metrics across different groups.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities
        sensitive_features: Array indicating group membership
        sensitive_feature_names: Names for the sensitive feature groups

    Returns:
        Dictionary containing fairness metrics
    """
    if sensitive_feature_names is None:
        unique_groups = np.unique(sensitive_features)
        sensitive_feature_names = [f"group_{g}" for g in unique_groups]

    fairness_metrics = {}

    for i, group_name in enumerate(sensitive_feature_names):
        group_mask = sensitive_features == i
        if np.sum(group_mask) == 0:
            continue

        group_metrics = compute_classification_metrics(
            y_true[group_mask],
            y_pred[group_mask],
            y_proba[group_mask] if y_proba.ndim == 1 else y_proba[group_mask],
            average='binary'
        )

        fairness_metrics[group_name] = {
            'sample_size': int(np.sum(group_mask)),
            'accuracy': group_metrics['accuracy'],
            'precision': group_metrics['precision'],
            'recall': group_metrics['recall'],
            'f1': group_metrics['f1']
        }

        if 'auc_roc' in group_metrics:
            fairness_metrics[group_name]['auc_roc'] = group_metrics['auc_roc']

    # Calculate fairness gaps
    if len(fairness_metrics) >= 2:
        metrics_list = list(fairness_metrics.values())
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            values = [m[metric] for m in metrics_list]
            fairness_metrics[f'{metric}_gap'] = float(np.max(values) - np.min(values))

    return fairness_metrics
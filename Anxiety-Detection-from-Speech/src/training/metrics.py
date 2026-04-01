"""
Evaluation metrics for the Speech Anxiety Detection pipeline.

Computes:
- AUC-ROC (primary metric)
- UAR (Unweighted Average Recall — standard in AVEC challenges)
- F1-Score
- Sensitivity / Recall
- Specificity
- Precision / PPV
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    confusion_matrix,
    precision_score,
    recall_score,
    accuracy_score,
    roc_curve,
)
from typing import Dict, Optional, Tuple


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute all evaluation metrics for binary anxiety classification.

    Args:
        y_true: Ground truth labels (0 or 1), shape [N].
        y_prob: Predicted probabilities (0.0–1.0), shape [N].
        threshold: Decision threshold for binary predictions.

    Returns:
        Dict with all metrics.
    """
    # Binary predictions
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {}

    # AUC-ROC (threshold-independent)
    try:
        metrics["auc_roc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        # Only one class present in y_true
        metrics["auc_roc"] = 0.0

    # Confusion matrix components
    if len(np.unique(y_true)) > 1:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    else:
        if y_true[0] == 1:
            tp, fn = (y_pred == 1).sum(), (y_pred == 0).sum()
            tn, fp = 0, 0
        else:
            tn, fp = (y_pred == 0).sum(), (y_pred == 1).sum()
            tp, fn = 0, 0

    # Sensitivity (Recall) — Critical for screening
    sensitivity = tp / max(tp + fn, 1)
    metrics["sensitivity"] = float(sensitivity)

    # Specificity
    specificity = tn / max(tn + fp, 1)
    metrics["specificity"] = float(specificity)

    # UAR (Unweighted Average Recall) — standard in AVEC
    metrics["uar"] = float((sensitivity + specificity) / 2)

    # F1 Score
    metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))

    # Precision (PPV)
    metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))

    # Accuracy (for reference, not primary)
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))

    # Counts
    metrics["tp"] = int(tp)
    metrics["fp"] = int(fp)
    metrics["tn"] = int(tn)
    metrics["fn"] = int(fn)

    return metrics


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    optimize_for: str = "uar",
) -> Tuple[float, Dict[str, float]]:
    """
    Find the optimal decision threshold by sweeping values.

    Args:
        y_true: Ground truth labels.
        y_prob: Predicted probabilities.
        optimize_for: Metric to optimize ("uar", "f1", "sensitivity").

    Returns:
        Tuple of (best_threshold, best_metrics).
    """
    best_threshold = 0.5
    best_score = 0.0
    best_metrics = {}

    for threshold in np.arange(0.1, 0.9, 0.01):
        metrics = compute_metrics(y_true, y_prob, threshold=threshold)
        score = metrics.get(optimize_for, 0.0)

        if score > best_score:
            best_score = score
            best_threshold = threshold
            best_metrics = metrics

    best_metrics["optimal_threshold"] = float(best_threshold)
    return float(best_threshold), best_metrics


def format_metrics(metrics: Dict[str, float]) -> str:
    """Format metrics dict as a readable string."""
    lines = [
        f"  AUC-ROC:     {metrics.get('auc_roc', 0):.4f}",
        f"  UAR:         {metrics.get('uar', 0):.4f}",
        f"  F1:          {metrics.get('f1', 0):.4f}",
        f"  Sensitivity: {metrics.get('sensitivity', 0):.4f}",
        f"  Specificity: {metrics.get('specificity', 0):.4f}",
        f"  Precision:   {metrics.get('precision', 0):.4f}",
        f"  Accuracy:    {metrics.get('accuracy', 0):.4f}",
    ]
    return "\n".join(lines)

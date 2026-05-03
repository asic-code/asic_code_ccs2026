"""Classification metrics helpers for AE detection.

The detection set is balanced (n_ae == n_clean) in our pipeline, so
accuracy/precision/recall/F1 are well-defined without class-weight bias.
"""
from __future__ import annotations
import numpy as np
from sklearn.metrics import roc_curve


def _threshold_at_fpr(y_true: np.ndarray, y_score: np.ndarray, target_fpr: float) -> float:
    """Return score threshold with the largest TPR subject to fpr <= target."""
    fpr, tpr, thr = roc_curve(y_true, y_score)
    mask = fpr <= target_fpr
    if not mask.any():
        return float(thr.max())
    best = np.where(mask)[0][np.argmax(tpr[mask])]
    return float(thr[best])


def classification_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict:
    y_pred = (y_score >= threshold).astype(np.int64)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    n = tp + fp + tn + fn
    acc = (tp + tn) / n if n else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {
        "threshold": threshold,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
        "fpr": fp / (fp + tn) if (fp + tn) else 0.0,
    }


def metrics_at_fpr(y_true: np.ndarray, y_score: np.ndarray, target_fpr: float) -> dict:
    thr = _threshold_at_fpr(y_true, y_score, target_fpr)
    return classification_metrics(y_true, y_score, thr)

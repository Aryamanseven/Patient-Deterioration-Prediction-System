"""
Evaluation metrics — centralized, consistent, no duplication.

Every evaluation in the system uses these functions.
Metrics aligned with clinical ML best practices:
  - PR-AUC (primary, recommended for imbalanced clinical data)
  - ROC-AUC (standard discriminative metric)
  - Brier Score (measures calibration quality)
  - F1, Precision, Recall (with optimized threshold)
  - Confusion matrix

Reference: Saito & Rehmsmeier (2015). The Precision-Recall Plot Is More
Informative than the ROC Plot When Evaluating Binary Classifiers on
Imbalanced Datasets.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_binary(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """
    Compute all binary classification metrics.

    Returns dict with: roc_auc, pr_auc, brier_score, f1, precision, recall,
                       confusion_matrix, n_samples, pos_rate
    """
    y_true = np.asarray(y_true, dtype=int)
    y_proba = np.asarray(y_proba, dtype=float)
    y_pred = (y_proba >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "brier_score": float(brier_score_loss(y_true, y_proba)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "threshold": float(threshold),
        "confusion_matrix": cm.tolist(),
        "n_samples": int(len(y_true)),
        "pos_rate": float(y_true.mean()),
    }


def optimize_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    metric: str = "f1",
    min_thresh: float = 0.1,
    max_thresh: float = 0.9,
    step: float = 0.01,
) -> tuple[float, float]:
    """
    Find optimal classification threshold by grid search.

    Returns (best_threshold, best_score).
    """
    y_true = np.asarray(y_true, dtype=int)
    y_proba = np.asarray(y_proba, dtype=float)

    best_score = 0.0
    best_thresh = 0.5

    for t in np.arange(min_thresh, max_thresh + step, step):
        y_pred = (y_proba >= t).astype(int)
        if metric == "f1":
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == "recall":
            score = recall_score(y_true, y_pred, zero_division=0)
        elif metric == "precision":
            score = precision_score(y_true, y_pred, zero_division=0)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        if score > best_score:
            best_score = score
            best_thresh = t

    return float(best_thresh), float(best_score)


def compute_calibration(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
) -> dict[str, Any]:
    """
    Compute calibration metrics: ECE, MCE, reliability diagram data.

    ECE = Expected Calibration Error
    MCE = Maximum Calibration Error
    """
    y_true = np.asarray(y_true, dtype=float)
    y_proba = np.asarray(y_proba, dtype=float)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    mean_predicted = []
    fraction_positive = []
    bin_counts = []
    ece = 0.0
    mce = 0.0

    for i in range(n_bins):
        mask = (y_proba >= bin_edges[i]) & (y_proba < bin_edges[i + 1])
        if i == n_bins - 1:
            mask = (y_proba >= bin_edges[i]) & (y_proba <= bin_edges[i + 1])
        count = mask.sum()
        if count == 0:
            mean_predicted.append(float((bin_edges[i] + bin_edges[i + 1]) / 2))
            fraction_positive.append(0.0)
            bin_counts.append(0)
            continue

        mean_pred = float(y_proba[mask].mean())
        frac_pos = float(y_true[mask].mean())
        gap = abs(mean_pred - frac_pos)

        mean_predicted.append(mean_pred)
        fraction_positive.append(frac_pos)
        bin_counts.append(int(count))
        ece += gap * count
        mce = max(mce, gap)

    ece /= max(len(y_true), 1)

    return {
        "ece": float(ece),
        "mce": float(mce),
        "brier_score": float(brier_score_loss(y_true, y_proba)),
        "mean_predicted": mean_predicted,
        "fraction_positive": fraction_positive,
        "bin_counts": bin_counts,
        "n_bins": n_bins,
    }

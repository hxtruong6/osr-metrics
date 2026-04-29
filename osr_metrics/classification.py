from __future__ import annotations

import warnings

import numpy as np
from sklearn.metrics import average_precision_score, f1_score

# Leave-p-out and bootstrap can produce labels with no positives; sklearn warns.
warnings.filterwarnings("ignore", message="No positive class found in y_true")


def macro_auprc(probs: np.ndarray, labels: np.ndarray) -> float:
    """Macro-averaged Area Under Precision-Recall Curve.

    Args:
        probs:  predicted probabilities, shape [N, K].
        labels: ground-truth multi-hot, shape [N, K].

    Returns:
        Macro-averaged AUPRC in [0, 1]. Higher is better.
    """
    return float(average_precision_score(labels, probs, average="macro"))


def macro_auprc_id_labels(
    probs: np.ndarray,
    labels: np.ndarray,
    label_names: list[str],
    held_out_labels: list[str],
) -> float:
    """Macro-AUPRC over known (non-held-out) labels only.

    In leave-p-out, test_id has no positive examples of held-out labels.
    Including them yields 0.0 per-label AUPRC and unfairly penalizes the score.
    This restricts the metric to the 12 known labels for fair comparison.

    Args:
        probs:  predicted probabilities, shape [N, K].
        labels: ground-truth multi-hot, shape [N, K].
        label_names: list of K label strings in column order.
        held_out_labels: labels to exclude (e.g. Hernia, Pneumonia).

    Returns:
        Macro-averaged AUPRC over known labels only.
    """
    held_out_set = set(held_out_labels)
    known_indices = [k for k, name in enumerate(label_names) if name not in held_out_set]
    if not known_indices:
        return float(average_precision_score(labels, probs, average="macro"))
    probs_sub = probs[:, known_indices]
    labels_sub = labels[:, known_indices]
    return float(average_precision_score(labels_sub, probs_sub, average="macro"))


def per_label_auprc(probs: np.ndarray, labels: np.ndarray) -> list[float]:
    """Per-label AUPRC scores.

    Args:
        probs:  predicted probabilities, shape [N, K].
        labels: ground-truth multi-hot, shape [N, K].

    Returns:
        List of K AUPRC values, one per label.
    """
    K = labels.shape[1]
    scores = []
    for k in range(K):
        if labels[:, k].sum() == 0:
            scores.append(float("nan"))
        else:
            scores.append(float(average_precision_score(labels[:, k], probs[:, k])))
    return scores


def macro_f1_with_thresholds(
    probs: np.ndarray,
    labels: np.ndarray,
    thresholds: list[float],
) -> float:
    """Macro-F1 using per-label thresholds instead of fixed 0.5.

    Args:
        probs:      predicted probabilities, shape [N, K].
        labels:     ground-truth multi-hot, shape [N, K].
        thresholds: list of K thresholds, one per label.

    Returns:
        Macro-averaged F1 score.
    """
    K = probs.shape[1]
    preds = np.zeros_like(probs, dtype=int)
    for k in range(K):
        preds[:, k] = (probs[:, k] >= thresholds[k]).astype(int)
    return float(f1_score(labels, preds, average="macro", zero_division=0))


def f1_per_label(
    preds: np.ndarray, labels: np.ndarray
) -> dict[int, float]:
    """Per-label F1 score.

    Args:
        preds:  binary predictions, shape [N, K].
        labels: ground-truth multi-hot, shape [N, K].

    Returns:
        {label_idx: f1_score} for each label.
    """
    K = labels.shape[1]
    return {
        k: float(f1_score(labels[:, k], preds[:, k], zero_division=0))
        for k in range(K)
    }

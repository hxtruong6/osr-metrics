"""Closed-set classification metrics.

**Scope: multi-label only.** All functions in this module assume per-label
sigmoid probabilities (shape ``[N, K]``) and multi-hot ground truth
(shape ``[N, K]``). For multi-class (single-label) closed-set
evaluation, use ``sklearn.metrics.accuracy_score`` /
``sklearn.metrics.f1_score(..., average='macro')`` directly.
"""
from __future__ import annotations

import warnings
from math import isnan as math_isnan

import numpy as np
from sklearn.metrics import average_precision_score, f1_score

# Leave-p-out and bootstrap can produce labels with no positives; sklearn warns.
warnings.filterwarnings("ignore", message="No positive class found in y_true")


def macro_auprc(probs: np.ndarray, labels: np.ndarray) -> float:
    """Macro-averaged Area Under Precision-Recall Curve.

    | Applies to   | Task                       |
    |--------------|----------------------------|
    | Multi-label  | Closed-set classification  |

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

    | Applies to   | Task                                        |
    |--------------|---------------------------------------------|
    | Multi-label  | Closed-set classification (held-out aware)  |

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

    | Applies to   | Task                                |
    |--------------|-------------------------------------|
    | Multi-label  | Closed-set classification (per-label)|

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

    | Applies to   | Task                       |
    |--------------|----------------------------|
    | Multi-label  | Closed-set classification  |

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


def compute_rc_macro_f1(
    probs_known: np.ndarray,
    labels_known: np.ndarray,
    thresholds: list[float] | np.ndarray,
    mixed_mask: np.ndarray,
    id_only_mask: np.ndarray | None = None,
) -> dict:
    """Rejection-contagion macro-F1 on mixed-novelty images.

    | Applies to   | Task                                       |
    |--------------|--------------------------------------------|
    | Multi-label  | OS-MLC robustness (Regime B contagion)     |

    On mixed-novelty images (≥1 known label AND ≥1 novel label
    co-positive), measures whether the presence of a co-occurring
    novel label degrades prediction of the known labels. Computed as
    macro-F1 over the *K* known labels using only their ground-truth
    columns; the novel-label columns are absent from both ``labels_known``
    and ``probs_known`` by construction.

    If ``id_only_mask`` is provided, also reports the matching ID-only
    macro-F1 baseline and the **contagion penalty**
    ``delta = macro_f1_id_only - macro_f1_mixed`` (positive ⇒ degradation
    on contaminated images).

    Args:
        probs_known: per-label probabilities over the *K* known labels,
            shape ``[N, K]``.
        labels_known: multi-hot ground truth over the *K* known labels,
            shape ``[N, K]``.
        thresholds: per-label decision thresholds, length ``K``.
        mixed_mask: boolean mask, shape ``[N]``. ``True`` for
            mixed-novelty images.
        id_only_mask: optional boolean mask, shape ``[N]``. ``True`` for
            ID-only images (no novel labels positive). If given, the
            contagion delta is included in the output.

    Returns:
        ``{"macro_f1_mixed": float, "n_mixed": int, ...}``. When
        ``id_only_mask`` is given, also includes
        ``"macro_f1_id_only"``, ``"n_id_only"``, and ``"contagion_delta"``.
        ``macro_f1_*`` is ``nan`` if the corresponding subset is empty.
    """
    probs_known = np.asarray(probs_known, dtype=float)
    labels_known = np.asarray(labels_known).astype(int)
    thresholds = np.asarray(thresholds, dtype=float)
    mixed_mask = np.asarray(mixed_mask).astype(bool)

    if probs_known.shape != labels_known.shape:
        raise ValueError(
            "compute_rc_macro_f1: probs_known and labels_known must have "
            f"the same shape; got {probs_known.shape} vs {labels_known.shape}"
        )
    K = probs_known.shape[1]
    if thresholds.shape != (K,):
        raise ValueError(
            f"compute_rc_macro_f1: thresholds must have shape [K]=({K},); "
            f"got {thresholds.shape}"
        )
    if mixed_mask.shape != (probs_known.shape[0],):
        raise ValueError(
            "compute_rc_macro_f1: mixed_mask must have shape [N]; "
            f"got {mixed_mask.shape}"
        )

    def _macro_f1(p_sub: np.ndarray, y_sub: np.ndarray) -> float:
        if p_sub.shape[0] == 0:
            return float("nan")
        preds_sub = (p_sub >= thresholds[None, :]).astype(int)
        return float(
            f1_score(y_sub, preds_sub, average="macro", zero_division=0)
        )

    out: dict = {
        "macro_f1_mixed": _macro_f1(probs_known[mixed_mask], labels_known[mixed_mask]),
        "n_mixed": int(mixed_mask.sum()),
    }

    if id_only_mask is not None:
        id_only_mask = np.asarray(id_only_mask).astype(bool)
        if id_only_mask.shape != (probs_known.shape[0],):
            raise ValueError(
                "compute_rc_macro_f1: id_only_mask must have shape [N]; "
                f"got {id_only_mask.shape}"
            )
        macro_id = _macro_f1(probs_known[id_only_mask], labels_known[id_only_mask])
        out["macro_f1_id_only"] = macro_id
        out["n_id_only"] = int(id_only_mask.sum())
        if math_isnan(macro_id) or math_isnan(out["macro_f1_mixed"]):
            out["contagion_delta"] = float("nan")
        else:
            out["contagion_delta"] = macro_id - out["macro_f1_mixed"]
    return out


def f1_per_label(
    preds: np.ndarray, labels: np.ndarray
) -> dict[int, float]:
    """Per-label F1 score.

    | Applies to   | Task                                  |
    |--------------|---------------------------------------|
    | Multi-label  | Closed-set classification (per-label) |

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

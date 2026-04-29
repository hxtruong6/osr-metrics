"""Multi-class (single-label) closed-set classification metrics.

**Scope: multi-class (single-label) only.** Each sample has exactly one
ground-truth class out of K. For multi-label, use the per-label
functions in ``classification.py``.

Thin wrappers over ``sklearn.metrics`` that accept either integer
predictions ``[N]`` or a softmax / logit matrix ``[N, K]``, so callers
don't have to ``argmax`` themselves.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)


def _to_class_ids(preds: np.ndarray) -> np.ndarray:
    """Reduce 2-D logits/probs to 1-D class IDs; pass 1-D through."""
    arr = np.asarray(preds)
    if arr.ndim == 2:
        return np.asarray(arr.argmax(axis=1))
    if arr.ndim == 1:
        return arr
    raise ValueError(
        f"preds must be 1-D [N] or 2-D [N, K], got shape {arr.shape}"
    )


def top1_accuracy(preds: np.ndarray, y: np.ndarray) -> float:
    """Top-1 (argmax) accuracy.

    | Applies to                | Task                       |
    |---------------------------|----------------------------|
    | Multi-class (single-label)| Closed-set classification  |

    Args:
        preds: Integer class predictions ``[N]`` or softmax / logit
            matrix ``[N, K]``.
        y: Integer ground-truth classes, shape ``[N]``.

    Returns:
        Accuracy in ``[0, 1]``.
    """
    return float(accuracy_score(np.asarray(y), _to_class_ids(preds)))


def macro_f1_multiclass(preds: np.ndarray, y: np.ndarray) -> float:
    """Macro-averaged F1 score over K classes.

    | Applies to                | Task                       |
    |---------------------------|----------------------------|
    | Multi-class (single-label)| Closed-set classification  |

    Equal weight per class — robust to class imbalance compared to
    micro-F1 / accuracy.

    Args:
        preds: Integer class predictions ``[N]`` or softmax / logit
            matrix ``[N, K]``.
        y: Integer ground-truth classes, shape ``[N]``.

    Returns:
        Macro-F1 in ``[0, 1]``.
    """
    return float(
        f1_score(
            np.asarray(y), _to_class_ids(preds), average="macro", zero_division=0
        )
    )


def balanced_accuracy(preds: np.ndarray, y: np.ndarray) -> float:
    """Class-balanced accuracy (mean per-class recall).

    | Applies to                | Task                       |
    |---------------------------|----------------------------|
    | Multi-class (single-label)| Closed-set classification  |

    Useful when class frequencies are imbalanced — equivalent to mean
    recall over classes.

    Args:
        preds: Integer class predictions ``[N]`` or softmax / logit
            matrix ``[N, K]``.
        y: Integer ground-truth classes, shape ``[N]``.

    Returns:
        Balanced accuracy in ``[0, 1]``.
    """
    return float(balanced_accuracy_score(np.asarray(y), _to_class_ids(preds)))

"""Utility helpers (score-direction adapters, input validation).

The single most common user error in this library is passing
**confidence** scores (higher = more ID) where **OOD scores** (higher =
more OOD) are expected. Use ``as_ood_scores`` once at the boundary so
the rest of your pipeline can rely on the convention.
"""
from __future__ import annotations

import warnings
from typing import Literal

import numpy as np


ScoreDirection = Literal["ood", "confidence", "id"]


def as_ood_scores(
    scores: np.ndarray,
    direction: ScoreDirection = "ood",
) -> np.ndarray:
    """Normalise a score array to the library convention (higher = more OOD).

    | Applies to | Task        |
    |------------|-------------|
    | Any        | Adapter     |

    Args:
        scores: Per-sample score array, shape ``[N]``.
        direction: What ``scores`` currently represents:

            - ``"ood"``: higher = more OOD (already in library convention).
              Returned unchanged.
            - ``"confidence"`` or ``"id"``: higher = more confidently ID
              (e.g. softmax max-prob, max-logit). Sign is flipped.

    Returns:
        Score array in OOD convention (higher = more OOD).

    Example:
        >>> import numpy as np
        >>> from osr_metrics import as_ood_scores, auroc
        >>> max_prob = np.array([0.95, 0.10, 0.80, 0.05])  # higher = more ID
        >>> ood_labels = np.array([0, 1, 0, 1])
        >>> scores = as_ood_scores(max_prob, direction="confidence")
        >>> float(auroc(scores, ood_labels))
        1.0
    """
    arr = np.asarray(scores, dtype=float)
    if direction == "ood":
        return arr
    if direction in ("confidence", "id"):
        return -arr
    raise ValueError(
        f"direction must be 'ood', 'confidence', or 'id'; got {direction!r}"
    )


def warn_if_inverted_scores(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    threshold: float = 0.5,
) -> None:
    """Heuristic warning when AUROC < threshold suggests inverted scores.

    Computing AUROC on accidentally inverted scores yields ``1 - AUROC``,
    so values consistently far below 0.5 typically mean the score
    direction is wrong. This helper computes AUROC and warns if it falls
    below ``threshold`` (default 0.5).

    Args:
        scores: Per-sample OOD scores, shape ``[N]``. Higher should mean
            more OOD.
        labels: Binary OOD ground truth, shape ``[N]``.
        threshold: AUROC level below which a warning is emitted.
    """
    from sklearn.metrics import roc_auc_score

    labels_arr = np.asarray(labels)
    if labels_arr.sum() == 0 or (1 - labels_arr).sum() == 0:
        return  # cannot compute AUROC; silently skip
    auc = float(roc_auc_score(labels_arr, np.asarray(scores)))
    if auc < threshold:
        warnings.warn(
            f"AUROC = {auc:.3f} (< {threshold}). Are your scores inverted? "
            "This library expects higher = more OOD. If your scores are "
            "confidence (higher = more ID), wrap them with "
            "as_ood_scores(scores, direction='confidence').",
            stacklevel=2,
        )

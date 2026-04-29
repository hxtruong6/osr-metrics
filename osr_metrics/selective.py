"""Selective-prediction / risk–coverage metrics (task-agnostic).

**Scope:** any task with a per-sample non-negative loss vector. Selective
prediction evaluates how well an OOD/uncertainty score *ranks* samples by
their (closed-set) loss — does abstaining on high-score samples reduce
average risk?

**AURC measures rank quality on the supplied loss, not OOD detection.**
For OOD detection use ``auroc``; for joint OSR use ``compute_aoscr``.
See Jaeger et al. 2024 for a discussion of this conflation as one of the
most common selective-classification evaluation flaws.

**Score-direction convention:** higher = more OOD / more likely to
reject (library standard). Mathematically equivalent to the literature's
"confidence" convention (sign symmetry is exact under rank-only
metrics); see ``warn_if_inverted_aurc`` for a runtime check.

References
----------
- Geifman & El-Yaniv 2017 (NeurIPS), arXiv:1705.08500
- Geifman, Uziel & El-Yaniv 2019 (ICLR), arXiv:1805.08206
- Jaeger et al. 2024 (NeurIPS), arXiv:2407.01032
"""
from __future__ import annotations

import warnings

import numpy as np

__all__ = [
    "rc_curve",
    "aurc",
    "eaurc",
    "selective_risk_at_coverage",
    "selective_accuracy_at_coverage",
    "warn_if_inverted_aurc",
]


def _validate_score_loss(
    ood_score: np.ndarray,
    loss: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Shape / dtype / value checks for selective-prediction inputs."""
    score = np.asarray(ood_score, dtype=float)
    loss_arr = np.asarray(loss, dtype=float)
    if score.ndim != 1:
        raise ValueError(
            f"ood_score must be 1-D, got shape {score.shape}"
        )
    if loss_arr.ndim != 1:
        raise ValueError(
            f"loss must be 1-D, got shape {loss_arr.shape}"
        )
    if score.shape != loss_arr.shape:
        raise ValueError(
            f"ood_score and loss must have the same shape, "
            f"got {score.shape} and {loss_arr.shape}"
        )
    if score.size < 2:
        raise ValueError(
            "selective-prediction metrics require at least 2 samples"
        )
    if not np.all(np.isfinite(score)):
        raise ValueError("ood_score contains NaN or inf")
    if not np.all(np.isfinite(loss_arr)):
        raise ValueError("loss contains NaN or inf")
    if np.any(loss_arr < 0):
        raise ValueError("loss must be non-negative")
    return score, loss_arr


def _validate_coverage(coverage: float) -> float:
    """Coerce and bounds-check a coverage scalar."""
    if not isinstance(coverage, (int, float, np.floating, np.integer)):
        raise TypeError(
            f"coverage must be a real scalar, got {type(coverage).__name__}"
        )
    c = float(coverage)
    if not np.isfinite(c):
        raise ValueError(f"coverage must be finite, got {c}")
    if c <= 0.0 or c > 1.0:
        raise ValueError(f"coverage must be in (0, 1], got {c}")
    return c

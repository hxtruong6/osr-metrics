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
from scipy.stats import rankdata

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


def _rc_curve_with_ties(
    ood_score: np.ndarray,
    loss: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Risk–coverage curve, collapsed to one point per unique score.

    For tied scores, the curve has a single (coverage, risk, threshold)
    point at the *end* of each tied run, with risk = cum_mean[end_idx]
    (the natural selective risk = mean loss over all samples with score
    <= end-of-run threshold). This is input-order-invariant: the risk at
    each end-of-run point depends only on which samples have score <=
    that threshold, not on the within-run ordering.
    """
    score, loss_arr = _validate_score_loss(ood_score, loss)
    n = score.size

    # rankdata(method="average") assigns tied samples the same rank,
    # making run boundaries trivial to detect after a stable sort.
    ranks = rankdata(score, method="average")
    order = np.argsort(ranks, kind="stable")
    sorted_loss = loss_arr[order]
    sorted_ranks = ranks[order]
    sorted_score = score[order]

    # cum_mean[i] = mean(sorted_loss[: i + 1]); the natural selective
    # risk if the prefix sorted_loss[: i + 1] is the kept set.
    cum_sum = np.cumsum(sorted_loss)
    cum_mean = cum_sum / np.arange(1, n + 1)

    # Identify end-of-run indices (last sample in each tied rank-run).
    run_end_mask = np.empty(n, dtype=bool)
    run_end_mask[:-1] = sorted_ranks[1:] != sorted_ranks[:-1]
    run_end_mask[-1] = True
    end_idx = np.where(run_end_mask)[0]

    coverage = (end_idx + 1) / n
    risk = cum_mean[end_idx]
    threshold = sorted_score[end_idx]

    return coverage.astype(float), risk, threshold.astype(float)

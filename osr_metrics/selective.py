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


def rc_curve(
    ood_score: np.ndarray,
    loss: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Risk–coverage curve for an OOD/uncertainty score.

    | Applies to | Task                          |
    |------------|-------------------------------|
    | Any        | Selective prediction          |

    Args:
        ood_score: Score where higher = more likely to reject (library
            convention). Shape ``[N]``.
        loss: Per-sample non-negative loss (e.g.
            ``(y_true != y_pred).astype(float)`` for 0/1
            misclassification, NLL, squared error, …). Shape ``[N]``.

    Returns:
        Tuple ``(coverage, selective_risk, threshold)``:

        - ``coverage``: strictly increasing in ``(0, 1]``, shape ``[M]``.
        - ``selective_risk``: mean loss over the selected
          (lowest-score) subset at each coverage, shape ``[M]``.
        - ``threshold``: score threshold ``τ`` such that
          ``{x : ood_score(x) ≤ τ}`` has the given coverage. Useful for
          deployment.

        ``M ≤ N``; tied scores are collapsed via rank-averaging so the
        curve is invariant to input order and to monotone transforms of
        the score (including sign flip).
    """
    return _rc_curve_with_ties(ood_score, loss)


def aurc(ood_score: np.ndarray, loss: np.ndarray) -> float:
    """Area under the risk–coverage curve. Lower is better.

    | Applies to | Task                          |
    |------------|-------------------------------|
    | Any        | Selective prediction          |

    Computed as the Riemann sum
        AURC = (1/N) * sum_{k=1..N} cum_mean[k]
    where ``cum_mean[k]`` is the mean loss over the ``k`` lowest-score
    samples. This matches the canonical definition used by Geifman &
    El-Yaniv 2017 and the standard reference implementations (Galil 2023,
    Han 2024, TorchUncertainty).

    Tie handling: tied scores have their within-run losses replaced by
    the run mean before computing ``cum_mean``. This makes ``aurc``
    input-order-invariant (and identical to stable-sort semantics when
    no ties are present), at the cost of a small deviation from
    reference implementations on heavily-tied inputs.

    Note: ``aurc`` is *not* equal to ``np.trapezoid`` over the discrete
    ``rc_curve`` output. The two quantities differ by O(1/N) and
    measure related-but-distinct things — ``aurc`` is the canonical
    summary scalar, ``rc_curve`` is the visualization/deployment view.

    Args:
        ood_score: Higher = more likely to reject. Shape ``[N]``.
        loss: Per-sample non-negative loss. Shape ``[N]``.

    Returns:
        AURC in ``[0, max(loss)]``. Lower is better.
    """
    score, loss_arr = _validate_score_loss(ood_score, loss)
    n = score.size

    ranks = rankdata(score, method="average")
    order = np.argsort(ranks, kind="stable")
    sorted_loss = loss_arr[order]
    sorted_ranks = ranks[order]

    # Replace within-run losses with run mean → input-order-invariant.
    run_change = np.concatenate(([True], sorted_ranks[1:] != sorted_ranks[:-1]))
    run_id = np.cumsum(run_change) - 1
    run_counts = np.bincount(run_id)
    run_sums = np.bincount(run_id, weights=sorted_loss)
    run_means = run_sums / run_counts
    averaged_loss = run_means[run_id]

    cum_sum = np.cumsum(averaged_loss)
    cum_mean = cum_sum / np.arange(1, n + 1)
    return float(np.mean(cum_mean))


def eaurc(ood_score: np.ndarray, loss: np.ndarray) -> float:
    """Excess AURC over the optimal (oracle-ranked) baseline.

    | Applies to | Task                          |
    |------------|-------------------------------|
    | Any        | Selective prediction          |

    Computed empirically as ``aurc(score, loss) - aurc(loss, loss)``.
    Passing ``loss`` as the score is equivalent to ranking samples by
    their true loss ascending, which is the oracle ranking.

    For binary 0/1 ``loss``, the empirical baseline converges to the
    closed form ``r + (1 - r) * ln(1 - r)`` (with ``r = mean(loss)``)
    from Geifman, Uziel & El-Yaniv 2019, with O(1/N) finite-sample
    error. We use the empirical form throughout to keep ``eaurc``
    consistent with ``aurc`` at finite N.

    Args:
        ood_score: Higher = more likely to reject. Shape ``[N]``.
        loss: Per-sample non-negative loss. Shape ``[N]``.

    Returns:
        E-AURC. Equals zero iff the score ranks samples in nondecreasing
        order of true loss. Lower is better.
    """
    score, loss_arr = _validate_score_loss(ood_score, loss)
    return aurc(score, loss_arr) - aurc(loss_arr, loss_arr)

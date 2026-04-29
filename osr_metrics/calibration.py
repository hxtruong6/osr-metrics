"""Probability calibration metrics.

**Scope: multi-label / binary.** Both functions flatten ``probs`` and
``labels`` across (sample, label) pairs, treating each as an independent
binary observation. For multi-class (single-label) softmax calibration
(Guo 2017 form) — top-1 confidence vs correctness — use
``sklearn.calibration.calibration_curve`` or ``torchmetrics.CalibrationError``;
a multi-class overload is on the roadmap.

Two metrics:

1. ``expected_calibration_error`` -- ECE bins predicted probabilities and
   measures the gap between mean predicted confidence and empirical accuracy
   inside each bin (Guo et al. 2017, "On Calibration of Modern Neural
   Networks"). For multi-label problems we flatten across (sample, label)
   pairs so each label position contributes one (prob, target) pair.

2. ``brier_score`` -- Mean squared error between predicted probabilities and
   binary targets (Brier 1950). Strictly proper scoring rule -- penalises both
   miscalibration *and* lack of resolution. Lower is better.

Both operate on the same inputs already saved in ``scores.json``: the per-label
``probs`` array and the binary ``label_vecs``.
"""
from __future__ import annotations

import numpy as np


def expected_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Expected Calibration Error (equal-width binning) for multi-label probs.

    | Applies to       | Task         |
    |------------------|--------------|
    | Multi-label / binary | Calibration |

    Not for multi-class softmax (use top-1 confidence form instead).

    Flattens ``probs`` and ``labels`` across (sample, label) pairs so each
    label position is one observation. Buckets the probabilities into
    ``n_bins`` equal-width bins on ``[0, 1]``, then computes the
    sample-weighted mean of ``|mean_conf_b - acc_b|``::

        ECE = sum_b (|B_b| / N) * |conf(B_b) - acc(B_b)|

    where ``conf(B_b) = mean(probs in bin b)`` and
    ``acc(B_b) = mean(labels in bin b)`` (the empirical positive rate when
    we predict probability in this bin).

    Lower is better. ECE = 0 means perfectly calibrated (e.g. when the model
    says 0.7 the empirical positive rate is 70%). Typical well-calibrated
    deep models score 0.01-0.05; uncalibrated softmax models score 0.10-0.30.

    Args:
        probs: Predicted positive-class probabilities, shape ``[N, K]`` or
            ``[N]``. Values must be in ``[0, 1]``.
        labels: Binary ground-truth labels, same shape as ``probs``.
        n_bins: Number of equal-width bins on ``[0, 1]``. 15 is the
            Guo et al. default; use 10-20 in practice.

    Returns:
        ECE in ``[0, 1]``. Returns ``float('nan')`` if no observations.
    """
    probs = np.asarray(probs, dtype=float).ravel()
    labels = np.asarray(labels, dtype=float).ravel()

    if probs.size == 0:
        return float("nan")
    if probs.shape != labels.shape:
        raise ValueError(
            f"probs/labels shape mismatch: {probs.shape} vs {labels.shape}"
        )

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    # np.digitize returns 1..n_bins for in-range values; clip to handle
    # exact-1.0 values which would otherwise spill into bin n_bins+1.
    bin_ids = np.clip(np.digitize(probs, bin_edges[1:-1], right=False), 0, n_bins - 1)

    n = float(probs.size)
    ece = 0.0
    for b in range(n_bins):
        mask = bin_ids == b
        n_b = int(mask.sum())
        if n_b == 0:
            continue
        conf_b = float(probs[mask].mean())
        acc_b = float(labels[mask].mean())
        ece += (n_b / n) * abs(conf_b - acc_b)
    return float(ece)


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """Brier score (mean squared error) for multi-label probabilities.

    | Applies to           | Task         |
    |----------------------|--------------|
    | Multi-label / binary | Calibration  |

    For binary / multi-label problems::

        Brier = mean_{n,k} (probs[n,k] - labels[n,k])^2

    Strictly proper scoring rule: minimised iff predicted probabilities equal
    the true conditional positive rate. Decomposes into reliability (calibration)
    + resolution (sharpness) - uncertainty (irreducible).

    Range: ``[0, 1]``. Lower is better.

    * Always-predict-prevalence baseline: ``Brier = p*(1-p)`` per label.
    * Perfect predictor: ``Brier = 0``.

    Args:
        probs: Predicted probabilities, shape ``[N, K]`` or ``[N]``.
        labels: Binary ground truth, same shape.

    Returns:
        Brier score. ``float('nan')`` if empty.
    """
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=float)
    if probs.size == 0:
        return float("nan")
    if probs.shape != labels.shape:
        raise ValueError(
            f"probs/labels shape mismatch: {probs.shape} vs {labels.shape}"
        )
    return float(np.mean((probs - labels) ** 2))


def expected_calibration_error_multiclass(
    probs: np.ndarray,
    y: np.ndarray,
    n_bins: int = 15,
) -> float:
    """ECE for multi-class softmax (Guo 2017 form).

    | Applies to                | Task         |
    |---------------------------|--------------|
    | Multi-class (single-label)| Calibration  |

    Uses **top-1 confidence** (``max_k probs[n, k]``) and **top-1
    correctness** (``argmax_k probs[n, k] == y[n]``) — the canonical
    multi-class reliability-diagram protocol from Guo et al. 2017.

    For multi-label / per-label binary calibration, use
    ``expected_calibration_error`` instead.

    Args:
        probs: Softmax probabilities, shape ``[N, K]``. Each row should
            sum to 1.
        y: Integer class labels in ``[0, K)``, shape ``[N]``.
        n_bins: Number of equal-width bins on ``[0, 1]``.

    Returns:
        ECE in ``[0, 1]``. Returns ``float('nan')`` if no observations.
    """
    probs = np.asarray(probs, dtype=float)
    y = np.asarray(y).astype(int)
    if probs.ndim != 2:
        raise ValueError(f"probs must be 2-D [N, K], got shape {probs.shape}")
    if probs.shape[0] != y.shape[0]:
        raise ValueError(
            f"probs/y length mismatch: {probs.shape[0]} vs {y.shape[0]}"
        )
    if probs.shape[0] == 0:
        return float("nan")

    if probs.size and (probs.min() < 0.0 or probs.max() > 1.0 + 1e-6):
        raise ValueError(
            "expected_calibration_error_multiclass: probs out of [0, 1] "
            f"(min={probs.min():.4f}, max={probs.max():.4f}). Did you "
            "pass raw logits? Apply softmax first."
        )

    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == y).astype(float)
    return expected_calibration_error(confidences, correct, n_bins=n_bins)


def brier_score_multiclass(probs: np.ndarray, y: np.ndarray) -> float:
    """Multi-class Brier score (one-hot targets vs softmax probs).

    | Applies to                | Task         |
    |---------------------------|--------------|
    | Multi-class (single-label)| Calibration  |

    Computes::

        Brier = mean_n sum_k (probs[n, k] - onehot(y)[n, k])^2

    The sum-over-classes form (range ``[0, 2]``) is the standard
    multi-class definition. For multi-label / per-label binary Brier,
    use ``brier_score`` instead.

    Args:
        probs: Softmax probabilities, shape ``[N, K]``.
        y: Integer class labels in ``[0, K)``, shape ``[N]``.

    Returns:
        Brier score in ``[0, 2]``. Lower is better.
    """
    probs = np.asarray(probs, dtype=float)
    y = np.asarray(y).astype(int)
    if probs.ndim != 2:
        raise ValueError(f"probs must be 2-D [N, K], got shape {probs.shape}")
    if probs.shape[0] != y.shape[0]:
        raise ValueError(
            f"probs/y length mismatch: {probs.shape[0]} vs {y.shape[0]}"
        )
    if probs.shape[0] == 0:
        return float("nan")

    if probs.size and (probs.min() < 0.0 or probs.max() > 1.0 + 1e-6):
        raise ValueError(
            "brier_score_multiclass: probs out of [0, 1] "
            f"(min={probs.min():.4f}, max={probs.max():.4f}). Did you "
            "pass raw logits? Apply softmax first."
        )

    N, K = probs.shape
    onehot = np.zeros((N, K), dtype=float)
    onehot[np.arange(N), y] = 1.0
    return float(np.mean(((probs - onehot) ** 2).sum(axis=1)))

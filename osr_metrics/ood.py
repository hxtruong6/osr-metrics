"""Binary OOD-detection metrics (task-agnostic).

**Scope:** task-agnostic — operate on a 1-D score array and binary
ID-vs-OOD labels. Applicable to multi-class (single-label), multi-label,
and any setting where you can produce a per-sample novelty score.

**Score-direction convention:** higher = more OOD. ID-positive metrics
(``aupr_in``) handle the sign flip internally.
"""
from __future__ import annotations

import os
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from sklearn.metrics import roc_curve

# Bootstrap resampling can produce all-ID or all-OOD samples; sklearn warns.
warnings.filterwarnings("ignore", message="No positive class found in y_true")


def _validate_scores_labels(scores, labels, fn_name: str) -> None:
    """Shape / dtype / value checks shared by binary OOD metrics."""
    s = np.asarray(scores)
    y = np.asarray(labels)
    if s.ndim != 1:
        raise ValueError(
            f"{fn_name}: scores must be 1-D [N], got shape {s.shape}"
        )
    if y.ndim != 1:
        raise ValueError(
            f"{fn_name}: labels must be 1-D [N], got shape {y.shape}. "
            "If you have a multi-hot label matrix, pass an OOD indicator "
            "(1 = OOD, 0 = ID) instead."
        )
    if s.shape[0] != y.shape[0]:
        raise ValueError(
            f"{fn_name}: scores/labels length mismatch: "
            f"{s.shape[0]} vs {y.shape[0]}"
        )
    uniq = np.unique(y)
    if not np.all(np.isin(uniq, [0, 1])):
        raise ValueError(
            f"{fn_name}: labels must be binary {{0, 1}}; got values {uniq}. "
            "1 = OOD, 0 = in-distribution."
        )


def fpr_at_tpr(scores: np.ndarray, labels: np.ndarray, target_tpr: float = 0.95) -> float:
    """Compute False Positive Rate at a given True Positive Rate.

    | Applies to | Task             |
    |------------|------------------|
    | Any        | OOD detection    |

    Args:
        scores: OOD scores (higher = more OOD). Shape [N].
        labels: binary ground truth. 1 = OOD, 0 = in-distribution. Shape [N].
        target_tpr: TPR level to evaluate at (default: 0.95).

    Returns:
        FPR at the given TPR in [0, 1]. Lower is better.
    """
    _validate_scores_labels(scores, labels, "fpr_at_tpr")
    if not 0.0 < target_tpr <= 1.0:
        raise ValueError(
            f"fpr_at_tpr: target_tpr must be in (0, 1]; got {target_tpr}"
        )
    fpr, tpr, _ = roc_curve(labels, scores)
    idx = int(np.searchsorted(tpr, target_tpr))
    idx = min(idx, len(fpr) - 1)
    return float(fpr[idx])


def fpr_at_95tpr(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute False Positive Rate at 95% True Positive Rate.

    | Applies to | Task             |
    |------------|------------------|
    | Any        | OOD detection    |
    """
    return fpr_at_tpr(scores, labels, target_tpr=0.95)


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area Under the ROC Curve for OOD detection.

    | Applies to | Task             |
    |------------|------------------|
    | Any        | OOD detection    |
    """
    _validate_scores_labels(scores, labels, "auroc")
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(labels, scores))


def aupr_in(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area Under the Precision-Recall Curve with ID as positive class (AUPR-In).

    | Applies to | Task             |
    |------------|------------------|
    | Any        | OOD detection    |

    Args:
        scores: OOD scores (higher = more OOD). Shape [N].
        labels: binary ground truth. 1 = OOD, 0 = in-distribution. Shape [N].

    Returns:
        AUPR-In in [0, 1]. Higher is better.
    """
    _validate_scores_labels(scores, labels, "aupr_in")
    from sklearn.metrics import average_precision_score
    # ID is positive class: negate scores so ID images score higher
    return float(average_precision_score(1 - np.asarray(labels), -np.asarray(scores)))


def aupr_out(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area Under the Precision-Recall Curve with OOD as positive class (AUPR-Out).

    | Applies to | Task             |
    |------------|------------------|
    | Any        | OOD detection    |

    Args:
        scores: OOD scores (higher = more OOD). Shape [N].
        labels: binary ground truth. 1 = OOD, 0 = in-distribution. Shape [N].

    Returns:
        AUPR-Out in [0, 1]. Higher is better.
    """
    _validate_scores_labels(scores, labels, "aupr_out")
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(labels, scores))


def partition_ood_by_purity(
    label_vecs: np.ndarray,
    ood_mask: np.ndarray,
    label_names: list[str],
    held_out_labels: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Partition OOD images into pure-OOD and mixed-OOD subsets.

    | Applies to   | Task                        |
    |--------------|-----------------------------|
    | Multi-label  | OSR sub-population analysis |

    Pure OOD: images where ALL positive labels are in held_out_labels.
    Mixed OOD: OOD images that also have positive non-held-out labels.

    Args:
        label_vecs: shape [N, K] binary label matrix for all images.
        ood_mask: shape [N] boolean mask (True = OOD image).
        label_names: list of K label names (column order).
        held_out_labels: list of held-out label names.

    Returns:
        (pure_ood_mask, mixed_ood_mask): boolean masks over the full N images.
    """
    held_out_indices = {i for i, l in enumerate(label_names) if l in held_out_labels}
    non_held_out_indices = [i for i in range(len(label_names)) if i not in held_out_indices]

    # For each OOD image, check if it has any positive non-held-out labels
    has_non_held_out = label_vecs[:, non_held_out_indices].sum(axis=1) > 0

    pure_ood_mask = ood_mask & ~has_non_held_out
    mixed_ood_mask = ood_mask & has_non_held_out

    return pure_ood_mask, mixed_ood_mask


def oscr_curve(
    novelty_scores: np.ndarray,
    labels_ood: np.ndarray,
    cls_correct: np.ndarray,
    n_thresholds: int | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Open-Set Classification Rate (OSCR) curve (Dhamija et al. 2018).

    | Applies to                                        | Task                  |
    |---------------------------------------------------|-----------------------|
    | Multi-class (single-label); multi-label via flag  | OSR (classify+reject) |

    Multi-label callers must reduce per-sample correctness to a single
    indicator (1 = all labels predicted correctly, else 0) and pass it as
    ``cls_correct``.

    Sweeps a novelty threshold tau. At each tau:
      * a sample is **accepted** iff ``novelty_score <= tau``;
      * **FPR** = fraction of OOD samples (incorrectly) accepted;
      * **CCR** = fraction of ID samples that are accepted AND correctly classified.

    AOSCR is the area under the (FPR, CCR) curve in [0, 1]; higher is better.

    Canonical convention: this function follows Dhamija 2018 / Vaze 2022
    (FPR = OOD-acceptance rate, CCR = correct ID classification rate).
    Use ``compute_aoscr`` for the same metric invoked by an explicit
    (predictions, true_classes) interface.

    Args:
        novelty_scores: open-set scores (higher = more OOD). Shape [N].
        labels_ood: binary ground truth. 1 = OOD, 0 = ID. Shape [N].
        cls_correct: binary indicator of correct closed-set classification
            for each sample (1 = correct, 0 = incorrect). Shape [N].
            Only meaningful for ID samples; OOD values are ignored.
        n_thresholds: Deprecated and ignored. Curves now have at most
            ``N + 1`` points (one per unique score plus a ``(0, 0)`` anchor).

    Returns:
        (fpr_array, ccr_array, aoscr) with arrays sorted by FPR ascending.
    """
    if n_thresholds is not None:
        warnings.warn(
            "`n_thresholds` is deprecated and ignored; remove the argument.",
            DeprecationWarning,
            stacklevel=2,
        )
    from .osr import _oscr_curve_points
    return _oscr_curve_points(novelty_scores, labels_ood, cls_correct)


def bootstrap_ci(
    scores: np.ndarray,
    labels: np.ndarray,
    metric_fn,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
    stratify: bool = False,
    n_jobs: int = 1,
) -> tuple[float, float, float]:
    """Percentile bootstrap confidence interval for any scalar metric.

    | Applies to | Task                  |
    |------------|-----------------------|
    | Any        | Statistical comparison|

    Args:
        scores: per-sample scores, shape ``[N]``.
        labels: per-sample binary labels, shape ``[N]``.
        metric_fn: callable ``f(scores, labels) -> float``.
        n_bootstrap: number of bootstrap resamples.
        ci: confidence level in ``(0, 1)`` (e.g. 0.95).
        seed: RNG seed for reproducibility.
        stratify: if ``True``, resample positives and negatives separately
            so each bootstrap replicate has the original class proportion.
            Required when one class is rare (otherwise replicates can be
            all-positive or all-negative, producing NaN metrics).
        n_jobs: thread count for evaluating ``metric_fn``. ``1`` (default)
            is serial; ``-1`` is ``os.cpu_count()``. Clamped to
            ``min(n_jobs, n_bootstrap, os.cpu_count())``. Bit-exact with
            the serial path for any ``seed`` (indices are drawn in the
            main thread). Threading (not multiprocessing) is used so
            lambdas and closures work; sklearn metrics release the GIL.
            Useful only when the serial bootstrap takes more than a few
            seconds — at small N the dispatch overhead dominates.

    Returns:
        ``(lower, mean, upper)`` where ``lower``/``upper`` are the
        ``(1-ci)/2`` and ``1-(1-ci)/2`` percentiles of the bootstrap
        distribution. NaN replicates are dropped before percentile / mean
        calculation.
    """
    rng = np.random.RandomState(seed)
    n = len(scores)
    scores = np.asarray(scores)
    labels_arr = np.asarray(labels)

    if stratify:
        pos_idx = np.where(labels_arr == 1)[0]
        neg_idx = np.where(labels_arr == 0)[0]
        n_pos, n_neg = len(pos_idx), len(neg_idx)
        if n_pos == 0 or n_neg == 0:
            raise ValueError("stratify=True requires both classes to be present")

    # Draw all indices up front so parallel and serial paths share an RNG sequence.
    sampled_indices: list[np.ndarray] = []
    for _ in range(n_bootstrap):
        if stratify:
            sampled_indices.append(np.concatenate([
                rng.choice(pos_idx, size=n_pos, replace=True),
                rng.choice(neg_idx, size=n_neg, replace=True),
            ]))
        else:
            sampled_indices.append(rng.choice(n, size=n, replace=True))

    def _eval(idx: np.ndarray) -> float:
        return float(metric_fn(scores[idx], labels_arr[idx]))

    cpu = os.cpu_count() or 1
    requested = cpu if n_jobs == -1 else n_jobs
    max_workers = max(1, min(requested, n_bootstrap, cpu))

    if max_workers == 1:
        bootstrap_vals = np.fromiter(
            (_eval(idx) for idx in sampled_indices),
            dtype=float,
            count=n_bootstrap,
        )
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            bootstrap_vals = np.fromiter(
                ex.map(_eval, sampled_indices),
                dtype=float,
                count=n_bootstrap,
            )

    valid = bootstrap_vals[~np.isnan(bootstrap_vals)]
    if valid.size == 0:
        return float("nan"), float("nan"), float("nan")

    alpha = (1 - ci) / 2
    lower = float(np.percentile(valid, 100 * alpha))
    upper = float(np.percentile(valid, 100 * (1 - alpha)))
    return lower, float(np.mean(valid)), upper


def paired_bootstrap_diff(
    metric_fn,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    labels: np.ndarray,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
    stratify: bool = False,
) -> tuple[float, float, float, float]:
    """Paired bootstrap CI on ``metric(scores_a) - metric(scores_b)``.

    | Applies to | Task                     |
    |------------|--------------------------|
    | Any        | Statistical comparison   |

    Method ``a`` and method ``b`` are evaluated on the **same** bootstrap
    resample at each replicate (paired design), removing variance
    attributable to the sampled instances and yielding a tighter CI on
    the delta than two independent bootstraps.

    The two-sided p-value is computed against the null
    ``H_0: metric(a) = metric(b)`` as
    ``2 * min(P(diff >= 0), P(diff <= 0))`` over the bootstrap
    distribution. With ``n_bootstrap = 1000`` the smallest resolvable
    p-value is ``2 / 1000 = 0.002``; raise ``n_bootstrap`` for
    smaller-p reporting.

    Use this for non-AUROC paired comparisons (macro-F1, AUPR, AOSCR,
    AURC, AML-OSCR, …). For AUROC specifically, prefer ``delong_test``
    which is the analytic asymptotic test.

    Args:
        metric_fn: callable ``f(scores, labels) -> float``.
        scores_a: per-sample scores from method ``a``, shape ``[N]``.
        scores_b: per-sample scores from method ``b``, shape ``[N]``.
        labels: per-sample labels, shape ``[N]``.
        n_bootstrap: number of bootstrap resamples.
        ci: confidence level in ``(0, 1)``.
        seed: RNG seed.
        stratify: if ``True``, resample positives and negatives
            separately to preserve class proportion (recommended on
            rare-positive problems).

    Returns:
        ``(delta_mean, lower, upper, p_two_sided)``. ``delta_mean`` is the
        mean of ``metric(a) - metric(b)`` over bootstrap replicates;
        ``(lower, upper)`` are the percentile CI bounds. NaN replicates
        (either side) are dropped before percentile / mean / p-value
        calculation.
    """
    scores_a = np.asarray(scores_a)
    scores_b = np.asarray(scores_b)
    labels_arr = np.asarray(labels)
    n = scores_a.shape[0]
    if scores_b.shape[0] != n or labels_arr.shape[0] != n:
        raise ValueError(
            "paired_bootstrap_diff: scores_a, scores_b, labels must have "
            f"the same length; got {scores_a.shape[0]}, {scores_b.shape[0]}, "
            f"{labels_arr.shape[0]}"
        )

    rng = np.random.RandomState(seed)
    if stratify:
        pos_idx = np.where(labels_arr == 1)[0]
        neg_idx = np.where(labels_arr == 0)[0]
        n_pos, n_neg = len(pos_idx), len(neg_idx)
        if n_pos == 0 or n_neg == 0:
            raise ValueError("stratify=True requires both classes to be present")

    diffs = np.empty(n_bootstrap, dtype=float)
    for b in range(n_bootstrap):
        idx: np.ndarray
        if stratify:
            idx = np.concatenate([
                rng.choice(pos_idx, size=n_pos, replace=True),
                rng.choice(neg_idx, size=n_neg, replace=True),
            ])
        else:
            idx = np.asarray(rng.choice(n, size=n, replace=True))
        try:
            va = float(metric_fn(scores_a[idx], labels_arr[idx]))
            vb = float(metric_fn(scores_b[idx], labels_arr[idx]))
            diffs[b] = va - vb
        except Exception:
            diffs[b] = float("nan")

    valid = diffs[~np.isnan(diffs)]
    if valid.size == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")

    alpha = (1 - ci) / 2
    lower = float(np.percentile(valid, 100 * alpha))
    upper = float(np.percentile(valid, 100 * (1 - alpha)))
    delta_mean = float(np.mean(valid))

    p_ge = float((valid >= 0).sum()) / valid.size
    p_le = float((valid <= 0).sum()) / valid.size
    p_two_sided = float(min(1.0, 2.0 * min(p_ge, p_le)))

    return delta_mean, lower, upper, p_two_sided

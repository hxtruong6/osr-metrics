from __future__ import annotations

import warnings

import numpy as np
from sklearn.metrics import roc_curve

# Bootstrap resampling can produce all-ID or all-OOD samples; sklearn warns.
warnings.filterwarnings("ignore", message="No positive class found in y_true")


def fpr_at_tpr(scores: np.ndarray, labels: np.ndarray, target_tpr: float = 0.95) -> float:
    """Compute False Positive Rate at a given True Positive Rate.

    Args:
        scores: OOD scores (higher = more OOD). Shape [N].
        labels: binary ground truth. 1 = OOD, 0 = in-distribution. Shape [N].
        target_tpr: TPR level to evaluate at (default: 0.95).

    Returns:
        FPR at the given TPR in [0, 1]. Lower is better.
    """
    fpr, tpr, _ = roc_curve(labels, scores)
    idx = np.searchsorted(tpr, target_tpr)
    idx = min(idx, len(fpr) - 1)
    return float(fpr[idx])


def fpr_at_95tpr(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute False Positive Rate at 95% True Positive Rate."""
    return fpr_at_tpr(scores, labels, target_tpr=0.95)


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area Under the ROC Curve for OOD detection."""
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(labels, scores))


def aupr_in(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area Under the Precision-Recall Curve with ID as positive class (AUPR-In).

    Args:
        scores: OOD scores (higher = more OOD). Shape [N].
        labels: binary ground truth. 1 = OOD, 0 = in-distribution. Shape [N].

    Returns:
        AUPR-In in [0, 1]. Higher is better.
    """
    from sklearn.metrics import average_precision_score
    # ID is positive class: negate scores so ID images score higher
    return float(average_precision_score(1 - labels, -scores))


def aupr_out(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area Under the Precision-Recall Curve with OOD as positive class (AUPR-Out).

    Args:
        scores: OOD scores (higher = more OOD). Shape [N].
        labels: binary ground truth. 1 = OOD, 0 = in-distribution. Shape [N].

    Returns:
        AUPR-Out in [0, 1]. Higher is better.
    """
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(labels, scores))


def partition_ood_by_purity(
    label_vecs: np.ndarray,
    ood_mask: np.ndarray,
    label_names: list[str],
    held_out_labels: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Partition OOD images into pure-OOD and mixed-OOD subsets.

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
    n_thresholds: int = 1000,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Open-Set Classification Rate (OSCR) curve (Dhamija et al. 2018).

    Sweeps a novelty threshold tau. At each tau:
      * a sample is **accepted** iff ``novelty_score <= tau``;
      * **FPR** = fraction of OOD samples (incorrectly) accepted;
      * **CCR** = fraction of ID samples that are accepted AND correctly classified.

    AOSCR is the area under the (FPR, CCR) curve in [0, 1]; higher is better.

    NOTE (canonical convention): this function follows Dhamija 2018 / Vaze 2022.
    Earlier versions in this repo computed FPR as the *ID-rejection rate*
    (an internally consistent but non-canonical variant). All call-sites have
    been migrated. Use ``src.metrics.osr.compute_aoscr`` for the same metric
    invoked by an explicit (predictions, true_classes) interface.

    Args:
        novelty_scores: open-set scores (higher = more OOD). Shape [N].
        labels_ood: binary ground truth. 1 = OOD, 0 = ID. Shape [N].
        cls_correct: binary indicator of correct closed-set classification
            for each sample (1 = correct, 0 = incorrect). Shape [N].
            Only meaningful for ID samples; OOD values are ignored.
        n_thresholds: number of threshold steps for the curve.

    Returns:
        (fpr_array, ccr_array, aoscr) with arrays sorted by FPR ascending.
    """
    novelty_scores = np.asarray(novelty_scores, dtype=float)
    labels_ood = np.asarray(labels_ood).astype(int)
    cls_correct = np.asarray(cls_correct).astype(int)

    id_mask = labels_ood == 0
    ood_mask = labels_ood == 1
    n_id = int(id_mask.sum())
    n_ood = int(ood_mask.sum())

    if n_id == 0 or n_ood == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 0.0]), 0.0

    lo, hi = float(novelty_scores.min()), float(novelty_scores.max())
    if lo == hi:
        return np.array([0.0, 1.0]), np.array([0.0, 0.0]), 0.0

    thresholds = np.linspace(lo, hi, n_thresholds)

    id_scores = novelty_scores[id_mask]
    id_correct = cls_correct[id_mask]
    ood_scores = novelty_scores[ood_mask]

    fpr_list = np.empty(n_thresholds, dtype=float)
    ccr_list = np.empty(n_thresholds, dtype=float)

    for i, tau in enumerate(thresholds):
        # Accepted = score <= tau.
        # CCR: ID samples accepted AND classified correctly.
        ccr_list[i] = float(((id_scores <= tau) & (id_correct == 1)).sum()) / n_id
        # FPR: OOD samples (wrongly) accepted.
        fpr_list[i] = float((ood_scores <= tau).sum()) / n_ood

    sort_idx = np.argsort(fpr_list)
    fpr_sorted = fpr_list[sort_idx]
    ccr_sorted = ccr_list[sort_idx]

    _trapz = getattr(np, "trapezoid", None) or np.trapz
    aoscr = float(_trapz(ccr_sorted, fpr_sorted))
    return fpr_sorted, ccr_sorted, aoscr


def bootstrap_ci(
    scores: np.ndarray,
    labels: np.ndarray,
    metric_fn,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
    stratify: bool = False,
) -> tuple[float, float, float]:
    """Percentile bootstrap confidence interval for any scalar metric.

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

    Returns:
        ``(lower, mean, upper)`` where ``lower``/``upper`` are the
        ``(1-ci)/2`` and ``1-(1-ci)/2`` percentiles of the bootstrap
        distribution. NaN replicates are dropped before percentile / mean
        calculation.
    """
    rng = np.random.RandomState(seed)
    n = len(scores)

    if stratify:
        labels_arr = np.asarray(labels)
        pos_idx = np.where(labels_arr == 1)[0]
        neg_idx = np.where(labels_arr == 0)[0]
        n_pos, n_neg = len(pos_idx), len(neg_idx)
        if n_pos == 0 or n_neg == 0:
            raise ValueError("stratify=True requires both classes to be present")

    bootstrap_vals = np.empty(n_bootstrap, dtype=float)
    for b in range(n_bootstrap):
        if stratify:
            sampled = np.concatenate([
                rng.choice(pos_idx, size=n_pos, replace=True),
                rng.choice(neg_idx, size=n_neg, replace=True),
            ])
        else:
            sampled = rng.choice(n, size=n, replace=True)
        bootstrap_vals[b] = float(metric_fn(scores[sampled], labels[sampled]))

    valid = bootstrap_vals[~np.isnan(bootstrap_vals)]
    if valid.size == 0:
        return float("nan"), float("nan"), float("nan")

    alpha = (1 - ci) / 2
    lower = float(np.percentile(valid, 100 * alpha))
    upper = float(np.percentile(valid, 100 * (1 - alpha)))
    return lower, float(np.mean(valid)), upper

"""DeLong test for comparing two AUROC curves on the same data.

**Scope: task-agnostic.** Operates on two score arrays + shared binary
labels. Use whenever you have AUROC for two methods on the *same*
samples (multi-class OOD, multi-label OSR, anomaly detection — anywhere
``auroc`` applies).

Implements the O(n log n) DeLong test using placement values and
covariance estimation, following DeLong et al. (1988).
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def _placement_values(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute placement values for positive and negative cases (O(n log n)).

    For each positive case, the placement value is the fraction of negatives
    scored below it (with 0.5 credit for ties). For each negative case, it is
    the fraction of positives scored above it (with 0.5 credit for ties).

    Implementation: uses ``scipy.stats.rankdata`` mid-rank ranking on the
    combined score vector, then derives within-group placements from the
    standard identity::

        rank_total(x) = rank_within_group(x) + n_other_below(x) + 0.5 * n_other_tied(x)

    Numerically equivalent to the O(n^2) brute-force version to ~1e-12.
    """
    pos_mask = labels == 1
    neg_mask = labels == 0

    pos_scores = scores[pos_mask]
    neg_scores = scores[neg_mask]

    n_pos = pos_scores.size
    n_neg = neg_scores.size

    if n_pos == 0 or n_neg == 0:
        return np.zeros(n_pos), np.zeros(n_neg)

    # Mid-ranks across the union: for tied values, all tied entries get the
    # average of their would-be ranks, which gives 0.5 credit for ties.
    all_ranks = stats.rankdata(scores, method="average")
    pos_ranks_total = all_ranks[pos_mask]
    neg_ranks_total = all_ranks[neg_mask]

    # Within-group mid-ranks.
    pos_ranks_within = stats.rankdata(pos_scores, method="average")
    neg_ranks_within = stats.rankdata(neg_scores, method="average")

    # For positives we want (# negs below + 0.5 * # negs tied).
    # rank_total - rank_within gives exactly that for the OTHER group.
    v_pos = (pos_ranks_total - pos_ranks_within) / n_neg
    # For negatives we want (# pos ABOVE + 0.5 * # pos tied).
    # rank_total - rank_within gives (# pos BELOW + 0.5 * # pos tied),
    # so flip via the identity v_above + v_below + v_tied = 1 with mid-rank credit.
    v_neg = 1.0 - (neg_ranks_total - neg_ranks_within) / n_pos

    return v_pos, v_neg


def delong_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, float]:
    """Compare two AUROC curves on the same data using the DeLong test.

    | Applies to | Task                                |
    |------------|-------------------------------------|
    | Any        | Statistical comparison (paired AUROC)|

    Both methods must be evaluated on the **same samples** with the
    **same labels**; the test is paired.

    Args:
        scores_a: OOD scores from method A. Shape [N].
        scores_b: OOD scores from method B. Shape [N].
        labels: Binary ground truth. 1 = OOD, 0 = ID. Shape [N].

    Returns:
        (z_stat, p_value): two-sided test statistic and p-value.
        p_value near 1.0 means no significant difference;
        p_value < 0.05 means significantly different AUROCs.
    """
    scores_a = np.asarray(scores_a, dtype=np.float64)
    scores_b = np.asarray(scores_b, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)

    # Placement values for both methods
    v_pos_a, v_neg_a = _placement_values(scores_a, labels)
    v_pos_b, v_neg_b = _placement_values(scores_b, labels)

    n_pos = (labels == 1).sum()
    n_neg = (labels == 0).sum()

    # AUC estimates
    auc_a = np.mean(v_pos_a)
    auc_b = np.mean(v_pos_b)

    # Covariance matrix of (AUC_a, AUC_b)
    # S10: covariance from positive cases
    diff_pos = np.column_stack([v_pos_a - auc_a, v_pos_b - auc_b])
    s10 = diff_pos.T @ diff_pos / (n_pos - 1) if n_pos > 1 else np.zeros((2, 2))

    # S01: covariance from negative cases
    diff_neg = np.column_stack([v_neg_a - (1 - auc_a), v_neg_b - (1 - auc_b)])
    s01 = diff_neg.T @ diff_neg / (n_neg - 1) if n_neg > 1 else np.zeros((2, 2))

    # Combined covariance of AUC difference
    s = s10 / n_pos + s01 / n_neg

    # Variance of (AUC_a - AUC_b)
    var_diff = s[0, 0] + s[1, 1] - 2 * s[0, 1]

    if var_diff < 1e-16:
        # AUCs are identical or variance is zero
        return 0.0, 1.0

    z = (auc_a - auc_b) / np.sqrt(var_diff)
    p_value = 2 * stats.norm.sf(abs(z))

    return float(z), float(p_value)

"""Regression tests for the three bugs fixed in v0.1.0:

1. ``oscr_curve`` now follows the canonical Dhamija/Vaze convention
   (FPR = OOD acceptance rate), agreeing with ``compute_aoscr``.
2. ``delong._placement_values`` is now O(n log n) rank-based, numerically
   equivalent to the brute-force version.
3. ``bootstrap_ci`` accepts ``stratify=True`` for imbalanced data.
"""
import numpy as np
import pytest

from osr_metrics import auroc, bootstrap_ci, compute_aoscr, oscr_curve
from osr_metrics.delong import _placement_values, delong_test


def test_oscr_curve_matches_compute_aoscr():
    """After the fix, oscr_curve and compute_aoscr produce identical AOSCR."""
    rng = np.random.default_rng(42)
    N = 600
    labels_ood = rng.integers(0, 2, N)
    scores = rng.normal(loc=labels_ood * 0.8, scale=1.0)
    cls_correct = rng.integers(0, 2, N)

    _, _, aoscr_v1 = oscr_curve(scores, labels_ood, cls_correct)
    aoscr_v2 = compute_aoscr(scores, labels_ood, cls_correct, np.ones(N))

    assert aoscr_v1 == pytest.approx(aoscr_v2, abs=1e-9)


def _brute_placements(scores, labels):
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    n_pos, n_neg = len(pos), len(neg)
    v_pos = np.zeros(n_pos)
    v_neg = np.zeros(n_neg)
    for i, s in enumerate(pos):
        v_pos[i] = (np.sum(neg < s) + 0.5 * np.sum(neg == s)) / n_neg
    for i, s in enumerate(neg):
        v_neg[i] = (np.sum(pos > s) + 0.5 * np.sum(pos == s)) / n_pos
    return v_pos, v_neg


@pytest.mark.parametrize("n,with_ties", [(100, False), (500, True), (2000, True)])
def test_delong_rank_based_matches_brute_force(n, with_ties):
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, n)
    scores = rng.normal(loc=labels * 0.5, scale=1.0)
    if with_ties:
        scores = np.round(scores, 1)
    vp_new, vn_new = _placement_values(scores, labels)
    vp_ref, vn_ref = _brute_placements(scores, labels)
    np.testing.assert_allclose(vp_new, vp_ref, atol=1e-12)
    np.testing.assert_allclose(vn_new, vn_ref, atol=1e-12)


def test_delong_identical_inputs():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, 200)
    scores = rng.normal(0, 1, 200)
    z, p = delong_test(scores, scores, labels)
    assert z == pytest.approx(0.0, abs=1e-9)
    assert p == pytest.approx(1.0, abs=1e-9)


def test_delong_anti_correlated_significant():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, 500)
    scores = rng.normal(loc=labels * 0.5, scale=1.0)
    z, p = delong_test(scores, -scores, labels)
    # Anti-correlated scores must produce a significant difference; the
    # exact z depends on N and effect size, so use a moderate threshold.
    assert abs(z) > 2.5
    assert p < 0.05


def test_bootstrap_ci_stratify_handles_imbalance():
    """With 980 ID + 20 OOD, every replicate must have both classes when stratified."""
    rng = np.random.default_rng(0)
    n_id, n_ood = 980, 20
    labels = np.concatenate([np.zeros(n_id), np.ones(n_ood)])
    scores = np.concatenate([rng.normal(0, 1, n_id), rng.normal(0.8, 1, n_ood)])

    lo, mean, hi = bootstrap_ci(
        scores, labels, auroc, n_bootstrap=200, seed=0, stratify=True
    )
    assert not np.isnan(mean)
    assert lo <= mean <= hi
    # Sanity: the bootstrap mean should be close to the point estimate.
    point = auroc(scores, labels)
    assert abs(mean - point) < 0.05


def test_bootstrap_ci_stratify_requires_both_classes():
    with pytest.raises(ValueError):
        bootstrap_ci(
            np.random.randn(100),
            np.zeros(100),
            auroc,
            n_bootstrap=50,
            stratify=True,
        )


def test_bootstrap_ci_unstratified_still_works():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, 500)
    scores = rng.normal(loc=labels * 0.5, scale=1.0)
    lo, mean, hi = bootstrap_ci(scores, labels, auroc, n_bootstrap=200, seed=0)
    assert lo <= mean <= hi

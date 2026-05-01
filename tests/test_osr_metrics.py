"""Tests for OSR (Open-Set Recognition) metrics: AOSCR + NF-rejection@TPR."""
import numpy as np
import pytest

from osr_metrics import oscr_curve
from osr_metrics.osr import compute_aoscr, compute_nf_rejection_at_tpr


# --------------------------------------------------------------------------
# AOSCR
# --------------------------------------------------------------------------

def test_aoscr_perfect_classifier():
    """Perfect setup: OOD scored high, all ID samples classified correctly.

    With perfect novelty separation AND perfect closed-set classification,
    the OSCR curve hits CCR=1 at FPR=0 and stays at 1, so AOSCR == 1.0.
    """
    n_id, n_ood = 50, 50
    # ID scores low, OOD scores high — perfect novelty separation.
    scores = np.concatenate([np.linspace(-2, -1, n_id), np.linspace(1, 2, n_ood)])
    ood_labels = np.concatenate([np.zeros(n_id), np.ones(n_ood)]).astype(int)
    # All ID samples correctly classified; OOD class doesn't matter for CCR.
    preds = np.concatenate([np.ones(n_id), np.zeros(n_ood)]).astype(int)
    truth = np.concatenate([np.ones(n_id), np.ones(n_ood)]).astype(int)

    aoscr = compute_aoscr(scores, ood_labels, preds, truth)
    # Exact implementation hits 1.0 to floating-point noise (no grid bias).
    assert aoscr == pytest.approx(1.0, abs=1e-12)


def test_aoscr_random_classifier():
    """Random scores + random predictions => AOSCR near 0.5 * accuracy."""
    rng = np.random.RandomState(0)
    n = 1000
    scores = rng.randn(n)
    ood_labels = rng.randint(0, 2, size=n)
    # Random closed-set accuracy ~50%.
    preds = rng.randint(0, 2, size=n)
    truth = rng.randint(0, 2, size=n)

    aoscr = compute_aoscr(scores, ood_labels, preds, truth)
    # Random scores keep area ~ 0.5 * P(correct) ~ 0.25; sampling noise allowed.
    assert 0.15 < aoscr < 0.40


def test_aoscr_realistic_midrange():
    """Moderately good detector: ID skewed low, OOD skewed high, ~80% accuracy.

    Expected AOSCR is well above random (0.25) but below perfect (1.0).
    """
    rng = np.random.RandomState(123)
    n_id, n_ood = 200, 200
    id_scores = rng.normal(loc=-0.5, scale=1.0, size=n_id)
    ood_scores = rng.normal(loc=0.5, scale=1.0, size=n_ood)
    scores = np.concatenate([id_scores, ood_scores])
    ood_labels = np.concatenate([np.zeros(n_id), np.ones(n_ood)]).astype(int)

    truth = np.ones(n_id + n_ood, dtype=int)
    preds = truth.copy()
    # Flip 20% of ID predictions => 80% closed-set accuracy.
    flip_idx = rng.choice(n_id, size=int(0.2 * n_id), replace=False)
    preds[flip_idx] = 0

    aoscr = compute_aoscr(scores, ood_labels, preds, truth)
    assert 0.45 < aoscr < 0.85


# --------------------------------------------------------------------------
# Vectorized AOSCR
# --------------------------------------------------------------------------

def test_aoscr_n_thresholds_deprecated():
    rng = np.random.default_rng(0)
    n = 200
    scores = rng.normal(size=n)
    ood_labels = rng.integers(0, 2, n)
    preds = rng.integers(0, 5, n)
    truth = rng.integers(0, 5, n)

    with pytest.warns(DeprecationWarning, match="n_thresholds"):
        compute_aoscr(scores, ood_labels, preds, truth, n_thresholds=500)


def test_oscr_curve_n_thresholds_deprecated():
    rng = np.random.default_rng(0)
    n = 200
    scores = rng.normal(size=n)
    ood_labels = rng.integers(0, 2, n)
    cls_correct = rng.integers(0, 2, n)

    with pytest.warns(DeprecationWarning, match="n_thresholds"):
        oscr_curve(scores, ood_labels, cls_correct, n_thresholds=500)


def test_oscr_curve_one_point_per_unique_score():
    rng = np.random.default_rng(1)
    n = 600
    scores = rng.normal(size=n)  # continuous → all unique
    ood_labels = rng.integers(0, 2, n)
    cls_correct = rng.integers(0, 2, n)

    fpr, ccr, aoscr = oscr_curve(scores, ood_labels, cls_correct)
    assert fpr.shape == (n + 1,)
    assert ccr.shape == (n + 1,)
    assert fpr[0] == 0.0 and ccr[0] == 0.0
    assert fpr[-1] == pytest.approx(1.0, abs=1e-12)
    assert 0.0 <= aoscr <= 1.0


def test_aoscr_invariant_to_score_ties():
    rng = np.random.default_rng(2)
    n = 400
    scores = np.round(rng.normal(size=n), 1)  # force ties
    ood_labels = rng.integers(0, 2, n)
    preds = rng.integers(0, 3, n)
    truth = rng.integers(0, 3, n)

    a1 = compute_aoscr(scores, ood_labels, preds, truth)
    perm = rng.permutation(n)
    a2 = compute_aoscr(scores[perm], ood_labels[perm], preds[perm], truth[perm])
    assert a1 == pytest.approx(a2, abs=1e-12)


# --------------------------------------------------------------------------
# NF-rejection @ TPR
# --------------------------------------------------------------------------

def test_nf_rejection_perfect():
    """Perfect: NF samples score above all ID-disease scores => rejection = 1.0."""
    # 50 ID-disease (score=0), 30 NF (score=2), 20 OOD (score=3).
    n_id, n_nf, n_ood = 50, 30, 20
    scores = np.concatenate([np.zeros(n_id), np.full(n_nf, 2.0), np.full(n_ood, 3.0)])
    ood_labels = np.concatenate([np.zeros(n_id), np.zeros(n_nf), np.ones(n_ood)]).astype(int)
    nf_labels = np.concatenate([np.zeros(n_id), np.ones(n_nf), np.zeros(n_ood)]).astype(int)

    rate = compute_nf_rejection_at_tpr(scores, ood_labels, nf_labels, tpr=0.95)
    assert rate == pytest.approx(1.0, abs=1e-6)


def test_nf_rejection_random():
    """Random scores: NF rejection rate at TPR=0.95 should be ~0.05 (the FPR)."""
    rng = np.random.RandomState(42)
    n_id, n_nf = 1000, 1000
    scores = rng.randn(n_id + n_nf)
    ood_labels = np.zeros(n_id + n_nf, dtype=int)
    nf_labels = np.concatenate([np.zeros(n_id), np.ones(n_nf)]).astype(int)

    rate = compute_nf_rejection_at_tpr(scores, ood_labels, nf_labels, tpr=0.95)
    # When NF and ID-disease come from the same distribution, the threshold
    # set at TPR=0.95 on ID-disease rejects ~5% of NF (sampling noise allowed).
    assert 0.0 <= rate <= 0.15


def test_nf_rejection_realistic_midrange():
    """Realistic: NF distribution shifted higher than ID-disease => moderate rejection."""
    rng = np.random.RandomState(7)
    n_id, n_nf = 500, 500
    id_scores = rng.normal(loc=0.0, scale=1.0, size=n_id)
    nf_scores = rng.normal(loc=1.0, scale=1.0, size=n_nf)
    scores = np.concatenate([id_scores, nf_scores])
    ood_labels = np.zeros(n_id + n_nf, dtype=int)
    nf_labels = np.concatenate([np.zeros(n_id), np.ones(n_nf)]).astype(int)

    rate = compute_nf_rejection_at_tpr(scores, ood_labels, nf_labels, tpr=0.95)
    # NF mean is 1 sigma above ID-disease => more NF above the 95th percentile of ID.
    assert 0.10 < rate < 0.60


def test_nf_rejection_empty_subset_returns_nan():
    """If either ID-disease or NF subset is empty, return NaN."""
    scores = np.array([0.1, 0.2, 0.3])
    ood_labels = np.array([1, 1, 1])  # no ID at all
    nf_labels = np.array([0, 0, 0])
    rate = compute_nf_rejection_at_tpr(scores, ood_labels, nf_labels)
    assert np.isnan(rate)

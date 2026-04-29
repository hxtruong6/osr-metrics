"""Tests for osr_metrics.selective."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import rankdata

from osr_metrics.selective import (
    _rc_curve_with_ties,
    _validate_coverage,
    _validate_score_loss,
    aurc,
    rc_curve,
)


class TestValidateScoreLoss:
    def test_accepts_matching_1d_arrays(self):
        score, loss = _validate_score_loss(
            np.array([0.1, 0.2, 0.3]),
            np.array([0.0, 1.0, 0.0]),
        )
        assert score.shape == (3,)
        assert loss.shape == (3,)
        assert score.dtype == np.float64
        assert loss.dtype == np.float64

    def test_accepts_lists(self):
        score, loss = _validate_score_loss([0.1, 0.2], [0.0, 1.0])
        assert score.shape == (2,) and loss.shape == (2,)

    def test_rejects_2d_score(self):
        with pytest.raises(ValueError, match="ood_score must be 1-D"):
            _validate_score_loss(np.zeros((3, 2)), np.zeros(3))

    def test_rejects_2d_loss(self):
        with pytest.raises(ValueError, match="loss must be 1-D"):
            _validate_score_loss(np.zeros(3), np.zeros((3, 2)))

    def test_rejects_shape_mismatch(self):
        with pytest.raises(ValueError, match="same shape"):
            _validate_score_loss(np.zeros(3), np.zeros(4))

    def test_rejects_too_few_samples(self):
        with pytest.raises(ValueError, match="at least 2"):
            _validate_score_loss(np.zeros(1), np.zeros(1))
        with pytest.raises(ValueError, match="at least 2"):
            _validate_score_loss(np.zeros(0), np.zeros(0))

    def test_rejects_nan_in_score(self):
        with pytest.raises(ValueError, match="ood_score contains NaN"):
            _validate_score_loss(np.array([0.1, np.nan]), np.array([0.0, 1.0]))

    def test_rejects_nan_in_loss(self):
        with pytest.raises(ValueError, match="loss contains NaN"):
            _validate_score_loss(np.array([0.1, 0.2]), np.array([0.0, np.nan]))

    def test_rejects_inf_in_score(self):
        with pytest.raises(ValueError, match="ood_score contains NaN or inf"):
            _validate_score_loss(np.array([0.1, np.inf]), np.array([0.0, 1.0]))

    def test_rejects_inf_in_loss(self):
        with pytest.raises(ValueError, match="loss contains NaN or inf"):
            _validate_score_loss(np.array([0.1, 0.2]), np.array([0.0, np.inf]))

    def test_rejects_negative_loss(self):
        with pytest.raises(ValueError, match="non-negative"):
            _validate_score_loss(np.array([0.1, 0.2]), np.array([0.0, -0.1]))


class TestValidateCoverage:
    def test_accepts_in_range(self):
        assert _validate_coverage(0.5) == 0.5
        assert _validate_coverage(1.0) == 1.0
        assert _validate_coverage(1e-9) == 1e-9

    def test_accepts_int(self):
        assert _validate_coverage(1) == 1.0

    def test_accepts_numpy_scalar(self):
        assert _validate_coverage(np.float64(0.5)) == 0.5

    def test_rejects_zero(self):
        with pytest.raises(ValueError, match=r"\(0, 1\]"):
            _validate_coverage(0.0)

    def test_rejects_negative(self):
        with pytest.raises(ValueError, match=r"\(0, 1\]"):
            _validate_coverage(-0.1)

    def test_rejects_above_one(self):
        with pytest.raises(ValueError, match=r"\(0, 1\]"):
            _validate_coverage(1.0001)

    def test_rejects_nan(self):
        with pytest.raises(ValueError, match="finite"):
            _validate_coverage(float("nan"))

    def test_rejects_array(self):
        with pytest.raises(TypeError, match="real scalar"):
            _validate_coverage(np.array([0.5]))

    def test_rejects_string(self):
        with pytest.raises(TypeError, match="real scalar"):
            _validate_coverage("0.5")


def _brute_force_rc_curve_no_ties(score, loss):
    """O(N^2) reference for the non-tied case.

    For k = 1..N, take the k samples with the lowest score and average
    their loss. Returns (coverage, risk, threshold) of length N.
    """
    score = np.asarray(score, dtype=float)
    loss = np.asarray(loss, dtype=float)
    n = len(score)
    order = np.argsort(score, kind="stable")
    coverage = np.empty(n, dtype=float)
    risk = np.empty(n, dtype=float)
    threshold = np.empty(n, dtype=float)
    for k in range(1, n + 1):
        sel = order[:k]
        coverage[k - 1] = k / n
        risk[k - 1] = float(np.mean(loss[sel]))
        threshold[k - 1] = float(score[sel[-1]])
    return coverage, risk, threshold


class TestRcCurveWithTies:
    def test_no_ties_matches_brute_force(self):
        for n in (10, 100, 500):
            for seed in range(5):
                rng2 = np.random.RandomState(seed)
                score = rng2.standard_normal(n)
                loss = rng2.binomial(1, 0.3, size=n).astype(float)
                cov, risk, thr = _rc_curve_with_ties(score, loss)
                bcov, brisk, bthr = _brute_force_rc_curve_no_ties(score, loss)
                np.testing.assert_allclose(cov, bcov, atol=1e-12)
                np.testing.assert_allclose(risk, brisk, atol=1e-12)
                np.testing.assert_allclose(thr, bthr, atol=1e-12)

    def test_returns_three_arrays_of_equal_length(self):
        cov, risk, thr = _rc_curve_with_ties(
            np.array([0.1, 0.2, 0.3, 0.4]),
            np.array([0.0, 1.0, 0.0, 1.0]),
        )
        assert cov.shape == risk.shape == thr.shape

    def test_coverage_strictly_increasing(self):
        rng = np.random.RandomState(1)
        score = rng.standard_normal(50)
        loss = rng.binomial(1, 0.4, size=50).astype(float)
        cov, _, _ = _rc_curve_with_ties(score, loss)
        assert np.all(np.diff(cov) > 0)

    def test_right_endpoint_is_overall_risk(self):
        score = np.array([0.1, 0.2, 0.3, 0.4])
        loss = np.array([0.0, 1.0, 0.0, 1.0])
        cov, risk, _ = _rc_curve_with_ties(score, loss)
        assert cov[-1] == 1.0
        assert risk[-1] == pytest.approx(0.5)

    def test_all_tied_scores_collapses_to_one_point(self):
        score = np.array([0.5, 0.5, 0.5, 0.5])
        loss = np.array([0.0, 1.0, 0.0, 1.0])
        cov, risk, thr = _rc_curve_with_ties(score, loss)
        assert cov.shape == (1,)
        assert cov[0] == 1.0
        assert risk[0] == pytest.approx(0.5)
        assert thr[0] == 0.5

    def test_partial_ties_collapse_runs(self):
        # Three groups by unique score: {0.1, 0.1, 0.1}, {0.2, 0.2}, {0.3}.
        # Risk at the end of each run is cum_mean[end_idx] = mean of
        # losses over all samples with score <= end-of-run threshold.
        score = np.array([0.1, 0.1, 0.1, 0.2, 0.2, 0.3])
        loss = np.array([0.0, 1.0, 1.0, 0.0, 0.0, 1.0])
        cov, risk, thr = _rc_curve_with_ties(score, loss)
        assert cov.shape == (3,)
        np.testing.assert_allclose(cov, [3 / 6, 5 / 6, 6 / 6], atol=1e-12)
        np.testing.assert_allclose(
            risk,
            [
                (0 + 1 + 1) / 3,           # end of {0.1, 0.1, 0.1}
                (0 + 1 + 1 + 0 + 0) / 5,   # end of {0.2, 0.2}
                (0 + 1 + 1 + 0 + 0 + 1) / 6,  # end of {0.3}
            ],
            atol=1e-12,
        )
        np.testing.assert_allclose(thr, [0.1, 0.2, 0.3], atol=1e-12)

    def test_input_order_invariant(self):
        score = np.array([0.3, 0.1, 0.4, 0.2])
        loss = np.array([1.0, 0.0, 0.0, 1.0])
        cov_a, risk_a, _ = _rc_curve_with_ties(score, loss)
        perm = np.array([2, 0, 3, 1])
        cov_b, risk_b, _ = _rc_curve_with_ties(score[perm], loss[perm])
        np.testing.assert_allclose(cov_a, cov_b, atol=1e-12)
        np.testing.assert_allclose(risk_a, risk_b, atol=1e-12)

    def test_input_order_invariant_with_ties(self):
        score = np.array([0.1, 0.1, 0.2, 0.2])
        loss = np.array([1.0, 0.0, 1.0, 0.0])
        cov_a, risk_a, _ = _rc_curve_with_ties(score, loss)
        perm = np.array([3, 0, 2, 1])
        cov_b, risk_b, _ = _rc_curve_with_ties(score[perm], loss[perm])
        np.testing.assert_allclose(cov_a, cov_b, atol=1e-12)
        np.testing.assert_allclose(risk_a, risk_b, atol=1e-12)


def _brute_force_aurc_no_ties(score, loss):
    cov, risk, _ = _brute_force_rc_curve_no_ties(score, loss)
    _trapz = getattr(np, "trapezoid", None) or np.trapz
    return float(_trapz(risk, cov))


class TestRcCurvePublic:
    def test_matches_workhorse(self):
        score = np.array([0.3, 0.1, 0.4, 0.2])
        loss = np.array([1.0, 0.0, 0.0, 1.0])
        a = rc_curve(score, loss)
        b = _rc_curve_with_ties(score, loss)
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x, y)

    def test_returns_tuple_of_three(self):
        out = rc_curve(np.array([0.1, 0.2]), np.array([0.0, 1.0]))
        assert isinstance(out, tuple)
        assert len(out) == 3
        for arr in out:
            assert isinstance(arr, np.ndarray)


class TestAurc:
    def test_matches_brute_force_no_ties(self):
        for n in (10, 100, 1000):
            for seed in range(5):
                rng2 = np.random.RandomState(seed)
                score = rng2.standard_normal(n)
                loss = rng2.binomial(1, 0.3, size=n).astype(float)
                got = aurc(score, loss)
                want = _brute_force_aurc_no_ties(score, loss)
                assert got == pytest.approx(want, abs=1e-12)

    def test_perfect_ranker_minimises(self):
        rng = np.random.RandomState(3)
        loss = rng.binomial(1, 0.4, size=200).astype(float)
        random_score = rng.standard_normal(200)
        assert aurc(loss, loss) <= aurc(random_score, loss) + 1e-9

    def test_worst_ranker_maximises(self):
        rng = np.random.RandomState(7)
        loss = rng.binomial(1, 0.4, size=200).astype(float)
        random_score = rng.standard_normal(200)
        assert aurc(-loss, loss) >= aurc(random_score, loss) - 0.05

    def test_random_ranker_near_mean_loss(self):
        rng = np.random.RandomState(5)
        loss = rng.binomial(1, 0.3, size=10000).astype(float)
        diffs = []
        for seed in range(20):
            rng2 = np.random.RandomState(100 + seed)
            score = rng2.standard_normal(10000)
            diffs.append(abs(aurc(score, loss) - loss.mean()))
        assert np.mean(diffs) < 0.02

    def test_rank_only_dependence(self):
        rng = np.random.RandomState(6)
        score = rng.standard_normal(50)
        loss = rng.binomial(1, 0.5, size=50).astype(float)
        ranks = rankdata(score, method="average") / 50
        assert aurc(score, loss) == pytest.approx(aurc(ranks, loss), abs=1e-12)

    def test_all_correct_aurc_zero(self):
        score = np.array([0.1, 0.2, 0.3])
        loss = np.zeros(3)
        assert aurc(score, loss) == 0.0

    def test_all_wrong_aurc_one(self):
        score = np.array([0.1, 0.2, 0.3])
        loss = np.ones(3)
        assert aurc(score, loss) == pytest.approx(2.0 / 3.0, abs=1e-12)

    def test_validation_propagates(self):
        with pytest.raises(ValueError, match="non-negative"):
            aurc(np.array([0.1, 0.2]), np.array([0.0, -0.1]))

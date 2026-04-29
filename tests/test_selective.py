"""Tests for osr_metrics.selective."""
from __future__ import annotations

import numpy as np
import pytest

from osr_metrics.selective import _validate_score_loss, _validate_coverage


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
        with pytest.raises(ValueError, match="ood_score contains NaN"):
            _validate_score_loss(np.array([0.1, np.inf]), np.array([0.0, 1.0]))

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

"""Tests for calibration metrics (ECE, Brier)."""
import numpy as np
import pytest

from osr_metrics import brier_score, expected_calibration_error


def test_ece_overconfident_wrong():
    """All-0.9 predictions on all-zero labels → ECE = |0.9 - 0.0| = 0.9."""
    probs = np.full(100, 0.9)
    labels = np.zeros(100)
    assert expected_calibration_error(probs, labels) == pytest.approx(0.9, abs=1e-9)


def test_ece_perfect_calibration():
    """Probabilities matching empirical positive rate → ECE near 0."""
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 5000)
    y = (rng.uniform(0, 1, 5000) < p).astype(int)
    assert expected_calibration_error(p, y, n_bins=15) < 0.05


def test_ece_multilabel_flatten():
    """Multi-label flattens to (sample, label) pairs; result equals 1D version."""
    rng = np.random.default_rng(1)
    probs = rng.uniform(0, 1, (200, 14))
    labels = (rng.uniform(0, 1, (200, 14)) < probs).astype(int)
    ece_2d = expected_calibration_error(probs, labels)
    ece_flat = expected_calibration_error(probs.ravel(), labels.ravel())
    assert ece_2d == pytest.approx(ece_flat)


def test_ece_shape_mismatch_raises():
    with pytest.raises(ValueError):
        expected_calibration_error(np.zeros(10), np.zeros(11))


def test_brier_identity():
    """Brier = mean((p - y)^2)."""
    p = np.array([0.7, 0.3, 0.9])
    y = np.array([1.0, 0.0, 1.0])
    expected = float(np.mean((p - y) ** 2))
    assert brier_score(p, y) == pytest.approx(expected)


def test_brier_perfect_zero():
    p = np.array([1.0, 0.0, 1.0])
    y = np.array([1.0, 0.0, 1.0])
    assert brier_score(p, y) == pytest.approx(0.0)


def test_brier_worst_one():
    p = np.array([0.0, 1.0])
    y = np.array([1.0, 0.0])
    assert brier_score(p, y) == pytest.approx(1.0)


def test_empty_inputs_return_nan():
    assert np.isnan(expected_calibration_error(np.array([]), np.array([])))
    assert np.isnan(brier_score(np.array([]), np.array([])))

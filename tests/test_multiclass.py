"""Tests for multi-class (single-label) extensions: T5–T7."""
from __future__ import annotations

import numpy as np
import pytest

from osr_metrics import (
    balanced_accuracy,
    brier_score_multiclass,
    compute_aoscr,
    compute_aoscr_multiclass,
    expected_calibration_error_multiclass,
    macro_f1_multiclass,
    top1_accuracy,
)


# ---------------------------- T7: closed-set ---------------------------- #


def test_top1_accuracy_perfect():
    y = np.array([0, 1, 2, 1, 0])
    preds = y.copy()
    assert top1_accuracy(preds, y) == 1.0


def test_top1_accuracy_from_logits():
    logits = np.array([[2.0, 0.1, 0.0], [0.0, 1.0, 0.5], [0.0, 0.0, 3.0]])
    y = np.array([0, 1, 2])
    assert top1_accuracy(logits, y) == 1.0


def test_top1_accuracy_zero():
    y = np.array([0, 1, 2])
    preds = np.array([1, 2, 0])
    assert top1_accuracy(preds, y) == 0.0


def test_macro_f1_multiclass_perfect():
    y = np.array([0, 1, 2, 0, 1, 2])
    preds = y.copy()
    assert macro_f1_multiclass(preds, y) == pytest.approx(1.0)


def test_macro_f1_handles_logits():
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(50, 4))
    y = logits.argmax(axis=1)
    assert macro_f1_multiclass(logits, y) == pytest.approx(1.0)


def test_balanced_accuracy_imbalanced():
    # 90 of class 0, 10 of class 1; predict all 0 → BA = 0.5 (not 0.9)
    y = np.array([0] * 90 + [1] * 10)
    preds = np.zeros_like(y)
    assert balanced_accuracy(preds, y) == pytest.approx(0.5)
    assert top1_accuracy(preds, y) == pytest.approx(0.9)


def test_invalid_preds_shape_raises():
    with pytest.raises(ValueError, match="1-D"):
        top1_accuracy(np.zeros((2, 3, 4)), np.array([0, 1]))


# ---------------------------- T6: AOSCR wrapper ------------------------- #


def test_compute_aoscr_multiclass_matches_compute_aoscr():
    rng = np.random.default_rng(42)
    n = 200
    scores = rng.uniform(size=n)
    ood_labels = (rng.uniform(size=n) > 0.7).astype(int)
    y = rng.integers(0, 5, size=n)
    preds_int = rng.integers(0, 5, size=n)

    direct = compute_aoscr(scores, ood_labels, preds_int, y)
    via_wrapper = compute_aoscr_multiclass(scores, ood_labels, preds_int, y)
    assert direct == pytest.approx(via_wrapper)


def test_compute_aoscr_multiclass_logits_argmax():
    rng = np.random.default_rng(7)
    n, K = 150, 4
    scores = rng.uniform(size=n)
    ood_labels = (rng.uniform(size=n) > 0.5).astype(int)
    y = rng.integers(0, K, size=n)
    logits = rng.normal(size=(n, K))

    expected = compute_aoscr(scores, ood_labels, logits.argmax(axis=1), y)
    got = compute_aoscr_multiclass(scores, ood_labels, logits, y)
    assert expected == pytest.approx(got)


def test_compute_aoscr_multiclass_invalid_shape():
    with pytest.raises(ValueError, match="1-D"):
        compute_aoscr_multiclass(
            np.zeros(3), np.zeros(3), np.zeros((3, 4, 5)), np.zeros(3)
        )


# ---------------------------- T5: calibration overloads ----------------- #


def test_ece_multiclass_perfect_calibration():
    # Hard one-hot predictions matching y → confidence=1.0, accuracy=1.0
    n, K = 100, 3
    y = np.array([i % K for i in range(n)])
    probs = np.zeros((n, K))
    probs[np.arange(n), y] = 1.0
    assert expected_calibration_error_multiclass(probs, y) == pytest.approx(0.0)


def test_ece_multiclass_overconfident_wrong():
    # Always predict class 0 with confidence 1.0; ground truth always class 1
    # → confidence = 1.0, accuracy = 0.0, gap = 1.0 over the whole population
    n = 50
    probs = np.zeros((n, 3))
    probs[:, 0] = 1.0
    y = np.ones(n, dtype=int)
    assert expected_calibration_error_multiclass(probs, y) == pytest.approx(1.0)


def test_ece_multiclass_invalid_shape_1d():
    with pytest.raises(ValueError, match="2-D"):
        expected_calibration_error_multiclass(np.array([0.5, 0.6]), np.array([0, 1]))


def test_brier_multiclass_perfect():
    n, K = 20, 4
    y = np.array([i % K for i in range(n)])
    probs = np.zeros((n, K))
    probs[np.arange(n), y] = 1.0
    assert brier_score_multiclass(probs, y) == pytest.approx(0.0)


def test_brier_multiclass_uniform():
    # Uniform 1/K predictions, y any class → per-sample
    # SSE = (1 - 1/K)^2 + (K-1) * (1/K)^2 = 1 - 1/K
    n, K = 30, 4
    probs = np.full((n, K), 1.0 / K)
    y = np.zeros(n, dtype=int)
    expected = 1.0 - 1.0 / K
    assert brier_score_multiclass(probs, y) == pytest.approx(expected)


def test_brier_multiclass_invalid_shape():
    with pytest.raises(ValueError, match="2-D"):
        brier_score_multiclass(np.array([0.3, 0.7]), np.array([0, 1]))


def test_ece_multiclass_rejects_logits():
    # Logits typically have values outside [0, 1] — most common user error.
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(20, 4))
    y = rng.integers(0, 4, size=20)
    with pytest.raises(ValueError, match="logits"):
        expected_calibration_error_multiclass(logits, y)


def test_brier_multiclass_rejects_logits():
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(20, 4))
    y = rng.integers(0, 4, size=20)
    with pytest.raises(ValueError, match="logits"):
        brier_score_multiclass(logits, y)

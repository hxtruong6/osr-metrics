# tests/test_metrics.py
import math
import numpy as np
import pytest
from osr_metrics.ood import fpr_at_95tpr, aupr_in, aupr_out, partition_ood_by_purity, bootstrap_ci
from osr_metrics.classification import macro_auprc, macro_auprc_id_labels
from osr_metrics.stability import stability_score


# ----- OOD metrics -----

def test_fpr_at_95tpr_perfect_separation():
    """Perfect OOD detector: FPR@95TPR = 0."""
    scores = np.array([-1.0] * 100 + [1.0] * 100)
    labels = np.array([0] * 100 + [1] * 100)  # 1 = OOD
    fpr = fpr_at_95tpr(scores, labels)
    assert fpr == pytest.approx(0.0, abs=0.01)


def test_fpr_at_95tpr_random_detector():
    """Random OOD detector: FPR@95TPR ≈ 0.95."""
    rng = np.random.RandomState(42)
    scores = rng.randn(1000)
    labels = rng.randint(0, 2, size=1000)
    fpr = fpr_at_95tpr(scores, labels)
    assert 0.8 < fpr < 1.0


def test_bootstrap_ci_shape():
    """bootstrap_ci returns (lower, upper) with lower <= mean <= upper."""
    rng = np.random.RandomState(0)
    scores = rng.randn(200)
    labels = (scores > 0).astype(int)
    lower, mean, upper = bootstrap_ci(
        scores, labels, metric_fn=fpr_at_95tpr, n_bootstrap=200, ci=0.95, seed=42
    )
    assert lower <= mean <= upper


# ----- Classification metrics -----

def test_macro_auprc_perfect():
    """Perfect classifier: macro-AUPRC = 1.0."""
    probs = np.array([[0.9, 0.1], [0.1, 0.9]])
    labels = np.array([[1, 0], [0, 1]])
    score = macro_auprc(probs, labels)
    assert score == pytest.approx(1.0, abs=1e-5)


def test_macro_auprc_id_labels_excludes_held_out():
    """macro_auprc_id_labels excludes held-out labels from the average.

    With 3 labels A,B,C where C is held-out and always 0 on test_id,
    including C yields 0.0 for that column and drags down macro.
    Excluding C gives the correct macro over A and B only.
    """
    # 2 samples, 3 labels: A, B, C. C is held-out (always 0).
    probs = np.array([[0.9, 0.2, 0.5], [0.1, 0.8, 0.5]])  # good on A,B; random on C
    labels = np.array([[1, 0, 0], [0, 1, 0]])  # C always 0
    label_names = ["A", "B", "C"]
    held_out = ["C"]
    score_all = macro_auprc(probs, labels)
    score_id_only = macro_auprc_id_labels(probs, labels, label_names, held_out)
    # With C included, sklearn returns 0 for the all-negative column -> lower macro
    assert score_id_only >= score_all
    assert score_id_only == pytest.approx(macro_auprc(probs[:, :2], labels[:, :2]), abs=1e-5)


# ----- Stability score -----

def test_stability_score_constant():
    """Constant probability trajectory -> Stability Score = 1.0."""
    probs = [0.7] * 10
    score = stability_score(probs)
    assert score == pytest.approx(1.0, abs=1e-5)


def test_stability_score_range():
    """Stability Score is always in [0, 1]."""
    import random
    rng = random.Random(0)
    probs = [rng.random() for _ in range(20)]
    score = stability_score(probs)
    assert 0.0 <= score <= 1.0


# ----- AUPR-In / AUPR-Out -----

def test_aupr_in_perfect():
    """Perfect detector: AUPR-In = 1.0."""
    scores = np.array([-1.0] * 100 + [1.0] * 100)
    labels = np.array([0] * 100 + [1] * 100)
    val = aupr_in(scores, labels)
    assert val == pytest.approx(1.0, abs=0.01)


def test_aupr_out_perfect():
    """Perfect detector: AUPR-Out = 1.0."""
    scores = np.array([-1.0] * 100 + [1.0] * 100)
    labels = np.array([0] * 100 + [1] * 100)
    val = aupr_out(scores, labels)
    assert val == pytest.approx(1.0, abs=0.01)


def test_aupr_in_out_bounds():
    """AUPR-In and AUPR-Out are always in [0, 1]."""
    rng = np.random.RandomState(42)
    scores = rng.randn(500)
    labels = rng.randint(0, 2, size=500)
    val_in = aupr_in(scores, labels)
    val_out = aupr_out(scores, labels)
    assert 0.0 <= val_in <= 1.0
    assert 0.0 <= val_out <= 1.0


# ----- DeLong test -----

def test_delong_identical():
    """Identical scores → DeLong p ≈ 1.0 (no significant difference)."""
    from osr_metrics.delong import delong_test
    rng = np.random.RandomState(42)
    scores = rng.randn(200)
    labels = (scores > 0).astype(int)
    z, p = delong_test(scores, scores, labels)
    assert p == pytest.approx(1.0, abs=0.01)
    assert z == pytest.approx(0.0, abs=0.01)


def test_delong_different():
    """Clearly different AUROCs → DeLong p < 0.05."""
    from osr_metrics.delong import delong_test
    rng = np.random.RandomState(42)
    n = 500
    labels = np.array([0] * (n // 2) + [1] * (n // 2))
    # Method A: perfect separation
    scores_a = np.array([-1.0] * (n // 2) + [1.0] * (n // 2)) + rng.randn(n) * 0.1
    # Method B: random
    scores_b = rng.randn(n)
    z, p = delong_test(scores_a, scores_b, labels)
    assert p < 0.05


# ----- Partition OOD by purity -----

def test_partition_pure_vs_mixed():
    """partition_ood_by_purity correctly separates pure and mixed OOD images."""
    label_names = ["A", "B", "C", "D"]
    held_out = ["C", "D"]

    # 5 images: 2 ID, 2 pure OOD (only held-out labels), 1 mixed OOD (held-out + known)
    label_vecs = np.array([
        [1, 1, 0, 0],  # ID: only known labels
        [1, 0, 0, 0],  # ID: only known labels
        [0, 0, 1, 0],  # OOD: only label C (pure OOD)
        [0, 0, 0, 1],  # OOD: only label D (pure OOD)
        [1, 0, 1, 0],  # OOD: label A + C (mixed OOD)
    ])
    ood_mask = np.array([False, False, True, True, True])

    pure, mixed = partition_ood_by_purity(label_vecs, ood_mask, label_names, held_out)

    assert pure.sum() == 2  # images 2 and 3
    assert mixed.sum() == 1  # image 4
    assert pure[2] and pure[3]
    assert mixed[4]
    # ID images should not be in either
    assert not pure[0] and not pure[1]
    assert not mixed[0] and not mixed[1]


# ----- Stability score -----

def test_stability_score_formula():
    """Manual verification against formula: Stab = 1 - MAD."""
    probs = [0.3, 0.7, 0.5]
    mean_p = sum(probs) / len(probs)
    mad = sum(abs(p - mean_p) for p in probs) / len(probs)
    expected = 1.0 - mad
    assert stability_score(probs) == pytest.approx(expected, abs=1e-6)

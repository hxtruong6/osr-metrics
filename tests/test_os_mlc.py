"""Tests for the OS-MLC additions: AML-OSCR, RC-macro-F1,
per-novel-label discovery, and paired bootstrap diff.

Each metric is verified against a brute-force reference (loops, no
vectorisation) and against known invariants (perfect rejector ⇒ area
collapses to closed-set macro-F1; identical methods ⇒ paired bootstrap
diff has CI containing zero).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score

from osr_metrics import (
    aml_oscr_curve,
    auroc,
    compute_aml_oscr,
    compute_rc_macro_f1,
    paired_bootstrap_diff,
    per_novel_discovery_table,
)


def _brute_aml_oscr(novelty, ood, probs, labels, thresholds):
    """Reference implementation: loop over unique τ, compute macro-F1
    on accepted ID set with sklearn at every step."""
    novelty = np.asarray(novelty, dtype=float)
    ood = np.asarray(ood).astype(int)
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels).astype(int)
    thr = np.asarray(thresholds, dtype=float)

    id_mask = ood == 0
    n_ood = int((ood == 1).sum())
    if n_ood == 0 or id_mask.sum() == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 0.0]), 0.0

    preds = (probs >= thr[None, :]).astype(int)
    taus = sorted(set(novelty.tolist()))

    fprs = [0.0]
    f1s = [0.0]
    for tau in taus:
        accepted = novelty <= tau
        accepted_id = accepted & id_mask
        accepted_ood = accepted & ~id_mask
        fpr = accepted_ood.sum() / n_ood
        if accepted_id.sum() == 0:
            f1 = 0.0
        else:
            f1 = f1_score(
                labels[accepted_id],
                preds[accepted_id],
                average="macro",
                zero_division=0,
            )
        fprs.append(float(fpr))
        f1s.append(float(f1))

    fprs_a = np.array(fprs)
    f1s_a = np.array(f1s)
    _trapz = getattr(np, "trapezoid", None) or np.trapz
    return fprs_a, f1s_a, float(_trapz(f1s_a, fprs_a))


def test_aml_oscr_matches_brute_force():
    rng = np.random.default_rng(0)
    n, K = 80, 4
    novelty = rng.normal(size=n)
    ood = rng.integers(0, 2, size=n)
    probs = rng.uniform(size=(n, K))
    labels = rng.integers(0, 2, size=(n, K))
    thresholds = np.full(K, 0.5)

    fpr, f1, area = aml_oscr_curve(novelty, ood, probs, labels, thresholds)
    fpr_b, f1_b, area_b = _brute_aml_oscr(novelty, ood, probs, labels, thresholds)

    assert np.allclose(fpr, fpr_b, atol=1e-12)
    assert np.allclose(f1, f1_b, atol=1e-12)
    assert area == np.float64(area_b)  # implementations should agree exactly


def test_aml_oscr_perfect_rejector_recovers_closed_set_macro_f1():
    """If novelty score perfectly separates ID from OOD (low for ID,
    high for OOD), the curve hits (0, closed_set_macro_f1) and area
    equals closed_set_macro_f1 (because the curve is a step that jumps
    to its plateau at FPR=0 then stays flat across FPR ∈ [0, 1])."""
    rng = np.random.default_rng(1)
    n_id, n_ood, K = 50, 50, 3
    probs_id = rng.uniform(size=(n_id, K))
    labels_id = (probs_id > 0.5).astype(int)  # noisy but correlated → moderate F1
    probs_ood = rng.uniform(size=(n_ood, K))
    labels_ood = rng.integers(0, 2, size=(n_ood, K))

    probs = np.vstack([probs_id, probs_ood])
    labels = np.vstack([labels_id, labels_ood])
    novelty = np.concatenate([np.zeros(n_id), np.ones(n_ood)])  # perfect separation
    ood = np.concatenate([np.zeros(n_id, dtype=int), np.ones(n_ood, dtype=int)])
    thresholds = np.full(K, 0.5)

    fpr, f1, area = aml_oscr_curve(novelty, ood, probs, labels, thresholds)
    closed_set_macro_f1 = f1_score(
        labels_id,
        (probs_id >= 0.5).astype(int),
        average="macro",
        zero_division=0,
    )

    # at FPR=0 the curve has reached its plateau
    assert f1[1] == np.float64(closed_set_macro_f1)
    # area = plateau × width = closed_set_macro_f1 × 1.0
    assert abs(area - closed_set_macro_f1) < 1e-12


def test_aml_oscr_empty_class_returns_zero():
    novelty = np.array([0.1, 0.2, 0.3])
    ood = np.array([0, 0, 0])  # no OOD
    probs = np.array([[0.6, 0.2], [0.3, 0.7], [0.5, 0.5]])
    labels = np.array([[1, 0], [0, 1], [1, 1]])
    thresholds = np.array([0.5, 0.5])
    fpr, f1, area = aml_oscr_curve(novelty, ood, probs, labels, thresholds)
    assert area == 0.0
    assert np.allclose(fpr, [0.0, 1.0])
    assert np.allclose(f1, [0.0, 0.0])


def test_compute_aml_oscr_fpr_max_clips_correctly():
    rng = np.random.default_rng(2)
    n, K = 60, 3
    novelty = rng.normal(size=n)
    ood = rng.integers(0, 2, size=n)
    probs = rng.uniform(size=(n, K))
    labels = rng.integers(0, 2, size=(n, K))
    thresholds = np.full(K, 0.5)

    full = compute_aml_oscr(novelty, ood, probs, labels, thresholds)
    capped = compute_aml_oscr(novelty, ood, probs, labels, thresholds, fpr_max=0.2)
    full_at_1 = compute_aml_oscr(novelty, ood, probs, labels, thresholds, fpr_max=1.0)

    assert capped <= full + 1e-12
    assert abs(full_at_1 - full) < 1e-12


def test_per_novel_discovery_basic():
    # 4 novel labels {A, B, C, D}, 2 known labels {x, y}
    # 6 images:
    #   i0: novel = A,         known = x        → mixed, A alone
    #   i1: novel = A,         known = (none)   → pure,  A alone
    #   i2: novel = A, B,      known = x        → mixed, A+B  (one_co for both)
    #   i3: novel = A, B, C,   known = (none)   → pure,  three-way (two_plus_co for A)
    #   i4: novel = (none),    known = x, y     → not novel — should be ignored
    #   i5: novel = D,         known = (none)   → pure,  D alone
    novel = np.array(
        [
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 1, 0, 0],
            [1, 1, 1, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 1],
        ],
        dtype=int,
    )
    known = np.array(
        [
            [1, 0],
            [0, 0],
            [1, 0],
            [0, 0],
            [1, 1],
            [0, 0],
        ],
        dtype=int,
    )
    novelty = np.array([0.9, 0.1, 0.8, 0.95, 0.05, 0.2])
    table = per_novel_discovery_table(
        novelty, novel, known, threshold=0.5,
        novel_label_names=["A", "B", "C", "D"],
    )

    # A appears in i0, i1, i2, i3
    assert table["A"]["all"]["n"] == 4
    # flagged (>0.5): i0=0.9, i2=0.8, i3=0.95 → 3 of 4 = 0.75
    assert abs(table["A"]["all"]["discovery"] - 0.75) < 1e-12
    # alone (no co-novel): i0, i1 → 2 images, flagged i0 only → 0.5
    assert table["A"]["alone"]["n"] == 2
    assert abs(table["A"]["alone"]["discovery"] - 0.5) < 1e-12
    # one_co: i2 → 1 image, flagged → 1.0
    assert table["A"]["one_co"] == {"discovery": 1.0, "n": 1}
    # two_plus_co: i3 → 1 image, flagged → 1.0
    assert table["A"]["two_plus_co"] == {"discovery": 1.0, "n": 1}
    # pure regime: i1, i3 → 2 images, flagged i3 → 0.5
    assert table["A"]["pure"]["n"] == 2
    assert abs(table["A"]["pure"]["discovery"] - 0.5) < 1e-12
    # mixed: i0, i2 → 2 images, both flagged → 1.0
    assert table["A"]["mixed"] == {"discovery": 1.0, "n": 2}

    # D appears only once (i5, alone, pure, score 0.2 < 0.5)
    assert table["D"]["all"] == {"discovery": 0.0, "n": 1}
    assert table["D"]["alone"] == {"discovery": 0.0, "n": 1}
    # D never appears in mixed → nan
    assert table["D"]["mixed"]["n"] == 0
    assert np.isnan(table["D"]["mixed"]["discovery"])


def test_compute_rc_macro_f1_basic_and_contagion_delta():
    rng = np.random.default_rng(3)
    n, K = 40, 3
    probs = rng.uniform(size=(n, K))
    labels = rng.integers(0, 2, size=(n, K))
    thresholds = np.full(K, 0.5)
    mixed_mask = np.zeros(n, dtype=bool)
    mixed_mask[:20] = True
    id_only_mask = np.zeros(n, dtype=bool)
    id_only_mask[20:] = True

    out = compute_rc_macro_f1(probs, labels, thresholds, mixed_mask, id_only_mask)

    # cross-check against sklearn directly
    preds_mixed = (probs[:20] >= 0.5).astype(int)
    expected_mixed = f1_score(labels[:20], preds_mixed, average="macro", zero_division=0)
    preds_id = (probs[20:] >= 0.5).astype(int)
    expected_id = f1_score(labels[20:], preds_id, average="macro", zero_division=0)
    assert abs(out["macro_f1_mixed"] - expected_mixed) < 1e-12
    assert abs(out["macro_f1_id_only"] - expected_id) < 1e-12
    assert abs(out["contagion_delta"] - (expected_id - expected_mixed)) < 1e-12
    assert out["n_mixed"] == 20 and out["n_id_only"] == 20


def test_compute_rc_macro_f1_empty_mixed_returns_nan():
    n, K = 10, 2
    probs = np.zeros((n, K))
    labels = np.zeros((n, K), dtype=int)
    thresholds = np.full(K, 0.5)
    out = compute_rc_macro_f1(probs, labels, thresholds, mixed_mask=np.zeros(n, dtype=bool))
    assert np.isnan(out["macro_f1_mixed"])
    assert out["n_mixed"] == 0
    assert "contagion_delta" not in out


def test_paired_bootstrap_diff_identical_methods_ci_contains_zero():
    rng = np.random.default_rng(4)
    n = 100
    scores = rng.normal(size=n)
    labels = rng.integers(0, 2, size=n)
    delta, lo, hi, p = paired_bootstrap_diff(
        auroc, scores, scores, labels, n_bootstrap=500, seed=0
    )
    # identical inputs ⇒ every replicate's diff is exactly 0
    assert delta == 0.0
    assert lo == 0.0 and hi == 0.0
    # all replicates satisfy diff >= 0 AND diff <= 0 ⇒ p = 2 * min(1, 1) = 2 → clamped to 1
    assert p == 1.0


def test_paired_bootstrap_diff_strictly_better_method_low_p():
    """Method A has scores perfectly aligned with labels; B is random.
    The CI for AUROC delta should be strictly positive and p small."""
    rng = np.random.default_rng(5)
    n = 200
    labels = rng.integers(0, 2, size=n)
    scores_a = labels.astype(float) + 0.01 * rng.normal(size=n)  # near-perfect
    scores_b = rng.normal(size=n)  # random
    delta, lo, hi, p = paired_bootstrap_diff(
        auroc, scores_a, scores_b, labels, n_bootstrap=500, seed=0, stratify=True
    )
    assert delta > 0.3
    assert lo > 0.0  # CI excludes zero
    assert p < 0.01

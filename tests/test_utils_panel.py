"""Tests for utils (T8), panel (T13), and validation (T14)."""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from osr_metrics import (
    as_ood_scores,
    auroc,
    compute_panel,
    fpr_at_95tpr,
    warn_if_inverted_scores,
)


# ----------------------------- T8: as_ood_scores ------------------------- #


def test_as_ood_scores_passthrough():
    s = np.array([0.1, 0.5, 0.9])
    np.testing.assert_array_equal(as_ood_scores(s, "ood"), s)


def test_as_ood_scores_flips_confidence():
    s = np.array([0.95, 0.10, 0.80])
    np.testing.assert_array_equal(as_ood_scores(s, "confidence"), -s)
    np.testing.assert_array_equal(as_ood_scores(s, "id"), -s)


def test_as_ood_scores_invalid_direction():
    with pytest.raises(ValueError, match="direction"):
        as_ood_scores(np.array([0.0]), direction="weird")  # type: ignore[arg-type]


def test_confidence_flip_recovers_auroc():
    # Confidence: ID gets high values, OOD gets low values → AUROC on the
    # raw confidence is poor; flipping recovers a perfect detector.
    confidence = np.array([0.95, 0.99, 0.01, 0.05])
    ood_labels = np.array([0, 0, 1, 1])
    flipped = as_ood_scores(confidence, direction="confidence")
    assert auroc(flipped, ood_labels) == pytest.approx(1.0)


# ----------------------- T8: warn_if_inverted_scores --------------------- #


def test_warn_if_inverted_scores_triggers():
    confidence = np.array([0.95, 0.99, 0.01, 0.05])
    ood_labels = np.array([0, 0, 1, 1])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warn_if_inverted_scores(confidence, ood_labels)
    assert any("inverted" in str(w.message).lower() for w in caught)


def test_warn_if_inverted_scores_silent_when_correct():
    ood_scores = np.array([0.05, 0.01, 0.99, 0.95])
    ood_labels = np.array([0, 0, 1, 1])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warn_if_inverted_scores(ood_scores, ood_labels)
    assert not any(
        "inverted" in str(w.message).lower() for w in caught
    )


def test_warn_if_inverted_scores_skips_single_class():
    # Cannot compute AUROC; should not raise.
    warn_if_inverted_scores(np.array([0.1, 0.2]), np.array([0, 0]))


# -------------------------- T14: input validation ----------------------- #


def test_auroc_rejects_2d_labels():
    scores = np.array([0.1, 0.5])
    with pytest.raises(ValueError, match="1-D"):
        auroc(scores, np.array([[0, 1], [1, 0]]))


def test_auroc_rejects_non_binary_labels():
    scores = np.array([0.1, 0.5, 0.9])
    with pytest.raises(ValueError, match="binary"):
        auroc(scores, np.array([0, 1, 2]))


def test_auroc_rejects_length_mismatch():
    with pytest.raises(ValueError, match="length mismatch"):
        auroc(np.array([0.1, 0.5]), np.array([0, 1, 0]))


def test_fpr_at_tpr_rejects_invalid_target():
    from osr_metrics import fpr_at_tpr

    scores = np.array([0.1, 0.5])
    labels = np.array([0, 1])
    with pytest.raises(ValueError, match="target_tpr"):
        fpr_at_tpr(scores, labels, target_tpr=0.0)
    with pytest.raises(ValueError, match="target_tpr"):
        fpr_at_tpr(scores, labels, target_tpr=1.5)


# ----------------------------- T13: panel ------------------------------- #


def test_panel_multiclass_minimal():
    rng = np.random.default_rng(0)
    n, K = 80, 4
    scores = rng.uniform(size=n)
    ood_labels = (rng.uniform(size=n) > 0.6).astype(int)
    y = rng.integers(0, K, size=n)
    logits = rng.normal(size=(n, K))
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    softmax = e / e.sum(axis=1, keepdims=True)

    out = compute_panel(scores, ood_labels, probs=softmax, y=y)
    assert out["setting"] == "multiclass"
    assert "auroc" in out and 0.0 <= out["auroc"] <= 1.0
    assert "aoscr" in out
    assert "top1_accuracy" in out
    assert "ece" in out
    assert "brier" in out
    # Multi-label-only keys must be absent
    assert "fourclass" not in out
    assert "macro_auprc" not in out


def test_panel_multilabel_with_fourclass():
    rng = np.random.default_rng(1)
    n, K = 60, 5
    scores = rng.uniform(size=n)
    label_vecs = (rng.uniform(size=(n, K)) > 0.7).astype(int)
    probs = rng.uniform(size=(n, K))
    preds = (probs > 0.5).astype(int)
    label_names = [f"L{i}" for i in range(K)]
    held_out = ["L3", "L4"]
    # Build OOD labels from held-out presence
    held_idx = [label_names.index(h) for h in held_out]
    ood_labels = (label_vecs[:, held_idx].sum(axis=1) > 0).astype(int)

    out = compute_panel(
        scores,
        ood_labels,
        preds=preds,
        probs=probs,
        label_vecs=label_vecs,
        label_names=label_names,
        held_out_labels=held_out,
        setting="multilabel",
    )
    assert out["setting"] == "multilabel"
    assert "auroc" in out
    assert "macro_auprc" in out
    assert "ece" in out
    assert "fourclass" in out
    assert {"id_disease", "no_finding", "pure_ood", "mixed_ood"} <= set(
        out["fourclass"]["counts"].keys()
    )
    # Multi-class-only keys absent
    assert "top1_accuracy" not in out


def test_panel_no_inputs_returns_setting_only():
    out = compute_panel()
    assert out == {"setting": "multiclass"}


def test_panel_scores_only_no_labels():
    # Without ood_labels, the OOD/OSR block must be skipped.
    out = compute_panel(scores=np.array([0.1, 0.2, 0.3]))
    assert "auroc" not in out
    assert "aoscr" not in out

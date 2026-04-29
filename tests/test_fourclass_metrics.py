# tests/test_fourclass_metrics.py
"""Tests for four-class OOD partitioning and metrics (v17).

Four image types in a realistic deployment:
  1. ID disease  — only known (non-held-out) labels positive
  2. No Finding  — all-zero label vector (healthy patient)
  3. Pure OOD    — only held-out labels positive
  4. Mixed OOD   — both known AND held-out labels positive
"""
import numpy as np
import pytest

from osr_metrics.fourclass import build_fourclass_masks, compute_fourclass_metrics


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

LABEL_NAMES = ["Atelectasis", "Cardiomegaly", "Effusion", "Hernia", "Pneumonia"]
HELD_OUT = ["Hernia", "Pneumonia"]  # last two columns


def _make_dataset():
    """Construct a small 8-image dataset with all four types.

    Columns: Atelectasis(0), Cardiomegaly(1), Effusion(2), Hernia(3), Pneumonia(4)
    Held out: Hernia(3), Pneumonia(4)
    """
    label_vecs = np.array([
        [1, 0, 0, 0, 0],  # 0: ID disease (Atelectasis only)
        [0, 1, 1, 0, 0],  # 1: ID disease (Cardiomegaly + Effusion)
        [0, 0, 0, 0, 0],  # 2: No Finding
        [0, 0, 0, 0, 0],  # 3: No Finding
        [0, 0, 0, 1, 0],  # 4: Pure OOD (Hernia only)
        [0, 0, 0, 0, 1],  # 5: Pure OOD (Pneumonia only)
        [1, 0, 0, 1, 0],  # 6: Mixed OOD (Atelectasis + Hernia)
        [0, 1, 0, 0, 1],  # 7: Mixed OOD (Cardiomegaly + Pneumonia)
    ])
    return label_vecs


# ---------------------------------------------------------------------------
# build_fourclass_masks tests
# ---------------------------------------------------------------------------

class TestBuildFourclassMasks:

    def test_correct_counts(self):
        """Each type has the expected number of images."""
        masks = build_fourclass_masks(_make_dataset(), LABEL_NAMES, HELD_OUT)
        assert masks["id_disease"].sum() == 2
        assert masks["no_finding"].sum() == 2
        assert masks["pure_ood"].sum() == 2
        assert masks["mixed_ood"].sum() == 2

    def test_mutually_exclusive(self):
        """Each image belongs to exactly one type."""
        masks = build_fourclass_masks(_make_dataset(), LABEL_NAMES, HELD_OUT)
        total = (
            masks["id_disease"].astype(int)
            + masks["no_finding"].astype(int)
            + masks["pure_ood"].astype(int)
            + masks["mixed_ood"].astype(int)
        )
        np.testing.assert_array_equal(total, np.ones(8, dtype=int))

    def test_covers_all_samples(self):
        """Union of all masks covers every sample."""
        masks = build_fourclass_masks(_make_dataset(), LABEL_NAMES, HELD_OUT)
        union = (
            masks["id_disease"]
            | masks["no_finding"]
            | masks["pure_ood"]
            | masks["mixed_ood"]
        )
        assert union.all()

    def test_no_finding_is_all_zero(self):
        """No Finding images have all-zero label vectors."""
        label_vecs = _make_dataset()
        masks = build_fourclass_masks(label_vecs, LABEL_NAMES, HELD_OUT)
        nf_vecs = label_vecs[masks["no_finding"]]
        assert nf_vecs.sum() == 0

    def test_id_disease_has_no_held_out_labels(self):
        """ID disease images have zero in all held-out columns."""
        label_vecs = _make_dataset()
        masks = build_fourclass_masks(label_vecs, LABEL_NAMES, HELD_OUT)
        held_out_idx = [LABEL_NAMES.index(h) for h in HELD_OUT]
        id_vecs = label_vecs[masks["id_disease"]]
        assert id_vecs[:, held_out_idx].sum() == 0
        # But they must have at least one known label positive
        known_idx = [i for i in range(len(LABEL_NAMES)) if i not in held_out_idx]
        assert (id_vecs[:, known_idx].sum(axis=1) > 0).all()

    def test_pure_ood_has_only_held_out_labels(self):
        """Pure OOD images have positive labels only in held-out columns."""
        label_vecs = _make_dataset()
        masks = build_fourclass_masks(label_vecs, LABEL_NAMES, HELD_OUT)
        held_out_idx = [LABEL_NAMES.index(h) for h in HELD_OUT]
        known_idx = [i for i in range(len(LABEL_NAMES)) if i not in held_out_idx]
        pure_vecs = label_vecs[masks["pure_ood"]]
        # No known labels positive
        assert pure_vecs[:, known_idx].sum() == 0
        # At least one held-out label positive
        assert (pure_vecs[:, held_out_idx].sum(axis=1) > 0).all()

    def test_mixed_ood_has_both(self):
        """Mixed OOD images have at least one known AND one held-out label."""
        label_vecs = _make_dataset()
        masks = build_fourclass_masks(label_vecs, LABEL_NAMES, HELD_OUT)
        held_out_idx = [LABEL_NAMES.index(h) for h in HELD_OUT]
        known_idx = [i for i in range(len(LABEL_NAMES)) if i not in held_out_idx]
        mixed_vecs = label_vecs[masks["mixed_ood"]]
        assert (mixed_vecs[:, known_idx].sum(axis=1) > 0).all()
        assert (mixed_vecs[:, held_out_idx].sum(axis=1) > 0).all()

    def test_specific_indices(self):
        """Verify specific image indices match expected types."""
        masks = build_fourclass_masks(_make_dataset(), LABEL_NAMES, HELD_OUT)
        # Images 0,1 = ID disease
        assert masks["id_disease"][0] and masks["id_disease"][1]
        # Images 2,3 = No Finding
        assert masks["no_finding"][2] and masks["no_finding"][3]
        # Images 4,5 = Pure OOD
        assert masks["pure_ood"][4] and masks["pure_ood"][5]
        # Images 6,7 = Mixed OOD
        assert masks["mixed_ood"][6] and masks["mixed_ood"][7]

    def test_no_held_out_labels_means_no_ood(self):
        """If held_out_labels is empty, no images are OOD (pure or mixed)."""
        label_vecs = _make_dataset()
        masks = build_fourclass_masks(label_vecs, LABEL_NAMES, held_out_labels=[])
        assert masks["pure_ood"].sum() == 0
        assert masks["mixed_ood"].sum() == 0
        # Everything is either ID disease or No Finding
        assert masks["id_disease"].sum() + masks["no_finding"].sum() == len(label_vecs)

    def test_all_zero_dataset(self):
        """A dataset of only No Finding images."""
        label_vecs = np.zeros((5, 5), dtype=int)
        masks = build_fourclass_masks(label_vecs, LABEL_NAMES, HELD_OUT)
        assert masks["no_finding"].sum() == 5
        assert masks["id_disease"].sum() == 0
        assert masks["pure_ood"].sum() == 0
        assert masks["mixed_ood"].sum() == 0


# ---------------------------------------------------------------------------
# compute_fourclass_metrics tests
# ---------------------------------------------------------------------------

class TestComputeFourclassMetrics:

    def test_all_expected_keys_returned(self):
        """Return dict contains all required metric keys."""
        label_vecs = _make_dataset()
        ood_scores = np.array([0.1, 0.2, 0.15, 0.18, 0.9, 0.85, 0.7, 0.75])
        result = compute_fourclass_metrics(
            ood_scores, label_vecs, LABEL_NAMES, HELD_OUT
        )
        expected_keys = {
            "auroc_full",
            "fpr95_full",
            "auroc_pure",
            "fpr95_pure",
            "auroc_mixed",
            "fpr95_mixed",
            "auroc_mixed_vs_id_disease",
            "auroc_nf_vs_pure",
            "auroc_disease_only",
            "fpr95_disease_only",
            "counts",
        }
        assert expected_keys == set(result.keys())

    def test_counts_dict(self):
        """Counts dict has correct values."""
        label_vecs = _make_dataset()
        ood_scores = np.zeros(8)
        result = compute_fourclass_metrics(
            ood_scores, label_vecs, LABEL_NAMES, HELD_OUT
        )
        counts = result["counts"]
        assert counts["id_disease"] == 2
        assert counts["no_finding"] == 2
        assert counts["pure_ood"] == 2
        assert counts["mixed_ood"] == 2

    def test_perfect_separation_auroc_1(self):
        """When OOD scores perfectly separate ID from OOD, AUROC = 1.0."""
        label_vecs = _make_dataset()
        # ID disease + NF get low scores, OOD get high scores
        ood_scores = np.array([
            -1.0, -1.0,   # ID disease
            -1.0, -1.0,   # No Finding
            1.0, 1.0,     # Pure OOD
            1.0, 1.0,     # Mixed OOD
        ])
        result = compute_fourclass_metrics(
            ood_scores, label_vecs, LABEL_NAMES, HELD_OUT
        )
        assert result["auroc_full"] == pytest.approx(1.0, abs=1e-5)
        assert result["auroc_pure"] == pytest.approx(1.0, abs=1e-5)
        assert result["auroc_mixed"] == pytest.approx(1.0, abs=1e-5)
        assert result["auroc_nf_vs_pure"] == pytest.approx(1.0, abs=1e-5)
        assert result["fpr95_full"] == pytest.approx(0.0, abs=0.01)

    def test_nf_pure_ood_confusion_auroc_near_half(self):
        """When NF and Pure OOD have identical score distributions, AUROC ~ 0.5."""
        rng = np.random.RandomState(42)
        n_per_type = 200
        # Build a larger dataset: n_per_type of each type
        id_vecs = np.zeros((n_per_type, 5), dtype=int)
        id_vecs[:, 0] = 1  # Atelectasis
        nf_vecs = np.zeros((n_per_type, 5), dtype=int)
        pure_vecs = np.zeros((n_per_type, 5), dtype=int)
        pure_vecs[:, 3] = 1  # Hernia
        mixed_vecs = np.zeros((n_per_type, 5), dtype=int)
        mixed_vecs[:, 0] = 1  # Atelectasis
        mixed_vecs[:, 3] = 1  # Hernia

        label_vecs = np.vstack([id_vecs, nf_vecs, pure_vecs, mixed_vecs])

        # NF and Pure OOD get scores from the same distribution (indistinguishable)
        nf_scores = rng.randn(n_per_type)
        pure_scores = rng.randn(n_per_type)
        # ID and Mixed get clearly distinct scores
        id_scores = rng.randn(n_per_type) - 5.0   # very low
        mixed_scores = rng.randn(n_per_type) + 5.0  # very high

        ood_scores = np.concatenate([id_scores, nf_scores, pure_scores, mixed_scores])

        result = compute_fourclass_metrics(
            ood_scores, label_vecs, LABEL_NAMES, HELD_OUT
        )
        # NF vs Pure OOD should be near chance
        assert 0.4 < result["auroc_nf_vs_pure"] < 0.6

    def test_disease_only_backward_compat(self):
        """auroc_disease_only matches v15 behavior: ID-disease vs all OOD (no NF)."""
        label_vecs = _make_dataset()
        # Perfect: ID disease gets -1, OOD gets +1, NF gets 0 (excluded)
        ood_scores = np.array([
            -1.0, -1.0,   # ID disease
            0.0, 0.0,     # No Finding (should be excluded from disease_only)
            1.0, 1.0,     # Pure OOD
            1.0, 1.0,     # Mixed OOD
        ])
        result = compute_fourclass_metrics(
            ood_scores, label_vecs, LABEL_NAMES, HELD_OUT
        )
        assert result["auroc_disease_only"] == pytest.approx(1.0, abs=1e-5)
        assert result["fpr95_disease_only"] == pytest.approx(0.0, abs=0.01)

    def test_metrics_are_floats(self):
        """All metric values are floats (not numpy types)."""
        label_vecs = _make_dataset()
        ood_scores = np.linspace(-1, 1, 8)
        result = compute_fourclass_metrics(
            ood_scores, label_vecs, LABEL_NAMES, HELD_OUT
        )
        for key in result:
            if key == "counts":
                for v in result["counts"].values():
                    assert isinstance(v, int)
            else:
                assert isinstance(result[key], float), f"{key} is {type(result[key])}"

    def test_no_pure_ood_returns_nan(self):
        """If there are no pure OOD samples, auroc_pure and fpr95_pure are NaN."""
        # Only ID + NF + Mixed (no pure OOD)
        label_vecs = np.array([
            [1, 0, 0, 0, 0],  # ID
            [0, 0, 0, 0, 0],  # NF
            [1, 0, 0, 1, 0],  # Mixed
        ])
        ood_scores = np.array([0.1, 0.2, 0.9])
        result = compute_fourclass_metrics(
            ood_scores, label_vecs, LABEL_NAMES, HELD_OUT
        )
        assert np.isnan(result["auroc_pure"])
        assert np.isnan(result["fpr95_pure"])
        assert np.isnan(result["auroc_nf_vs_pure"])
        assert result["counts"]["pure_ood"] == 0

    def test_no_mixed_ood_returns_nan(self):
        """If there are no mixed OOD samples, auroc_mixed and fpr95_mixed are NaN."""
        label_vecs = np.array([
            [1, 0, 0, 0, 0],  # ID
            [0, 0, 0, 0, 0],  # NF
            [0, 0, 0, 1, 0],  # Pure OOD
        ])
        ood_scores = np.array([0.1, 0.2, 0.9])
        result = compute_fourclass_metrics(
            ood_scores, label_vecs, LABEL_NAMES, HELD_OUT
        )
        assert np.isnan(result["auroc_mixed"])
        assert np.isnan(result["fpr95_mixed"])
        assert result["counts"]["mixed_ood"] == 0

    def test_no_nf_images(self):
        """Dataset with no No Finding images still works correctly."""
        label_vecs = np.array([
            [1, 0, 0, 0, 0],  # ID
            [0, 1, 0, 0, 0],  # ID
            [0, 0, 0, 1, 0],  # Pure OOD
            [1, 0, 0, 1, 0],  # Mixed OOD
        ])
        ood_scores = np.array([-1.0, -1.0, 1.0, 1.0])
        result = compute_fourclass_metrics(
            ood_scores, label_vecs, LABEL_NAMES, HELD_OUT
        )
        assert result["counts"]["no_finding"] == 0
        # Full AUROC still computable: ID vs (Pure + Mixed)
        assert result["auroc_full"] == pytest.approx(1.0, abs=1e-5)
        # NF vs Pure: no NF => NaN
        assert np.isnan(result["auroc_nf_vs_pure"])


def test_auroc_mixed_vs_id_disease_present_and_correct():
    """New near-OOD pairing: ID-disease only (no NF) vs Mixed OOD.

    Constructs a dataset where the OOD score perfectly separates Mixed OOD
    (high score) from ID-disease (low score), and checks that the metric
    equals 1.0 regardless of the NF / Pure-OOD scores (which must be excluded).
    """
    import numpy as np
    from osr_metrics.fourclass import compute_fourclass_metrics

    label_vecs = _make_dataset()
    # Indices: 0,1=id_disease  2,3=NF  4,5=pure_OOD  6,7=mixed_OOD
    # Make NF and pure_OOD scores noise that would hurt other pairings if leaked in.
    ood_scores = np.array([0.1, 0.2, 5.0, -5.0, 0.5, 0.6, 0.9, 0.95])
    out = compute_fourclass_metrics(ood_scores, label_vecs, LABEL_NAMES, HELD_OUT)
    assert "auroc_mixed_vs_id_disease" in out
    assert out["auroc_mixed_vs_id_disease"] == 1.0


def test_auroc_mixed_vs_id_disease_excludes_nf_and_pure():
    """The new metric must depend only on ID-disease and Mixed scores."""
    import numpy as np
    from osr_metrics.fourclass import compute_fourclass_metrics

    label_vecs = _make_dataset()
    base = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.9, 0.95])
    out_a = compute_fourclass_metrics(base, label_vecs, LABEL_NAMES, HELD_OUT)
    # Perturb ONLY NF and pure_OOD scores; mixed-vs-id metric must be unchanged.
    perturbed = base.copy()
    perturbed[2] = 99.0  # NF
    perturbed[3] = -99.0  # NF
    perturbed[4] = 99.0  # pure_OOD
    perturbed[5] = -99.0  # pure_OOD
    out_b = compute_fourclass_metrics(perturbed, label_vecs, LABEL_NAMES, HELD_OUT)
    assert out_a["auroc_mixed_vs_id_disease"] == out_b["auroc_mixed_vs_id_disease"]

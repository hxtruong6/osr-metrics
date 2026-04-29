"""Four-class OOD partitioning and metrics.

**Scope: multi-label only.** The four-way partition relies on multi-hot
label vectors (No-Finding = all-zero vector, Mixed-OOD = known and
held-out labels co-occur). For multi-class (single-label) OSR, use
``ood.auroc`` / ``osr.compute_aoscr`` directly — every sample falls into
exactly one class so the four-class partition collapses to plain ID/OOD.

In a realistic deployment, a chest X-ray OOD detection system must handle
four distinct image types:

  1. **ID disease**  -- only known (non-held-out) labels are positive
  2. **No Finding**  -- all-zero label vector (healthy patient)
  3. **Pure OOD**    -- only held-out labels are positive
  4. **Mixed OOD**   -- both known AND held-out labels are positive

This module provides:
  - ``build_fourclass_masks``: partition images into the four types
  - ``compute_fourclass_metrics``: compute OOD detection metrics across
    clinically meaningful pairings of the four types
"""
from __future__ import annotations

import math

import numpy as np

from .ood import auroc, fpr_at_95tpr


def build_fourclass_masks(
    label_vecs: np.ndarray,
    label_names: list[str],
    held_out_labels: list[str],
) -> dict[str, np.ndarray]:
    """Partition images into four mutually exclusive types.

    | Applies to   | Task                          |
    |--------------|-------------------------------|
    | Multi-label  | OSR sub-population partition  |

    Args:
        label_vecs: Binary multi-hot label matrix, shape ``[N, K]``.
        label_names: List of *K* label names (column order of *label_vecs*).
        held_out_labels: Labels treated as unknown/OOD.

    Returns:
        Dictionary with keys ``'id_disease'``, ``'no_finding'``,
        ``'pure_ood'``, ``'mixed_ood'``, each a boolean array of shape
        ``[N]``.  The four masks are mutually exclusive and exhaustive.
    """
    N = label_vecs.shape[0]

    held_out_idx = [i for i, name in enumerate(label_names) if name in held_out_labels]
    known_idx = [i for i in range(len(label_names)) if i not in held_out_idx]

    has_known = label_vecs[:, known_idx].sum(axis=1) > 0 if known_idx else np.zeros(N, dtype=bool)
    has_held_out = label_vecs[:, held_out_idx].sum(axis=1) > 0 if held_out_idx else np.zeros(N, dtype=bool)

    # All-zero label vector = No Finding (healthy)
    no_finding = ~has_known & ~has_held_out

    # Only known labels positive
    id_disease = has_known & ~has_held_out

    # Only held-out labels positive
    pure_ood = ~has_known & has_held_out

    # Both known and held-out labels positive
    mixed_ood = has_known & has_held_out

    return {
        "id_disease": id_disease,
        "no_finding": no_finding,
        "pure_ood": pure_ood,
        "mixed_ood": mixed_ood,
    }


def _safe_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Return AUROC, or NaN if either class is absent."""
    if len(scores) == 0 or labels.sum() == 0 or (1 - labels).sum() == 0:
        return float("nan")
    return auroc(scores, labels)


def _safe_fpr95(scores: np.ndarray, labels: np.ndarray) -> float:
    """Return FPR@95TPR, or NaN if either class is absent."""
    if len(scores) == 0 or labels.sum() == 0 or (1 - labels).sum() == 0:
        return float("nan")
    return fpr_at_95tpr(scores, labels)


def compute_fourclass_metrics(
    ood_scores: np.ndarray,
    label_vecs: np.ndarray,
    label_names: list[str],
    held_out_labels: list[str],
) -> dict:
    """Compute OOD detection metrics across clinically meaningful pairings.

    | Applies to   | Task                          |
    |--------------|-------------------------------|
    | Multi-label  | OSR (near-/far-OOD breakdown) |

    Args:
        ood_scores: Per-image OOD scores, shape ``[N]``.  Higher = more OOD.
        label_vecs: Binary multi-hot label matrix, shape ``[N, K]``.
        label_names: List of *K* label names.
        held_out_labels: Labels treated as unknown/OOD.

    Returns:
        Dictionary with the following keys:

        - **auroc_full** / **fpr95_full**: (ID + NF) vs (Pure + Mixed) OOD
        - **auroc_pure** / **fpr95_pure**: (ID + NF) vs Pure OOD only
        - **auroc_mixed** / **fpr95_mixed**: (ID + NF) vs Mixed OOD only
        - **auroc_mixed_vs_id_disease**: ID-disease only (no NF) vs Mixed OOD.
          Near-OOD sensitivity diagnostic: can the score detect held-out
          content even when known disease is also present and NF is removed
          from the negatives?
        - **auroc_nf_vs_pure**: No Finding vs Pure OOD (critical clinical pair)
        - **auroc_disease_only** / **fpr95_disease_only**: ID-disease vs all
          OOD (backward-compatible with v15 which excludes NF)
        - **counts**: per-type sample counts
    """
    masks = build_fourclass_masks(label_vecs, label_names, held_out_labels)

    id_mask = masks["id_disease"]
    nf_mask = masks["no_finding"]
    pure_mask = masks["pure_ood"]
    mixed_mask = masks["mixed_ood"]

    in_dist = id_mask | nf_mask  # "negative" class (label 0)
    all_ood = pure_mask | mixed_mask  # "positive" class (label 1)

    # --- Full: (ID + NF) vs (Pure + Mixed) ---
    sel_full = in_dist | all_ood
    labels_full = all_ood[sel_full].astype(int)
    scores_full = ood_scores[sel_full]

    # --- Pure: (ID + NF) vs Pure OOD ---
    sel_pure = in_dist | pure_mask
    labels_pure = pure_mask[sel_pure].astype(int)
    scores_pure = ood_scores[sel_pure]

    # --- Mixed: (ID + NF) vs Mixed OOD ---
    sel_mixed = in_dist | mixed_mask
    labels_mixed = mixed_mask[sel_mixed].astype(int)
    scores_mixed = ood_scores[sel_mixed]

    # --- Near-OOD: ID-disease only (no NF) vs Mixed OOD ---
    sel_mixed_vs_idd = id_mask | mixed_mask
    labels_mixed_vs_idd = mixed_mask[sel_mixed_vs_idd].astype(int)
    scores_mixed_vs_idd = ood_scores[sel_mixed_vs_idd]

    # --- NF vs Pure OOD (critical clinical pair) ---
    sel_nf_pure = nf_mask | pure_mask
    labels_nf_pure = pure_mask[sel_nf_pure].astype(int)
    scores_nf_pure = ood_scores[sel_nf_pure]

    # --- Disease-only (backward compat): ID-disease vs all OOD ---
    sel_disease = id_mask | all_ood
    labels_disease = all_ood[sel_disease].astype(int)
    scores_disease = ood_scores[sel_disease]

    return {
        "auroc_full": _safe_auroc(scores_full, labels_full),
        "fpr95_full": _safe_fpr95(scores_full, labels_full),
        "auroc_pure": _safe_auroc(scores_pure, labels_pure),
        "fpr95_pure": _safe_fpr95(scores_pure, labels_pure),
        "auroc_mixed": _safe_auroc(scores_mixed, labels_mixed),
        "fpr95_mixed": _safe_fpr95(scores_mixed, labels_mixed),
        "auroc_mixed_vs_id_disease": _safe_auroc(scores_mixed_vs_idd, labels_mixed_vs_idd),
        "auroc_nf_vs_pure": _safe_auroc(scores_nf_pure, labels_nf_pure),
        "auroc_disease_only": _safe_auroc(scores_disease, labels_disease),
        "fpr95_disease_only": _safe_fpr95(scores_disease, labels_disease),
        "counts": {
            "id_disease": int(id_mask.sum()),
            "no_finding": int(nf_mask.sum()),
            "pure_ood": int(pure_mask.sum()),
            "mixed_ood": int(mixed_mask.sum()),
        },
    }

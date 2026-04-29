"""osr-metrics: Open-Set Recognition and OOD-detection metrics for ML research.

All OOD/novelty scores follow the convention: **higher = more OOD**.
ID-positive metrics (e.g. ``aupr_in``) handle the sign internally.
"""
from __future__ import annotations

from .ood import (
    auroc,
    aupr_in,
    aupr_out,
    bootstrap_ci,
    fpr_at_95tpr,
    fpr_at_tpr,
    oscr_curve,
    partition_ood_by_purity,
)
from .osr import compute_aoscr, compute_nf_rejection_at_tpr
from .classification import (
    f1_per_label,
    macro_auprc,
    macro_auprc_id_labels,
    macro_f1_with_thresholds,
    per_label_auprc,
)
from .fourclass import build_fourclass_masks, compute_fourclass_metrics
from .calibration import brier_score, expected_calibration_error
from .delong import delong_test
from .stability import *  # noqa: F401, F403

__version__ = "0.1.0"

__all__ = [
    # OOD detection
    "auroc",
    "aupr_in",
    "aupr_out",
    "fpr_at_tpr",
    "fpr_at_95tpr",
    # Open-Set Recognition
    "compute_aoscr",
    "compute_nf_rejection_at_tpr",
    "oscr_curve",
    # Classification
    "macro_auprc",
    "macro_auprc_id_labels",
    "macro_f1_with_thresholds",
    "per_label_auprc",
    "f1_per_label",
    # Four-class partition
    "build_fourclass_masks",
    "compute_fourclass_metrics",
    "partition_ood_by_purity",
    # Calibration
    "expected_calibration_error",
    "brier_score",
    # Statistical comparison
    "delong_test",
    "bootstrap_ci",
]

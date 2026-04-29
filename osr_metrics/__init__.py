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
from .osr import (
    compute_aoscr,
    compute_aoscr_multiclass,
    compute_nf_rejection_at_tpr,
)
from .classification import (
    f1_per_label,
    macro_auprc,
    macro_auprc_id_labels,
    macro_f1_with_thresholds,
    per_label_auprc,
)
from .multiclass import (
    balanced_accuracy,
    macro_f1_multiclass,
    top1_accuracy,
)
from .fourclass import build_fourclass_masks, compute_fourclass_metrics
from .calibration import (
    brier_score,
    brier_score_multiclass,
    expected_calibration_error,
    expected_calibration_error_multiclass,
)
from .delong import delong_test
from .panel import compute_panel
from .utils import as_ood_scores, warn_if_inverted_scores
from .stability import *  # noqa: F401, F403

from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    __version__ = _pkg_version("osr-metrics")
except PackageNotFoundError:  # not installed (e.g. running from source tree without install)
    __version__ = "0.0.0+unknown"

__all__ = [
    # OOD detection
    "auroc",
    "aupr_in",
    "aupr_out",
    "fpr_at_tpr",
    "fpr_at_95tpr",
    # Open-Set Recognition
    "compute_aoscr",
    "compute_aoscr_multiclass",
    "compute_nf_rejection_at_tpr",
    "oscr_curve",
    # Multi-label classification
    "macro_auprc",
    "macro_auprc_id_labels",
    "macro_f1_with_thresholds",
    "per_label_auprc",
    "f1_per_label",
    # Multi-class (single-label) classification
    "top1_accuracy",
    "macro_f1_multiclass",
    "balanced_accuracy",
    # Four-class partition
    "build_fourclass_masks",
    "compute_fourclass_metrics",
    "partition_ood_by_purity",
    # Calibration
    "expected_calibration_error",
    "expected_calibration_error_multiclass",
    "brier_score",
    "brier_score_multiclass",
    # Statistical comparison
    "delong_test",
    "bootstrap_ci",
    # Utilities
    "as_ood_scores",
    "warn_if_inverted_scores",
    # Publication panel
    "compute_panel",
]

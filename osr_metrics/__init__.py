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
    paired_bootstrap_diff,
    partition_ood_by_purity,
)
from .osr import (
    aml_oscr_curve,
    compute_aml_oscr,
    compute_aoscr,
    compute_aoscr_multiclass,
    compute_nf_rejection_at_tpr,
    per_novel_discovery_table,
)
from .classification import (
    compute_rc_macro_f1,
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
from .selective import (
    aurc,
    eaurc,
    rc_curve,
    selective_accuracy_at_coverage,
    selective_risk_at_coverage,
    warn_if_inverted_aurc,
)
from .stability import *  # noqa: F401, F403

from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    __version__ = _pkg_version("osr-metrics")
except PackageNotFoundError:  # not installed (e.g. running from source tree without install)
    __version__ = "0.0.0+unknown"

__all__ = [
    # Entries are grouped by category and sorted alphabetically within each
    # group. Match the README capability matrix and docs/METRICS.md ordering.
    # OOD detection
    "aupr_in",
    "aupr_out",
    "auroc",
    "fpr_at_95tpr",
    "fpr_at_tpr",
    # Open-Set Recognition
    "compute_aoscr",
    "compute_aoscr_multiclass",
    "compute_nf_rejection_at_tpr",
    "oscr_curve",
    # Open-Set Multi-Label Classification (OS-MLC)
    "aml_oscr_curve",
    "compute_aml_oscr",
    "compute_rc_macro_f1",
    "per_novel_discovery_table",
    # Multi-label classification
    "f1_per_label",
    "macro_auprc",
    "macro_auprc_id_labels",
    "macro_f1_with_thresholds",
    "per_label_auprc",
    # Multi-class (single-label) classification
    "balanced_accuracy",
    "macro_f1_multiclass",
    "top1_accuracy",
    # Four-class partition
    "build_fourclass_masks",
    "compute_fourclass_metrics",
    "partition_ood_by_purity",
    # Selective prediction
    "aurc",
    "eaurc",
    "rc_curve",
    "selective_accuracy_at_coverage",
    "selective_risk_at_coverage",
    "warn_if_inverted_aurc",
    # Calibration
    "brier_score",
    "brier_score_multiclass",
    "expected_calibration_error",
    "expected_calibration_error_multiclass",
    # Statistical comparison
    "bootstrap_ci",
    "delong_test",
    "paired_bootstrap_diff",
    # Utilities
    "as_ood_scores",
    "warn_if_inverted_scores",
    # Publication panel
    "compute_panel",
]

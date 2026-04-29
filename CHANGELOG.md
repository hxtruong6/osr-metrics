# Changelog

All notable changes to `osr-metrics` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.2] — 2026-04-29

### Added
- PyPI distribution: `pip install osr-metrics`.
- GitHub Actions CI: pytest matrix on Python 3.10/3.11/3.12, `python -m build`, `twine check`.
- Tag-driven release workflow with Trusted Publishing (OIDC). Pre-release tags
  (`vX.Y.ZrcN`) publish to TestPyPI; stable tags (`vX.Y.Z`) publish to PyPI.
- `osr_metrics.__version__` now sourced from installed metadata via
  `importlib.metadata` (was a hardcoded string).
- `docs/RELEASING.md` — maintainer checklist for cutting releases.
- `docs/USAGE.md` — decision-tree guide for picking the right metric.
- `docs/EXAMPLES.md` — end-to-end runnable example covering the full
  publication metric panel, DeLong comparison, and seed aggregation.
- `CHANGELOG.md` (this file).
- `CITATION.cff` — machine-readable citation metadata.

### Changed
- `pyproject.toml` metadata: real Repository/Issues/Changelog URLs, author
  email, SPDX-format license, removed deprecated MIT classifier.
- README primary install instruction is now `pip install osr-metrics`;
  editable install moved to a Development subsection.

### Removed
- Tracked build artifacts (`*.egg-info/`, `.pytest_cache/`) — they were
  always supposed to be ignored.

## [0.1.1] — 2026-04-29

### Added
- `auroc_mixed_vs_id_disease` key in `compute_fourclass_metrics`: AUROC of
  ID-disease only (no NF) vs Mixed OOD. Isolates near-OOD sensitivity by
  removing the no-finding population from the negatives.
- README and docstrings document the five available AUROC pairings
  (`auroc_full`, `auroc_pure`, `auroc_mixed`, `auroc_mixed_vs_id_disease`,
  `auroc_nf_vs_pure`) and what scientific question each one answers.
- Two regression tests for the new pairing.

### Changed
- README's four-class section now includes a side-by-side table of the five
  pairings.

## [0.1.0] — 2026-04-29

### Added
- Initial release. 17 metric functions across six groups:
  - **OOD detection**: `auroc`, `fpr_at_tpr`, `fpr_at_95tpr`, `aupr_in`,
    `aupr_out`.
  - **Open-Set Recognition**: `compute_aoscr` (canonical Dhamija/Vaze
    convention), `oscr_curve`, `compute_nf_rejection_at_tpr`.
  - **Multi-label classification**: `macro_auprc`, `macro_auprc_id_labels`,
    `macro_f1_with_thresholds`, `per_label_auprc`, `f1_per_label`.
  - **Four-class OSR partitioning**: `build_fourclass_masks`,
    `compute_fourclass_metrics`, `partition_ood_by_purity`.
  - **Calibration**: `expected_calibration_error` (Guo et al. 2017),
    `brier_score`.
  - **Statistical comparison**: `delong_test` (DeLong 1988, O(n log n)
    rank-based), `bootstrap_ci` (with optional class-stratification).
- 57 tests covering numerical equivalence to first-principles references,
  edge cases, and known invariants.
- `pyproject.toml` for editable install (`pip install -e .`) on Python 3.10+.
- MIT license, README with quickstart examples.

### Bug fixes (vs. lifted source code)
- `oscr_curve` rewritten to follow the canonical Dhamija/Vaze convention
  (FPR = OOD acceptance rate). Previously used a non-canonical variant that
  computed FPR as the ID rejection rate, producing different AOSCR values.
- `delong._placement_values` reduced from O(n²) Python loop to O(n log n)
  rank-based implementation using `scipy.stats.rankdata`. Numerically
  equivalent to brute-force (max diff 1e-12), ~60× faster on N=8000.
- `bootstrap_ci` accepts `stratify=True` to resample positives and negatives
  separately. Required for very imbalanced OOD splits where unstratified
  replicates can be all-positive or all-negative (producing NaN AUROC).

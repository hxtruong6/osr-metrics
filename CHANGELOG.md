# Changelog

All notable changes to `osr-metrics` are documented here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.3] — 2026-04-29

Documentation release. Clarifies which task each metric applies to —
multi-class classification, multi-label classification, or both — so
users can find the right function without reading source.

### Added
- **Capability matrix in README.md** — 14-row × 6-column grid mapping
  every public function to its supported task types, plus a new "Scope"
  section that explicitly states this library covers semantic-shift
  detection only (covariate shift, regression, and continual learning
  are out of scope). Four numbered footnotes call out the ⚠ cases.
- **Per-function "Applies to" tables in docstrings** — all 16 public
  functions across `ood.py`, `osr.py`, `fourclass.py`, `classification.py`,
  `calibration.py`, and `delong.py` now show their applicable task types
  at the top of `help(...)`.
- **Module-level scope banners** in each metric module:
  - `ood.py` — task-agnostic
  - `osr.py` — multi-class + multi-label, with NF helper multi-label only
  - `fourclass.py` — multi-label only
  - `classification.py` — multi-label only
  - `calibration.py` — multi-label / binary; multi-class on roadmap
  - `delong.py` — task-agnostic, paired

### Changed
- `docs/USAGE.md` restructured around a three-step preamble (task type →
  goal → look up function) with a mermaid flowchart. Now points readers
  at the README capability matrix as the primary reference.

## [0.1.2] — 2026-04-29

First public release on PyPI. Install with `pip install osr-metrics`.

### Added
- PyPI distribution: `pip install osr-metrics` now works.
- `osr_metrics.__version__` reflects the installed package version, read
  via `importlib.metadata` instead of a hardcoded string. The two cannot
  drift after install.
- `docs/USAGE.md` — decision-tree guide for picking the right metric.
- `docs/EXAMPLES.md` — end-to-end example covering the full publication
  metric panel, DeLong comparison, and seed aggregation.
- `docs/RELEASING.md` — maintainer checklist for cutting releases.
- `CITATION.cff` — machine-readable citation metadata.
- This `CHANGELOG.md`.

### Changed
- README primary install instruction is now `pip install osr-metrics`;
  editable install moved under "Development install".
- `pyproject.toml` metadata modernized: real Repository/Homepage/Issues/
  Changelog URLs, author email, SPDX-format license.
- Releases are tag-driven via GitHub Actions with PyPI Trusted Publishing
  (OIDC, no API tokens). Pre-release tags `vX.Y.ZrcN` ship to TestPyPI;
  stable tags `vX.Y.Z` ship to PyPI. See `docs/RELEASING.md`.

### Removed
- Tracked build artifacts (`*.egg-info/`, `.pytest_cache/`) that were
  always supposed to be ignored.

## [0.1.1] — 2026-04-29

### Added
- `auroc_mixed_vs_id_disease` key in `compute_fourclass_metrics`: AUROC
  of ID-disease only (no NF) vs Mixed OOD. Isolates near-OOD sensitivity
  by removing the no-finding population from the negatives.
- Documentation of the five available AUROC pairings (`auroc_full`,
  `auroc_pure`, `auroc_mixed`, `auroc_mixed_vs_id_disease`,
  `auroc_nf_vs_pure`) and what scientific question each one answers.
- Two regression tests for the new pairing.

### Changed
- README's four-class section now shows a side-by-side table of the
  five pairings.

## [0.1.0] — 2026-04-29

Initial release. 17 metric functions across six groups, plain-numpy API,
no PyTorch/TensorFlow dependency.

### Added
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
  rank-based), `bootstrap_ci` with optional class-stratification.
- 57 tests covering numerical equivalence to first-principles references,
  edge cases, and known invariants.
- MIT license; README with quickstart examples.

### Fixed
- `oscr_curve` follows the canonical Dhamija/Vaze convention
  (FPR = OOD acceptance rate). A previously circulating variant computed
  FPR as the ID rejection rate, producing different AOSCR values.
- `delong._placement_values` reduced from an O(n²) Python loop to an
  O(n log n) rank-based implementation using `scipy.stats.rankdata`.
  Numerically equivalent to brute-force (max diff 1e-12), ~60× faster on
  N=8000.
- `bootstrap_ci` accepts `stratify=True` to resample positives and
  negatives separately. Required for very imbalanced OOD splits where
  unstratified replicates can be all-positive or all-negative (producing
  NaN AUROC).

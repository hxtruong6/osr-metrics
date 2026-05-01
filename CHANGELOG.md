# Changelog

All notable changes to `osr-metrics` are documented here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.4.0] — 2026-05-01

Performance release: AOSCR/OSCR rewritten as O(N log N), `bootstrap_ci` gains optional threading.

### Added

- `bootstrap_ci` accepts `n_jobs` (default `1`). `n_jobs > 1` runs `metric_fn` over replicates on a `ThreadPoolExecutor`; `-1` uses `os.cpu_count()`; clamped to `min(n_jobs, n_bootstrap, os.cpu_count())`. Indices are drawn in the main thread, so output is bit-exact with the serial path for any `seed`. Threading (not multiprocessing) keeps lambdas and closures working. Measured at `n_bootstrap=1000`, metric=`auroc`, 4-core: parallelism hurts at N=1k, ~1.6× at N=10k (`n_jobs=2`), ~2.9× at N=100k (`n_jobs=4`). Use only when serial takes more than a few seconds.

### Changed

- `compute_aoscr`, `compute_aoscr_multiclass`, and `oscr_curve` switched to an exact O(N log N) sort + cumulative-counts implementation. Faster (73× at N=1k, 14× at N=10k, 6.5× at N=100k, 5× at N=1M) and free of grid-discretization bias (agreement with the old loop within 2e-5). `oscr_curve` now returns ≤ N+1 points (one per unique score plus a `(0, 0)` anchor) instead of `n_thresholds` points, with tied scores collapsed for input-order invariance.
- `n_thresholds` is deprecated on `compute_aoscr`, `compute_aoscr_multiclass`, and `oscr_curve`. Passing it emits `DeprecationWarning`; it will be removed in a later release.

## [0.3.1] — 2026-04-30

Docs and CI maintenance release. No public-API changes.

### Added

- Officially support Python 3.13 and 3.14: trove classifiers and CI matrix entries. No code changes required — the library is pure-numpy/scipy/sklearn and all dependencies ship wheels for both versions.
- Selective-prediction quick-start in `README.md` showing `aurc`, `eaurc`, and `selective_risk_at_coverage` alongside the panel `loss=` integration.
- Pre-publish checklist in `docs/RELEASING.md` covering branch hygiene, tests, mypy, build, version bump, CHANGELOG promotion, and TestPyPI dry-run decision.

### Changed

- README Downloads badge swapped from shields.io `pypi/dm` to pepy.tech (total + monthly), linking to the pepy project page.
- GitHub Actions workflows pinned to current Node-24-compatible major versions (`actions/checkout@v5`, `actions/setup-python@v6`, `actions/upload-artifact@v5`, `actions/download-artifact@v5`) and opted into Node 24 runtime via `FORCE_JAVASCRIPT_ACTIONS_TO_NODE24=true` to silence deprecation warnings until upstream actions ship Node-24 declarations.

## [0.3.0] — 2026-04-30

Selective prediction / risk–coverage support. Adds the canonical AURC
family alongside the existing OOD/OSR/calibration toolbox so users can
report a complete selective-prediction picture from cached scores
without a torch dependency.

### Added

- New module `osr_metrics.selective` with selective-prediction / risk–coverage metrics: `rc_curve`, `aurc`, `eaurc`, `selective_risk_at_coverage`, `selective_accuracy_at_coverage`, `warn_if_inverted_aurc`. Follows the library's "higher = more OOD" convention; rank-averaged tie handling for sign-symmetric, input-order-independent results. AURC uses the canonical Riemann-sum formulation matching Geifman & El-Yaniv 2017 and standard reference implementations (Galil 2023, Han 2024).

### Changed

- `compute_panel` accepts an optional `loss=` keyword argument. When supplied alongside `scores=`, the panel additionally computes `aurc`, `eaurc`, and `selective_risk@95`. Existing call sites are unaffected.

## [0.2.0] — 2026-04-30

Multi-class first-class support, one-call publication panel, and an
explicit "which metric for which setting?" surface across docs, code,
and validation. The library now closes the multi-class functional gap
that previously forced users to drop down to `sklearn` and remember
`argmax` / `softmax` conventions per metric.

### Added
- **Multi-class (single-label) closed-set metrics** in new
  `osr_metrics/multiclass.py`:
  - `top1_accuracy(preds, y)`
  - `macro_f1_multiclass(preds, y)`
  - `balanced_accuracy(preds, y)`
  - All accept either integer predictions `[N]` or a logits / softmax
    matrix `[N, K]` — no `argmax` boilerplate.
- **Multi-class OSR convenience wrapper** `compute_aoscr_multiclass`
  in `osr_metrics/osr.py`. Accepts integer predictions or logits;
  internally reduces to class IDs and calls `compute_aoscr`.
- **Multi-class calibration overloads** in `osr_metrics/calibration.py`:
  - `expected_calibration_error_multiclass(probs, y)` — Guo 2017 form
    (top-1 confidence vs top-1 correctness).
  - `brier_score_multiclass(probs, y)` — sum-over-classes form, range
    `[0, 2]`.
  - Both raise a clear `ValueError` if raw logits are passed by
    mistake (most common user error).
- **Score-direction adapter** in new `osr_metrics/utils.py`:
  - `as_ood_scores(scores, direction="ood"|"confidence"|"id")` — flips
    sign once at the boundary.
  - `warn_if_inverted_scores(scores, labels, threshold=0.5)` — emits a
    warning when AUROC < 0.5, suggesting the likely score-direction fix.
- **One-call publication panel** `compute_panel(...)` in new
  `osr_metrics/panel.py`. Auto-infers multi-class vs multi-label from
  input shapes; computes every metric whose required inputs are
  present; gracefully skips missing pieces. Returns a flat dict with
  one nested `fourclass` block for multi-label.
- **Input validation** for the OOD-detection entry points
  (`auroc`, `fpr_at_tpr`, `aupr_in`, `aupr_out`):
  - 1-D scores and labels, length match, binary `{0, 1}` labels,
    `target_tpr ∈ (0, 1]`. Errors include a hint pointing at the right
    fix.
- **Documentation**:
  - `docs/CONCEPTS.md` — glossary covering ID/OOD, OSR vs OOD
    detection, semantic vs covariate shift, near vs far OOD,
    multi-class vs multi-label, score direction, and what's out of
    scope.
  - `docs/PITFALLS.md` — the eight most-hit mistakes, with bad-vs-good
    code side by side. Linked from README and `USAGE.md`.
  - New multi-class worked example in `docs/EXAMPLES.md`, including
    softmax handling and the one-call `compute_panel` form.
- **mypy** configured in `pyproject.toml` with a practical (not
  `--strict`) ruleset that catches Optional misuse, untyped def bodies,
  and unreachable branches without forcing `NDArray[float64]`
  everywhere. Run via `mypy` from the repo root. Added to dev extras.
- **Tests**: 33 new tests across `tests/test_multiclass.py` (T5–T7) and
  `tests/test_utils_panel.py` (T8 / T13 / T14). Total: 95 passing.

### Changed
- README capability matrix updated to include all new functions; the
  former ⚠³ (multi-class closed-set) and ⚠⁴ (multi-class calibration)
  caveats are now ✅. Footnotes consolidated.
- `docs/USAGE.md` "Common mistakes" section trimmed and now points at
  the comprehensive `PITFALLS.md`. Decision-tree mermaid updated to
  reference the new multi-class functions.
- `docs/EXAMPLES.md` reorganised: multi-class example added as
  Example 1; multi-label panel renumbered to Example 2; the local
  helper `compute_panel` renamed to `compute_panel_manual` to avoid
  shadowing the new public `osr_metrics.compute_panel`.
- Module-level docstring of `osr_metrics/osr.py` no longer references
  the legacy `src.metrics.osr` path or the project-internal "v17"
  milestone label.

### Fixed
- `compute_aoscr_multiclass` correctly threads `n_thresholds` through
  to `compute_aoscr`.

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

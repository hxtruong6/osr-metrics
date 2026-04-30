# osr-metrics

[![PyPI version](https://img.shields.io/pypi/v/osr-metrics.svg)](https://pypi.org/project/osr-metrics/)
[![Python versions](https://img.shields.io/pypi/pyversions/osr-metrics.svg)](https://pypi.org/project/osr-metrics/)
[![Downloads](https://static.pepy.tech/badge/osr-metrics)](https://pepy.tech/projects/osr-metrics)
[![Downloads/month](https://static.pepy.tech/badge/osr-metrics/month)](https://pepy.tech/projects/osr-metrics)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/hxtruong6/osr-metrics/blob/main/LICENSE)
[![CI](https://github.com/hxtruong6/osr-metrics/actions/workflows/ci.yml/badge.svg)](https://github.com/hxtruong6/osr-metrics/actions/workflows/ci.yml)

**Plain-numpy metrics for Open-Set Recognition and OOD-detection research — no PyTorch, no datasets, just the math.**

## Why osr-metrics?

Most OSR / OOD libraries (PyTorch-OOD, OpenOOD) couple metrics with detection
methods, datasets, and a heavy framework. `osr-metrics` is *just the metrics* —
useful when you have cached scores from any pipeline and want to compute
AOSCR, FPR@95TPR, or DeLong on them, regardless of how those scores were
produced.

- **Framework-agnostic** — `numpy` arrays in, scalars or dicts out. No PyTorch / TensorFlow / dataset dependencies.
- **Verified formulas** — every metric checked against a first-principles brute-force reference.
- **Consistent conventions** — for every OOD/novelty score, **higher = more OOD**. ID-positive metrics (`aupr_in`) handle the sign flip internally.
- **Statistical rigor** — DeLong (O(n log n) rank-based) and stratified bootstrap CIs are first-class, not afterthoughts.

## What's inside

| Group | Metrics |
|---|---|
| OOD detection | `auroc`, `fpr_at_tpr`, `fpr_at_95tpr`, `aupr_in`, `aupr_out` |
| Open-Set Recognition | `compute_aoscr` (canonical Dhamija/Vaze), `compute_aoscr_multiclass`, `oscr_curve`, `compute_nf_rejection_at_tpr` |
| Multi-class (single-label) classification | `top1_accuracy`, `macro_f1_multiclass`, `balanced_accuracy` |
| Multi-label classification | `macro_auprc`, `macro_auprc_id_labels`, `macro_f1_with_thresholds`, `per_label_auprc`, `f1_per_label` |
| Four-class OSR partitioning | `build_fourclass_masks`, `compute_fourclass_metrics`, `partition_ood_by_purity` |
| Calibration | `expected_calibration_error`, `expected_calibration_error_multiclass`, `brier_score`, `brier_score_multiclass` |
| Statistical comparison | `delong_test` (O(n log n) rank-based), `bootstrap_ci` (with optional stratification) |
| Selective prediction | `rc_curve`, `aurc`, `eaurc`, `selective_risk_at_coverage`, `selective_accuracy_at_coverage`, `warn_if_inverted_aurc` |
| Utilities | `as_ood_scores` (score-direction adapter), `warn_if_inverted_scores`, `compute_panel` (one-call publication panel) |

All functions take plain `numpy` arrays and return scalars or simple
dictionaries — no PyTorch, TensorFlow, or framework lock-in.

## Scope

This library targets the **semantic-shift** setting (OSR / near-OOD /
far-OOD): novel class labels appear at test time. Covariate shift
(domain generalization), regression OOD, and continual / open-world
learning are **out of scope**.

## Capability matrix — which function for which setting?

Read across to find your setting; functions marked ✅ apply directly.
⚠ = applies with a small adapter (see footnote). ❌ = not applicable.

| Function | Multi-class<br>(single-label) | Multi-label | Pure OOD<br>detection | OSR<br>(classify+reject) | Calibration | Statistical<br>test |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| `auroc` | ✅ | ✅ | ✅ | — | — | — |
| `fpr_at_tpr` / `fpr_at_95tpr` | ✅ | ✅ | ✅ | — | — | — |
| `aupr_in` / `aupr_out` | ✅ | ✅ | ✅ | — | — | — |
| `compute_aoscr` / `oscr_curve` | ✅ | ⚠ ¹ | — | ✅ | — | — |
| `compute_aoscr_multiclass` | ✅ | ❌ | — | ✅ | — | — |
| `compute_nf_rejection_at_tpr` | ❌ | ✅ | — | ✅ ² | — | — |
| `partition_ood_by_purity` | ❌ | ✅ | — | ✅ ² | — | — |
| `build_fourclass_masks` / `compute_fourclass_metrics` | ❌ | ✅ | — | ✅ ² | — | — |
| `top1_accuracy` / `macro_f1_multiclass` / `balanced_accuracy` | ✅ | ❌ | — | — | — | — |
| `macro_auprc` / `macro_auprc_id_labels` | ❌ | ✅ | — | — | — | — |
| `per_label_auprc` / `f1_per_label` | ❌ | ✅ | — | — | — | — |
| `macro_f1_with_thresholds` | ❌ | ✅ | — | — | — | — |
| `expected_calibration_error` | ❌ | ✅ | — | — | ✅ | — |
| `expected_calibration_error_multiclass` | ✅ | ❌ | — | — | ✅ | — |
| `brier_score` | ❌ | ✅ | — | — | ✅ | — |
| `brier_score_multiclass` | ✅ | ❌ | — | — | ✅ | — |
| `delong_test` | ✅ | ✅ | ✅ | ✅ | — | ✅ |
| `bootstrap_ci` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `rc_curve` / `aurc` / `eaurc` | ✅ | ✅ | — | — | — | — |
| `selective_risk_at_coverage` / `selective_accuracy_at_coverage` | ✅ | ✅ | — | — | — | — |
| `warn_if_inverted_aurc` | ✅ | ✅ | — | — | — | — |
| `as_ood_scores` / `warn_if_inverted_scores` | ✅ | ✅ | ✅ | ✅ | — | — |
| `compute_panel` | ✅ | ✅ | ✅ | ✅ | ✅ | — |

¹ Multi-label OSCR/AOSCR: pass an exact-match indicator
(`1` if all labels predicted correctly, else `0`) as `class_predictions`
with `true_classes=ones(N)`. See `compute_aoscr` docstring. (For
multi-class, use `compute_aoscr_multiclass` instead — it accepts logits
or class-IDs directly.)

² Clinical / multi-label OSR helpers — depend on a per-sample
"No Finding" (all-zero label vector) indicator that has no analogue in
multi-class single-label settings.

## Score-direction convention

For every OOD/novelty metric in this library, **higher score = more OOD**.
ID-positive metrics (`aupr_in`) handle the sign flip internally so you don't
have to.

## Install

```bash
pip install osr-metrics
```

Requires Python 3.10–3.14, `numpy`, `scikit-learn`, `scipy`. CI runs on
Python 3.10, 3.11, 3.12, 3.13, and 3.14.

### Development install

```bash
git clone https://github.com/hxtruong6/osr-metrics.git
cd osr-metrics
pip install -e .[dev]
```

## Quick start

```python
import numpy as np
from osr_metrics import (
    auroc, fpr_at_95tpr,
    compute_aoscr_multiclass,
    expected_calibration_error_multiclass,
)

rng = np.random.default_rng(0)

# OOD detection: 800 ID points, 200 OOD points with shifted score distribution.
id_scores  = rng.normal(0.0, 1.0, size=800)   # ID:  N(0, 1)
ood_scores = rng.normal(2.0, 1.0, size=200)   # OOD: N(2, 1) — higher = more OOD
scores = np.concatenate([id_scores, ood_scores])
labels = np.concatenate([np.zeros(800), np.ones(200)])  # 1 = OOD
print(f"AUROC:     {auroc(scores, labels):.3f}")        # ~0.92
print(f"FPR@95TPR: {fpr_at_95tpr(scores, labels):.3f}") # ~0.36

# Open-Set Classification Rate: joint classify + reject, 80% closed-set accuracy.
n, k = 1000, 5
true_cls = rng.integers(0, k, size=n)
correct  = rng.random(n) < 0.80
pred_cls = np.where(correct, true_cls, (true_cls + 1) % k)
print(f"AOSCR:     {compute_aoscr_multiclass(scores, labels, pred_cls, true_cls):.3f}")

# Multi-class softmax calibration (Guo 2017 form).
probs = rng.dirichlet(np.ones(k) * 0.5, size=n)
probs[np.arange(n), true_cls] += 1.0          # bias toward the correct class
probs /= probs.sum(axis=1, keepdims=True)
print(f"ECE:       {expected_calibration_error_multiclass(probs, true_cls):.3f}")
```

## One-call publication panel

When you have all the inputs and just want the table:

```python
from osr_metrics import compute_panel

# Multi-class
out = compute_panel(scores, ood_labels, probs=softmax_NK, y=y_N)

# Multi-label
out = compute_panel(
    scores, ood_labels,
    preds=preds_NK, probs=probs_NK,
    label_vecs=labels_NK, label_names=names, held_out_labels=held_out,
    setting="multilabel",
)
```

The panel infers your setting from input shapes and computes every
metric whose required inputs are present.

## Selective prediction (risk–coverage)

When you have per-sample losses (e.g. `0/1` for misclassification) and a
confidence-style ranking score, report AURC and selective risk at a
chosen coverage:

```python
from osr_metrics import aurc, eaurc, selective_risk_at_coverage

# Convention: ood_score is "higher = more OOD" (i.e., reject first).
# If you have a confidence score, pass `-confidence`.
print(f"AURC:  {aurc(ood_score, loss):.4f}")
print(f"E-AURC: {eaurc(ood_score, loss):.4f}")
print(f"Risk@95% coverage: {selective_risk_at_coverage(ood_score, loss, 0.95):.4f}")
```

`compute_panel(..., loss=loss)` adds these to the publication panel.
See [`docs/USAGE.md`](docs/USAGE.md) and
[`docs/PITFALLS.md`](docs/PITFALLS.md) for score-direction guidance.

## Statistical comparison

```python
from osr_metrics import delong_test, bootstrap_ci, auroc

# Pairwise AUROC comparison (DeLong 1988)
z, p = delong_test(scores_method_a, scores_method_b, labels)
print(f"DeLong z={z:.3f}, p={p:.4f}")

# Bootstrap CI (use stratify=True for imbalanced data)
lo, mean, hi = bootstrap_ci(scores, labels, auroc, n_bootstrap=1000, stratify=True)
print(f"AUROC = {mean:.4f}  95% CI = [{lo:.4f}, {hi:.4f}]")
```

## Four-class OSR partitioning

For multi-label problems with held-out labels (chest X-ray OSR style):

```python
from osr_metrics import build_fourclass_masks, compute_fourclass_metrics

label_names = ["A", "B", "C", "D"]
held_out = ["C", "D"]
metrics = compute_fourclass_metrics(scores, label_vecs, label_names, held_out)
# Returns: auroc_full, fpr95_full, auroc_pure, auroc_mixed,
#          auroc_mixed_vs_id_disease, auroc_nf_vs_pure,
#          auroc_disease_only, counts...
```

Partitions images into four mutually exclusive classes:
- `id_disease` — only known labels
- `no_finding` — all-zero label vector
- `pure_ood` — only held-out labels
- `mixed_ood` — both known + held-out labels

Five AUROC pairings answer different questions:

| Key | Negatives | Positives | What it asks |
|---|---|---|---|
| `auroc_pure` | ID-disease + NF | Pure OOD | Upper-bound separability |
| `auroc_mixed` | ID-disease + NF | Mixed OOD | Mixed-OOD detection difficulty |
| `auroc_mixed_vs_id_disease` | ID-disease only | Mixed OOD | Near-OOD sensitivity (NF removed) |
| `auroc_nf_vs_pure` | NF only | Pure OOD | Diagnostic floor: healthy-vs-anything |
| `auroc_full` | ID-disease + NF | Pure + Mixed OOD | Full population measurement |

## Documentation

- [`docs/CONCEPTS.md`](docs/CONCEPTS.md) — glossary: ID/OOD, OSR, semantic vs covariate shift, near vs far OOD, multi-class vs multi-label.
- [`docs/USAGE.md`](docs/USAGE.md) — "which metric should I use?" decision tree.
- [`docs/PITFALLS.md`](docs/PITFALLS.md) — the eight most common mistakes, with bad-vs-good code side by side.
- [`docs/EXAMPLES.md`](docs/EXAMPLES.md) — end-to-end runnable examples
  including the full publication metric panel, DeLong comparison, and
  seed aggregation.
- [`REFERENCES.md`](REFERENCES.md) — bibliographic source for every metric.
- [`CHANGELOG.md`](CHANGELOG.md) — version history.
- [`CITATION.cff`](CITATION.cff) — citation metadata.

## Testing

```bash
pytest tests/ -v
```

Each metric is verified against a first-principles brute-force reference;
the test suite covers numerical equivalence, edge cases (empty class,
single-value scores), and known properties (DeLong z=0 on identical inputs,
ECE=0.9 on overconfident-wrong, etc.).

## Citation

If `osr-metrics` is useful in your research, please cite it:

```bibtex
@software{osr_metrics,
  author  = {Hoang Xuan Truong},
  title   = {osr-metrics: Open-Set Recognition and OOD-Detection Metrics for ML Research},
  url     = {https://github.com/hxtruong6/osr-metrics},
  year    = {2026}
}
```

Machine-readable metadata is in [`CITATION.cff`](CITATION.cff). When citing a
specific version, append `version = {X.Y.Z}` and reference the matching
[GitHub Release](https://github.com/hxtruong6/osr-metrics/releases).

## License

MIT.

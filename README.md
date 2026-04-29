# osr-metrics

[![PyPI version](https://img.shields.io/pypi/v/osr-metrics.svg)](https://pypi.org/project/osr-metrics/)
[![Python versions](https://img.shields.io/pypi/pyversions/osr-metrics.svg)](https://pypi.org/project/osr-metrics/)
[![License: MIT](https://img.shields.io/pypi/l/osr-metrics.svg)](https://github.com/hxtruong6/osr-metrics/blob/main/LICENSE)
[![CI](https://github.com/hxtruong6/osr-metrics/actions/workflows/ci.yml/badge.svg)](https://github.com/hxtruong6/osr-metrics/actions/workflows/ci.yml)

Open-Set Recognition (OSR) and OOD-detection metrics for machine-learning research.

A small, framework-agnostic Python library that bundles the metrics needed
for credible OSR / OOD-detection publications, with consistent score-direction
conventions and first-principles-verified formulas.

## What's inside

| Group | Metrics |
|---|---|
| OOD detection | `auroc`, `fpr_at_tpr`, `fpr_at_95tpr`, `aupr_in`, `aupr_out` |
| Open-Set Recognition | `compute_aoscr` (canonical Dhamija/Vaze), `oscr_curve`, `compute_nf_rejection_at_tpr` |
| Multi-label classification | `macro_auprc`, `macro_auprc_id_labels`, `macro_f1_with_thresholds`, `per_label_auprc`, `f1_per_label` |
| Four-class OSR partitioning | `build_fourclass_masks`, `compute_fourclass_metrics`, `partition_ood_by_purity` |
| Calibration | `expected_calibration_error`, `brier_score` |
| Statistical comparison | `delong_test` (O(n log n) rank-based), `bootstrap_ci` (with optional stratification) |

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
| `compute_nf_rejection_at_tpr` | ❌ | ✅ | — | ✅ ² | — | — |
| `partition_ood_by_purity` | ❌ | ✅ | — | ✅ ² | — | — |
| `build_fourclass_masks` / `compute_fourclass_metrics` | ❌ | ✅ | — | ✅ ² | — | — |
| `macro_auprc` / `macro_auprc_id_labels` | ❌ ³ | ✅ | — | — | — | — |
| `per_label_auprc` / `f1_per_label` | ❌ ³ | ✅ | — | — | — | — |
| `macro_f1_with_thresholds` | ❌ ³ | ✅ | — | — | — | — |
| `expected_calibration_error` | ⚠ ⁴ | ✅ | — | — | ✅ | — |
| `brier_score` | ⚠ ⁴ | ✅ | — | — | ✅ | — |
| `delong_test` | ✅ | ✅ | ✅ | ✅ | — | ✅ |
| `bootstrap_ci` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

¹ Multi-label OSCR/AOSCR: pass an exact-match indicator
(`1` if all labels predicted correctly, else `0`) as `class_predictions`
with `true_classes=ones(N)`. See `compute_aoscr` docstring.

² Clinical / multi-label OSR helpers — depend on a per-sample
"No Finding" (all-zero label vector) indicator that has no analogue in
multi-class single-label settings.

³ Multi-class single-label closed-set classification — use
`sklearn.metrics.accuracy_score` and
`sklearn.metrics.f1_score(..., average='macro')` directly. A native
multi-class wrapper is on the roadmap.

⁴ Multi-class softmax calibration (Guo 2017 form) is **not yet** the
form implemented here. Current functions flatten across (sample, label).
For multi-class softmax, use `sklearn.calibration.calibration_curve` or
`torchmetrics.CalibrationError` until the multi-class overload lands.

## Score-direction convention

For every OOD/novelty metric in this library, **higher score = more OOD**.
ID-positive metrics (`aupr_in`) handle the sign flip internally so you don't
have to.

## Install

```bash
pip install osr-metrics
```

Requires Python 3.10+, `numpy`, `scikit-learn`, `scipy`.

### Development install

```bash
git clone https://github.com/hxtruong6/osr-metrics.git
cd osr-metrics
pip install -e .[dev]
```

## Quick start

```python
import numpy as np
from osr_metrics import auroc, fpr_at_95tpr, compute_aoscr, expected_calibration_error

# OOD detection
scores = np.random.randn(1000)          # higher = more OOD
labels = np.random.randint(0, 2, 1000)  # 1 = OOD, 0 = ID
print("AUROC:", auroc(scores, labels))
print("FPR@95TPR:", fpr_at_95tpr(scores, labels))

# Open-Set Classification Rate (joint classify+reject)
cls_pred = np.random.randint(0, 5, 1000)
cls_true = np.random.randint(0, 5, 1000)
print("AOSCR:", compute_aoscr(scores, labels, cls_pred, cls_true))

# Calibration
probs = np.random.uniform(0, 1, (1000, 14))
multi_labels = (np.random.uniform(0, 1, (1000, 14)) < probs).astype(int)
print("ECE:", expected_calibration_error(probs, multi_labels))
```

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

## Why another metrics library?

Most OOD/OSR libraries (PyTorch-OOD, OpenOOD) couple metrics with detection
methods, datasets, and a heavy framework. `osr-metrics` is just the metrics —
useful when you want to compute AOSCR or DeLong on cached scores from any
pipeline, regardless of how those scores were produced.

## Documentation

- [`docs/USAGE.md`](docs/USAGE.md) — "which metric should I use?" decision tree.
- [`docs/EXAMPLES.md`](docs/EXAMPLES.md) — end-to-end runnable examples
  including the full publication metric panel, DeLong comparison, and
  seed aggregation.
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

## License

MIT.

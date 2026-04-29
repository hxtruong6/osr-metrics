# osr-metrics

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

## Score-direction convention

For every OOD/novelty metric in this library, **higher score = more OOD**.
ID-positive metrics (`aupr_in`) handle the sign flip internally so you don't
have to.

## Install

```bash
pip install -e .
# or, with dev tools:
pip install -e .[dev]
```

Requires Python 3.10+, `numpy`, `scikit-learn`, `scipy`.

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

# Which Metric Should I Use?

A decision guide. Find the row that describes your situation; the right
column tells you which `osr_metrics` functions to call.

## Conventions (read these first)

- **Score direction**: every OOD/novelty score in this library follows
  **higher = more OOD**. ID-positive metrics (`aupr_in`) handle the sign
  internally; you don't need to flip the sign yourself.
- **Multi-label probabilities**: use **sigmoid (per-label)**, not softmax.
- **Pairing comparisons**: when comparing methods on the same data, always
  use the same seed split.

## Decision tree

### A. I have OOD scores and binary OOD labels — how good is my detector?

| Question | Function |
|---|---|
| Threshold-free separability — single number for the headline | `auroc(scores, labels)` |
| Operating-point cost — false alarms when 95% of OOD is caught | `fpr_at_95tpr(scores, labels)` or `fpr_at_tpr(scores, labels, target_tpr=0.95)` |
| Asymmetric PR — quality of OOD-suspicious ranking | `aupr_out(scores, labels)` |
| Asymmetric PR — quality of ID-confident ranking | `aupr_in(scores, labels)` |

### B. I have multi-label predictions — how good is my closed-set classifier?

| Question | Function |
|---|---|
| Threshold-free, equal weight per label | `macro_auprc(probs, labels)` |
| Same, but exclude held-out labels (avoid all-zero AUPRC penalty) | `macro_auprc_id_labels(probs, labels, label_names, held_out_labels)` |
| Operating-point F1 with a single threshold per label | `macro_f1_with_thresholds(probs, labels, thresholds)` |
| Per-label diagnostics | `per_label_auprc(probs, labels)` and `f1_per_label(preds, labels)` |

### C. I have held-out labels — how do I report Open-Set Recognition results?

The headline number is **AOSCR**: it jointly evaluates classification
correctness on accepted ID samples vs OOD-rejection rate.

```python
from osr_metrics import compute_aoscr
aoscr = compute_aoscr(novelty_scores, ood_labels, class_predictions, true_classes)
```

For a richer view, get all five AUROC pairings in one call:

```python
from osr_metrics import compute_fourclass_metrics
m = compute_fourclass_metrics(ood_scores, label_vecs, label_names, held_out_labels)
```

Then pick the keys you want to report:

| Key | Report this when |
|---|---|
| `auroc_full` | You want the single headline OOD AUROC over the full population |
| `auroc_pure` | You want the cleanest "can the score detect held-out content?" number |
| `auroc_mixed` | You want to show how mixed-OOD (known + unknown co-present) is detected |
| `auroc_mixed_vs_id_disease` | You want near-OOD sensitivity with NF removed from negatives |
| `auroc_nf_vs_pure` | Diagnostic: floor case (healthy vs unknown disease) |
| `auroc_disease_only` | Backward-compatible legacy: ID-disease vs all OOD, NF excluded |

### D. I want to know if my probabilities are calibrated

| Question | Function |
|---|---|
| Are P=0.7 predictions hitting 70% empirical rate? | `expected_calibration_error(probs, labels)` |
| Combined calibration + sharpness (strictly proper rule) | `brier_score(probs, labels)` |

### E. I want to compare two methods statistically

| Situation | Function | Notes |
|---|---|---|
| Same seed, same data, two different methods | `delong_test(scores_a, scores_b, labels)` | Paired AUROC test (DeLong 1988). Returns (z, p). |
| One method, want a CI on its AUROC | `bootstrap_ci(scores, labels, auroc, n_bootstrap=1000)` | Use `stratify=True` if classes are very imbalanced. |
| Multiple seeds, want the mean ± std | `np.std([auroc per seed], ddof=1)` | One line, no library function needed. |

### F. I want a healthy-patient false-alarm number

```python
from osr_metrics import compute_nf_rejection_at_tpr
rej = compute_nf_rejection_at_tpr(scores, ood_labels, nf_labels, tpr=0.95)
```
Calibrates the threshold on ID-disease at TPR=0.95, then reports the rate at
which NF (no-finding, healthy) images are rejected by that threshold. Higher
= better (healthy patients flagged for review rather than auto-classified).

## Common mistakes

1. **Forgetting score direction**: if your model returns a *confidence* score
   (higher = more confident in ID), pass `-confidence` to all OOD functions,
   or wrap with a small adapter.
2. **Using `auroc_pure` as the deployment number**: it is a *diagnostic
   upper bound* (mixed OOD removed from negatives). Report `auroc_full` or
   `aoscr` as the headline; report `auroc_pure` alongside, never alone.
3. **Bootstrapping without `stratify=True` on imbalanced data**: small OOD
   classes can produce all-ID replicates → NaN AUROC → biased CI. Pass
   `stratify=True`.
4. **Mixing different seed splits across DeLong tests**: DeLong is paired —
   it requires `scores_a` and `scores_b` evaluated on the *same* samples.

## Putting it together: the full publication panel

The metric panel for OSR papers (see `evaluation-metrics-panel.md` in the
parent project repo for the full table, ranked by must/should/nice):

```python
from osr_metrics import (
    auroc, fpr_at_95tpr, aupr_in, aupr_out,
    macro_auprc, macro_f1_with_thresholds,
    compute_aoscr, compute_fourclass_metrics, compute_nf_rejection_at_tpr,
    expected_calibration_error, brier_score,
    delong_test, bootstrap_ci,
)
```

That covers all 19 rows of the panel.

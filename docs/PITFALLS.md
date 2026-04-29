# Common Pitfalls

The mistakes this library is most often hit by, with the bad code and
the corrected code side by side.

## 1. Inverted score direction

**Symptom**: `auroc(...)` returns a value consistently **below 0.5**
(often around 1 minus what you expected).

**Why**: every OOD/novelty score in this library follows
*higher = more OOD*. If you pass a **confidence** score (higher = more
ID, e.g. softmax max-prob, max-logit, MSP), AUROC flips.

```python
# ❌ Wrong: max-prob is "higher = more ID"
max_prob = softmax(logits).max(axis=1)
auroc(max_prob, ood_labels)  # ~ 1 - true_AUROC

# ✅ Right: flip once at the boundary
from osr_metrics import as_ood_scores
scores = as_ood_scores(max_prob, direction="confidence")
auroc(scores, ood_labels)
```

You can also call `warn_if_inverted_scores(scores, ood_labels)` to get
a warning when `AUROC < 0.5`.

## 2. AUPR-In vs AUPR-Out confusion

**Symptom**: AUPR numbers look unexpectedly high or low after switching
methods.

**Why**: `aupr_in` and `aupr_out` measure different things — one
treats ID as the positive class, the other treats OOD. They are
**not** complementary.

```python
# Reporting one without the other is fine, but be explicit about which.
aupr_out_v = aupr_out(scores, labels)  # OOD as positive
aupr_in_v  = aupr_in(scores, labels)   # ID as positive (sign handled internally)
```

Never report a single "AUPR" without saying which side. The OSR / OOD
literature defaults to **AUPR-Out** as the headline; AUPR-In is the
ID-confidence ranking diagnostic.

## 3. AOSCR on multi-label without an exact-match indicator

**Symptom**: ` compute_aoscr` runs but the value is meaningless or zero.

**Why**: `compute_aoscr` expects a **single integer** prediction per
sample. For multi-label, you must reduce predictions to a binary
"all labels correct" indicator first.

```python
import numpy as np
from osr_metrics import compute_aoscr

# ❌ Wrong: passing the full multi-hot matrix
compute_aoscr(scores, ood_labels, preds_NK, labels_NK)  # shape mismatch / nonsense

# ✅ Right: exact-match reduction
exact_match = (preds_NK == labels_NK).all(axis=1).astype(int)
compute_aoscr(scores, ood_labels, exact_match, np.ones_like(exact_match))
```

For multi-class (single-label), use `compute_aoscr_multiclass` and pass
either integer predictions or a logits matrix:

```python
from osr_metrics import compute_aoscr_multiclass
compute_aoscr_multiclass(scores, ood_labels, logits_NK, y_N)
```

## 4. Multi-class softmax through the multi-label calibration metric

**Symptom**: ECE values are much smaller than expected for a multi-class
model, because most off-diagonal probabilities sit near zero and dominate
the average.

**Why**: `expected_calibration_error` flattens across (sample, label)
pairs — appropriate for multi-label sigmoid, not for multi-class
softmax. The Guo 2017 reliability diagram uses **top-1 confidence**
versus **top-1 correctness**.

```python
# ❌ Wrong on multi-class softmax
expected_calibration_error(softmax_NK, onehot_NK)

# ✅ Right
from osr_metrics import expected_calibration_error_multiclass
expected_calibration_error_multiclass(softmax_NK, y_N)
```

## 5. Bootstrapping rare-class data without `stratify=True`

**Symptom**: `bootstrap_ci` returns NaN for some replicates, or yields
a CI wider than expected.

**Why**: when one class is rare, plain bootstrap occasionally produces
all-positive or all-negative replicates → AUROC is undefined → NaN.

```python
# ❌ Wrong on imbalanced data
bootstrap_ci(scores, labels, auroc, n_bootstrap=1000)

# ✅ Right: resample positives and negatives separately
bootstrap_ci(scores, labels, auroc, n_bootstrap=1000, stratify=True)
```

## 6. Comparing methods with `delong_test` on different splits

**Symptom**: `delong_test` returns a `z` statistic, but it isn't valid.

**Why**: DeLong is a **paired** test. Both score arrays must be
evaluated on the **same samples** with the **same labels**. Different
seeds, different validation splits, or different sub-samples invalidate
the assumption.

```python
# ❌ Wrong: scores_a and scores_b come from different val splits
delong_test(scores_a, scores_b, labels_a)

# ✅ Right: same data, two methods
delong_test(method_a_scores, method_b_scores, shared_labels)
```

For unpaired comparisons (different splits, different model families
trained independently), use bootstrap CIs instead:

```python
lo_a, mean_a, hi_a = bootstrap_ci(scores_a, labels, auroc, stratify=True)
lo_b, mean_b, hi_b = bootstrap_ci(scores_b, labels, auroc, stratify=True)
# Compare CIs.
```

## 7. Reporting `auroc_pure` as the headline number

**Symptom**: AUROC numbers in your paper look great, but reviewers
notice they don't match the deployment population.

**Why**: `auroc_pure` is a **diagnostic upper bound**: mixed-OOD
samples are excluded from the negatives, so the score sees a cleaner
problem than deployment will. Always pair it with `auroc_full`.

```python
# ❌ Wrong: report only auroc_pure
print(metrics["auroc_pure"])

# ✅ Right: pair with auroc_full
print(f"full = {metrics['auroc_full']:.3f}, "
      f"pure = {metrics['auroc_pure']:.3f}, "
      f"mixed = {metrics['auroc_mixed']:.3f}")
```

## 8. Forgetting that "label" means "OOD label" (not "class label")

**Symptom**: `ValueError: labels must be binary {0, 1}` when calling
`auroc` / `fpr_at_95tpr` / `aupr_in` / `aupr_out`.

**Why**: in the OOD-detection functions, `labels` is the binary
**OOD ground truth** (1 = OOD, 0 = ID), not the closed-set class label.

```python
# ❌ Wrong: passing class IDs
auroc(scores, y_classes)

# ✅ Right: convert held-out classes to an OOD indicator
ood_labels = np.isin(y_classes, held_out_class_ids).astype(int)
auroc(scores, ood_labels)
```

## 9. AURC is not an OOD-detection metric

**Bad:**

```python
# Reporting AURC as if it measured OOD detection performance.
score = msp_logits  # or any uncertainty score
loss = (y_true != y_pred).astype(float)  # closed-set 0/1 loss
aurc_value = aurc(score, loss)
print(f"OOD detection AURC: {aurc_value:.3f}")  # WRONG framing
```

AURC measures how well `score` ranks samples by the *closed-set* `loss` — i.e. "does my uncertainty score also identify misclassified samples?" It does not measure OOD detection.

**Good:**

```python
# Use the right metric for the question being asked.
auroc_ood = auroc(score, ood_labels)            # OOD detection
aurc_value = aurc(score, loss)                  # selective classification
```

If you want a single number that captures both — joint OSR — use `compute_aoscr` instead.

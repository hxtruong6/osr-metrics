# End-to-End Examples

Worked examples mirroring how you'd actually use this library: load
cached predictions, compute the publication metric panel, run
statistical comparison.

The examples below use synthetic data so they run as-is.

- **Example 1** — multi-class (single-label) OSR (CIFAR-style)
- **Example 2** — multi-label OSR (chest-X-ray-style), full panel
- **Example 3** — pairwise DeLong comparison
- **Example 4** — seed aggregation
- **Example 5** — loading from a real `scores.json`

## Example 1: multi-class (single-label) OSR

For a CIFAR-style setup: K known classes (held-in) + a held-out class
treated as OOD. The library's `compute_panel` infers the multi-class
setting from input shapes and computes everything in one call.

```python
import numpy as np
from osr_metrics import compute_panel, as_ood_scores


def make_multiclass_run(seed: int = 0):
    """Synthetic multi-class OSR run.

    Returns:
        scores:      [N] novelty scores, higher = more OOD
        ood_labels:  [N] binary OOD ground truth (1 = held-out class)
        logits:      [N, K] closed-set logits (over K known classes)
        y:           [N] integer ground-truth class IDs in [0, K)
    """
    rng = np.random.default_rng(seed)
    N, K = 1000, 5
    held_out_class = K  # one held-out class with id == K

    # Sample 30% OOD
    is_ood = rng.uniform(0, 1, N) < 0.3
    y_full = np.where(is_ood, held_out_class, rng.integers(0, K, N))
    ood_labels = is_ood.astype(int)

    # Logits: high on the true class for ID; flat for OOD
    logits = rng.normal(0, 1, (N, K))
    id_mask = ~is_ood
    logits[id_mask, y_full[id_mask]] += 3.0  # ID samples get a peak

    # OOD score = -max-logit (lower max-logit → more OOD)
    confidence = logits.max(axis=1)
    scores = as_ood_scores(confidence, direction="confidence")

    # For the closed-set head, we report only on ID samples
    y = np.where(is_ood, 0, y_full)  # OOD labels masked; AOSCR uses ood_labels
    return scores, ood_labels, logits, y


scores, ood_labels, logits, y = make_multiclass_run(seed=0)

# Softmax for calibration; argmax for the closed-set head.
def softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

panel = compute_panel(
    scores,
    ood_labels,
    probs=softmax(logits),  # accepts logits or softmax for argmax;
                            # calibration metrics need true probs in [0, 1]
    y=y,
)

for k, v in sorted(panel.items()):
    if isinstance(v, float):
        print(f"  {k:30s} = {v:.4f}")
    else:
        print(f"  {k:30s} = {v}")
```

The output includes `auroc`, `fpr_at_95tpr`, `aupr_in/out`, `aoscr`,
`top1_accuracy`, `macro_f1`, `balanced_accuracy`, `ece`, `brier` — the
full multi-class headline panel from one call.

## Example 2: multi-label OSR — full publication panel

```python
import json
import numpy as np
from osr_metrics import (
    auroc, fpr_at_95tpr, aupr_in, aupr_out,
    macro_auprc, macro_f1_with_thresholds,
    compute_aoscr, compute_fourclass_metrics, compute_nf_rejection_at_tpr,
    expected_calibration_error, brier_score,
    bootstrap_ci,
)


def make_synthetic_run(seed: int = 0):
    """Stand-in for loading scores.json from a real experiment.

    Returns a dict with the four arrays every metric below needs:
      - ood_scores: per-image novelty scores, higher = more OOD
      - probs: per-image per-label sigmoid probabilities (multi-label classification)
      - label_vecs: per-image binary multi-hot ground truth
      - ood_labels: per-image binary OOD ground truth (1 = OOD)
    """
    rng = np.random.default_rng(seed)
    N = 1000
    K = 5
    label_names = ["A", "B", "C", "D", "E"]
    held_out = ["D", "E"]

    # Random multi-label ground truth.
    label_vecs = (rng.uniform(0, 1, (N, K)) < 0.15).astype(int)
    held_idx = [label_names.index(l) for l in held_out]
    ood_labels = (label_vecs[:, held_idx].sum(axis=1) > 0).astype(int)

    # Probabilities loosely tracking the ground truth.
    probs = np.clip(label_vecs * 0.6 + rng.uniform(0, 0.4, (N, K)), 0, 1)

    # OOD score: derived as 1 - max(prob over known labels), with noise.
    known_idx = [k for k in range(K) if k not in held_idx]
    confidence = probs[:, known_idx].max(axis=1)
    ood_scores = (1 - confidence) + rng.normal(0, 0.1, N) + ood_labels * 0.3

    return {
        "ood_scores": ood_scores,
        "probs": probs,
        "label_vecs": label_vecs,
        "ood_labels": ood_labels,
        "label_names": label_names,
        "held_out_labels": held_out,
    }


def compute_panel_manual(run: dict) -> dict:
    """Compute the full publication metric panel from one run.

    The same result can be obtained in a single call with
    ``osr_metrics.compute_panel`` (see the bottom of this example);
    this manual version shows what each function contributes so you can
    pick a subset.
    """
    s = run["ood_scores"]
    y_ood = run["ood_labels"]
    p = run["probs"]
    L = run["label_vecs"]
    names = run["label_names"]
    held = run["held_out_labels"]

    panel = {}

    # --- A. OOD detection (rows 1-2, 10-11 of the panel) ---
    panel["auroc"] = auroc(s, y_ood)
    panel["fpr95"] = fpr_at_95tpr(s, y_ood)
    panel["aupr_out"] = aupr_out(s, y_ood)
    panel["aupr_in"] = aupr_in(s, y_ood)

    # --- B. Closed-set classification (rows 4-5) ---
    panel["macro_auprc"] = macro_auprc(p, L)
    # For macro-F1 you need per-label thresholds tuned on val; here we use 0.5.
    panel["macro_f1_at_0.5"] = macro_f1_with_thresholds(p, L, [0.5] * p.shape[1])

    # --- C. Open-Set Recognition (rows 3, 6, 12, 13, 18) ---
    cls_correct = (p.round() == L).all(axis=1).astype(int)
    panel["aoscr"] = compute_aoscr(s, y_ood, cls_correct, np.ones_like(cls_correct))

    fc = compute_fourclass_metrics(s, L, names, held)
    panel["auroc_pure"] = fc["auroc_pure"]
    panel["auroc_mixed"] = fc["auroc_mixed"]
    panel["auroc_mixed_vs_id_disease"] = fc["auroc_mixed_vs_id_disease"]
    panel["auroc_nf_vs_pure"] = fc["auroc_nf_vs_pure"]
    panel["counts_per_class"] = fc["counts"]

    # --- D. Calibration (rows 15, 19) ---
    panel["ece"] = expected_calibration_error(p, L)
    panel["brier"] = brier_score(p, L)

    # --- E. NF rejection (row 16) ---
    nf_labels = (L.sum(axis=1) == 0).astype(int)
    panel["nf_rejection_at_95tpr"] = compute_nf_rejection_at_tpr(s, y_ood, nf_labels)

    # --- F. Bootstrap CI on the headline (row 8) ---
    lo, mean, hi = bootstrap_ci(s, y_ood, auroc, n_bootstrap=500, stratify=True)
    panel["auroc_bootstrap_mean"] = mean
    panel["auroc_bootstrap_ci_low"] = lo
    panel["auroc_bootstrap_ci_high"] = hi

    return panel


if __name__ == "__main__":
    run = make_synthetic_run(seed=0)
    panel = compute_panel_manual(run)

    # Pretty-print, separating the dict from the float keys.
    counts = panel.pop("counts_per_class")
    for k, v in sorted(panel.items()):
        print(f"  {k:35s} = {v:.4f}")
    print(f"\n  per-class counts: {counts}")
```

### Same panel via one call

```python
from osr_metrics import compute_panel

run = make_synthetic_run(seed=0)
panel = compute_panel(
    run["ood_scores"],
    run["ood_labels"],
    preds=run["probs"].round().astype(int),
    probs=run["probs"],
    label_vecs=run["label_vecs"],
    label_names=run["label_names"],
    held_out_labels=run["held_out_labels"],
    setting="multilabel",
)
```

Expected output (synthetic):

```
  aoscr                               = 0.5xxx
  auroc                               = 0.7xxx
  auroc_bootstrap_ci_high             = 0.7xxx
  auroc_bootstrap_ci_low              = 0.6xxx
  auroc_bootstrap_mean                = 0.7xxx
  auroc_mixed                         = 0.6xxx
  auroc_mixed_vs_id_disease           = 0.5xxx
  auroc_nf_vs_pure                    = 0.7xxx
  auroc_pure                          = 0.7xxx
  aupr_in                             = 0.7xxx
  aupr_out                            = 0.6xxx
  brier                               = 0.1xxx
  ece                                 = 0.0xxx
  fpr95                               = 0.7xxx
  macro_auprc                         = 0.5xxx
  macro_f1_at_0.5                     = 0.4xxx
  nf_rejection_at_95tpr               = 0.0xxx

  per-class counts: {'id_disease': ..., 'no_finding': ..., 'pure_ood': ..., 'mixed_ood': ...}
```

## Example 3: pairwise method comparison with DeLong

```python
import numpy as np
from osr_metrics import auroc, delong_test

rng = np.random.default_rng(0)
N = 1000
labels = rng.integers(0, 2, N)

# Method A: weak detector
scores_a = rng.normal(loc=labels * 0.3, scale=1.0)
# Method B: strong detector
scores_b = rng.normal(loc=labels * 0.8, scale=1.0)

print(f"AUROC A: {auroc(scores_a, labels):.4f}")
print(f"AUROC B: {auroc(scores_b, labels):.4f}")

z, p = delong_test(scores_b, scores_a, labels)
print(f"DeLong: z = {z:.3f}, p = {p:.2e}")
print(f"Significant difference (p < 0.05): {p < 0.05}")
```

DeLong is **paired** — `scores_a` and `scores_b` must come from the same set
of samples (`labels`). Use this when comparing methods on a fixed test split.

For seed-averaged comparisons across multiple seeds (different random splits
per seed), use a paired t-test on the per-seed AUROC values instead.

## Example 4: aggregating across seeds

```python
import numpy as np
from osr_metrics import auroc

per_seed_aurocs = []
for seed in range(5):
    run = make_synthetic_run(seed=seed)
    per_seed_aurocs.append(auroc(run["ood_scores"], run["ood_labels"]))

per_seed = np.array(per_seed_aurocs)
print(f"AUROC mean ± std: {per_seed.mean():.4f} ± {per_seed.std(ddof=1):.4f}")
```

The std across seeds is the **reproducibility** number that goes alongside
the mean in any paper table.

## Example 5: loading from a real scores.json

If your run produces a JSON file with the same keys as `make_synthetic_run`
returns, swap it in:

```python
import json
import numpy as np

with open("path/to/scores.json") as f:
    raw = json.load(f)

run = {
    "ood_scores": np.array(raw["ood_scores"]),
    "probs": np.array(raw["probs"]),
    "label_vecs": np.array(raw["label_vecs"]),
    "ood_labels": np.array(raw["ood_labels"]),
    "label_names": raw["label_names"],
    "held_out_labels": raw["held_out_labels"],
}

panel = compute_panel_manual(run)
```

That's it — no GPU required, no model loaded; the metrics work on cached
predictions only.

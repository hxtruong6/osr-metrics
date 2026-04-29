# Concepts

A one-page glossary of the terms used throughout this library and its
documentation. If you came here because you were unsure which metric
applies to your problem, read this first, then jump to the
[capability matrix](../README.md#capability-matrix--which-function-for-which-setting)
in the README.

## ID vs OOD

- **In-distribution (ID)**: a sample drawn from the same distribution
  as the training data — known label, known input domain.
- **Out-of-distribution (OOD)**: a sample whose label or input domain
  differs from the training data. *What kind* of difference matters
  (see [shift types](#shift-types) below).

## Detection vs recognition

- **OOD detection**: a binary problem. *"Is this sample ID or OOD?"*
  No need to classify ID samples — just produce a per-sample novelty
  score and threshold it.
- **Open-Set Recognition (OSR)**: a joint problem. *"Classify if known,
  reject if unknown."* Requires both a closed-set prediction (which of
  the K known classes?) and a rejection score. The headline metric is
  **AOSCR** — the area under the curve that trades correct
  classification on accepted ID against false-acceptance of OOD.
- **Generalized OOD** (Yang et al. 2022, OpenOOD): the umbrella over
  OSR, OOD detection, novelty detection, anomaly detection, and
  covariate-shift detection.

OSR is a **strict superset** of OOD detection: every OSR system
contains an OOD detector, but adds the closed-set classification head.

## Shift types

| Shift | What changes | Same-task example | This library? |
|---|---|---|---|
| **Semantic shift** | Label space (new classes appear) | New disease at test time | ✅ In scope |
| **Covariate shift** | Input domain (sensor / style / hospital) | Same diseases, different scanner | ❌ Out of scope |
| **Concept drift** | P(y \| x) changes over time | Label semantics evolve | ❌ Out of scope |

This library targets the **semantic-shift** setting. If your concern
is a different scanner or noise level, you want a domain-generalization
toolkit, not OSR metrics.

## Near-OOD vs far-OOD

A **soft** distinction within semantic shift. Both are OOD; they differ
in how easy they are to detect.

- **Far-OOD**: semantically very different from ID
  (e.g. CIFAR-10 vs SVHN, natural images vs MNIST). Easier; AUROC
  typically high.
- **Near-OOD ("hard OOD")**: semantically close to ID
  (e.g. CIFAR-10 vs CIFAR-100, one disease vs a related disease in the
  same modality). Harder; AUROC typically much lower.

In this library:
- `auroc_pure` (from `compute_fourclass_metrics`) approximates **far**
  separability — held-out content with no known-class co-occurrence.
- `auroc_mixed_vs_id_disease` approximates **near** separability —
  held-out content co-occurs with known disease.

## Multi-class vs multi-label

| | Multi-class (single-label) | Multi-label |
|---|---|---|
| Ground truth | One integer per sample | Multi-hot vector per sample |
| Output head | Softmax | Sigmoid (per-label) |
| Closed-set metric | Accuracy / macro-F1 | Per-label AUPRC / macro-F1 |
| OSR metric | `compute_aoscr_multiclass` | `compute_aoscr` (exact-match) + `compute_fourclass_metrics` |
| Calibration metric | `*_multiclass` overloads | `expected_calibration_error` / `brier_score` |

The four-class partition (ID-disease / No-Finding / Pure-OOD /
Mixed-OOD) only exists for **multi-label** — every sample falls into
exactly one class in the multi-class case.

## Score-direction convention

For every OOD/novelty score in this library, **higher = more OOD**.
ID-positive metrics (`aupr_in`) handle the sign flip internally. If
your model returns a confidence score (higher = more ID, e.g. softmax
max-prob), wrap it once at the boundary:

```python
from osr_metrics import as_ood_scores, auroc
scores = as_ood_scores(softmax_max_prob, direction="confidence")
auroc(scores, ood_labels)
```

If you forget, AUROC will come out as `1 - AUROC` — consistently far
below 0.5. The library will warn if you call `warn_if_inverted_scores`
explicitly; or you can spot it by eye.

## What this library does not cover

- **Regression OOD** — AUROC-style detection works on any score, but
  the OSR-specific metrics (AOSCR, four-class partition) assume a
  classification task.
- **Density estimation** — out of scope; use a likelihood toolkit.
- **Continual / open-world learning** — needs forward-/backward-transfer
  metrics that this library does not implement.
- **Detection methods** — this library provides only metrics. Pair it
  with PyTorch-OOD, OpenOOD, or your own scoring pipeline.

## Further reading

The full bibliographic source list is in
[`REFERENCES.md`](../REFERENCES.md). The most important pointers for
the conceptual framing above:

- **Yang et al. 2022, OpenOOD** — the generalized OOD taxonomy used
  here (semantic vs covariate, near vs far).
- **Vaze et al. 2022** — the canonical OSR formulation and AOSCR.
- **Dhamija et al. 2018** — the OSCR curve.
- **Hendrycks & Gimpel 2017** — the AUROC + FPR@95TPR convention for
  OOD detection that the rest of the field built on.

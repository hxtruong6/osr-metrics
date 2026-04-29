"""Open-Set Recognition (OSR) metrics: joint classify + reject.

**Scope:**

- ``compute_aoscr`` / ``oscr_curve`` (in ``ood.py``): multi-class
  (single-label) **and** multi-label. For multi-label, callers reduce
  predictions to an exact-match indicator (see ``compute_aoscr``
  docstring). The canonical setting is multi-class single-label
  (Dhamija 2018, Vaze 2022).
- ``compute_nf_rejection_at_tpr``: multi-label only — depends on a
  per-sample "No Finding" (all-zero label vector) indicator.

Two metrics are implemented here:

1. ``compute_aoscr`` -- Area under the Open-Set Classification Rate (OSCR)
   curve (Dhamija et al. 2018; popularised for OSR by Vaze et al. 2022,
   "Open-Set Recognition: A Good Closed-Set Classifier Is All You Need").
   The curve plots the *Correct Classification Rate* on accepted ID samples
   against the *False Positive Rate* (fraction of OOD samples accepted) as a
   threshold on the novelty score sweeps from low to high.  AOSCR is the area
   under this curve, in [0, 1] (higher is better).

2. ``compute_nf_rejection_at_tpr`` -- Among in-distribution samples whose
   ground-truth label vector is "No Finding" (all zeros), what fraction are
   rejected when the OOD threshold is calibrated to keep TPR=0.95 on real
   ID-disease samples?  This quantifies the rate at which a healthy patient
   is sent to a human reviewer because the model is uncertain.
"""
from __future__ import annotations

import numpy as np


def compute_aoscr(
    scores: np.ndarray,
    ood_labels: np.ndarray,
    class_predictions: np.ndarray,
    true_classes: np.ndarray,
    n_thresholds: int = 1000,
) -> float:
    """Area under the Open-Set Classification Rate (OSCR) curve.

    | Applies to                                        | Task                  |
    |---------------------------------------------------|-----------------------|
    | Multi-class (single-label); multi-label via flag  | OSR (classify+reject) |

    For multi-class, pass integer class IDs directly to
    ``class_predictions`` and ``true_classes``. For multi-label, pass an
    exact-match indicator (1 = all labels predicted correctly, else 0)
    as ``class_predictions`` paired with ``true_classes`` of all-ones.

    The OSCR curve sweeps a threshold ``tau`` over the novelty score:

      * a sample is **rejected** if ``score > tau`` (treated as OOD);
      * an ID sample contributes to the *Correct Classification Rate* (CCR)
        only if it is **accepted** AND classified correctly;
      * an OOD sample contributes to the *False Positive Rate* (FPR) if it
        is **accepted** (i.e. erroneously kept as ID).

    AOSCR is the area under the (FPR, CCR) curve, integrated by the
    trapezoidal rule.  Random performance is ~0.5; perfect performance is 1.

    Args:
        scores: Novelty scores, shape ``[N]``.  Higher = more OOD.
        ood_labels: Binary OOD ground truth, shape ``[N]``.  ``1`` = OOD,
            ``0`` = in-distribution.
        class_predictions: Per-sample predicted closed-set class, shape
            ``[N]``.  For multi-label, callers may pass a flag (``1`` if all
            label predictions are correct, else ``0``) and pair it with
            ``true_classes`` of all-ones; equivalently, callers may collapse
            multi-label predictions to a single "exact-match" indicator.
        true_classes: Per-sample ground-truth closed-set class, shape ``[N]``.
        n_thresholds: Number of threshold steps for the curve.

    Returns:
        AOSCR in ``[0, 1]``.  Higher is better.  Returns ``0.0`` if either
        the ID or OOD class is empty.
    """
    scores = np.asarray(scores, dtype=float)
    ood_labels = np.asarray(ood_labels).astype(int)
    class_predictions = np.asarray(class_predictions)
    true_classes = np.asarray(true_classes)

    id_mask = ood_labels == 0
    ood_mask = ood_labels == 1
    n_id = int(id_mask.sum())
    n_ood = int(ood_mask.sum())

    if n_id == 0 or n_ood == 0:
        return 0.0

    cls_correct = (class_predictions == true_classes).astype(int)

    # Build threshold grid over the full score range; pad ends so the curve
    # spans FPR = 0 (tau = +inf -> accept nothing) and FPR = 1 (tau = -inf -> accept all).
    lo, hi = float(scores.min()), float(scores.max())
    if lo == hi:
        # Degenerate: every sample has the same score.  Curve is a single point.
        return 0.0
    thresholds = np.linspace(lo, hi, n_thresholds)

    fpr_list = np.empty(n_thresholds, dtype=float)
    ccr_list = np.empty(n_thresholds, dtype=float)

    id_scores = scores[id_mask]
    id_correct = cls_correct[id_mask]
    ood_scores = scores[ood_mask]

    for i, tau in enumerate(thresholds):
        # Accepted = score <= tau.
        id_accepted = id_scores <= tau
        ood_accepted = ood_scores <= tau

        ccr_list[i] = float((id_accepted & (id_correct == 1)).sum()) / n_id
        fpr_list[i] = float(ood_accepted.sum()) / n_ood

    # Sort by FPR ascending so np.trapezoid produces a positive area.
    order = np.argsort(fpr_list)
    fpr_sorted = fpr_list[order]
    ccr_sorted = ccr_list[order]

    _trapz = getattr(np, "trapezoid", None) or np.trapz
    return float(_trapz(ccr_sorted, fpr_sorted))


def compute_nf_rejection_at_tpr(
    scores: np.ndarray,
    ood_labels: np.ndarray,
    nf_labels: np.ndarray,
    tpr: float = 0.95,
) -> float:
    """Fraction of No-Finding samples rejected at a fixed ID-disease TPR.

    | Applies to   | Task                       |
    |--------------|----------------------------|
    | Multi-label  | Clinical OSR (NF handling) |

    Multi-label only: requires per-sample "No Finding" indicator
    (all-zero label vector). Has no analogue in multi-class single-label
    settings.

    Calibration protocol:

      1. Restrict to the **real ID-disease** subset (``ood_labels == 0`` and
         ``nf_labels == 0``).  These are the samples whose retention defines
         "TPR" in deployment.
      2. Find the largest threshold ``tau`` such that the fraction of
         ID-disease samples with ``score <= tau`` is at least ``tpr``.
      3. Apply that threshold to the **No-Finding** subset and return the
         fraction whose score exceeds ``tau`` (i.e. *rejected*).

    Higher is better: a healthy patient should be flagged for review, not
    auto-classified as a known disease.

    Args:
        scores: Per-sample OOD/novelty scores, shape ``[N]``.  Higher = more OOD.
        ood_labels: Binary OOD ground truth, shape ``[N]``.  ``1`` = OOD,
            ``0`` = in-distribution.
        nf_labels: Binary "No Finding" indicator, shape ``[N]``.  ``1`` if the
            sample's label vector is all-zero (healthy), else ``0``.
        tpr: Target true-positive (retention) rate on ID-disease samples.

    Returns:
        Rejection rate on No-Finding samples, in ``[0, 1]``.  Returns
        ``float('nan')`` if either the ID-disease or No-Finding subset is
        empty (cannot calibrate / cannot evaluate).
    """
    scores = np.asarray(scores, dtype=float)
    ood_labels = np.asarray(ood_labels).astype(int)
    nf_labels = np.asarray(nf_labels).astype(int)

    id_disease_mask = (ood_labels == 0) & (nf_labels == 0)
    nf_mask = (ood_labels == 0) & (nf_labels == 1)

    if id_disease_mask.sum() == 0 or nf_mask.sum() == 0:
        return float("nan")

    id_disease_scores = scores[id_disease_mask]
    # Threshold at the ``tpr``-th quantile of ID-disease scores: keep at
    # least ``tpr`` fraction with score <= tau.  np.quantile with the
    # default linear interpolation gives the smallest value v such that
    # P(X <= v) >= tpr (within sampling).
    tau = float(np.quantile(id_disease_scores, tpr))

    nf_scores = scores[nf_mask]
    rejected = float((nf_scores > tau).sum()) / float(nf_mask.sum())
    return rejected

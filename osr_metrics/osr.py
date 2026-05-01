"""Open-Set Recognition (OSR) metrics: joint classify + reject.

**Scope:**

- ``compute_aoscr`` / ``oscr_curve`` (in ``ood.py``): multi-class
  (single-label) **and** multi-label. For multi-label, callers reduce
  predictions to an exact-match indicator (see ``compute_aoscr``
  docstring). The canonical setting is multi-class single-label
  (Dhamija 2018, Vaze 2022).
- ``compute_nf_rejection_at_tpr``: multi-label only — depends on a
  per-sample "No Finding" (all-zero label vector) indicator.
- ``aml_oscr_curve`` / ``compute_aml_oscr``: multi-label only —
  macro-F1 analogue of OSCR for Open-Set Multi-Label Classification.
- ``per_novel_discovery_table``: multi-label only — per-novel-label
  detection rate at a fixed novelty threshold, stratified by co-novel
  cardinality and by pure/mixed-novelty regime.
"""
from __future__ import annotations

import warnings

import numpy as np


def _oscr_curve_points(
    scores: np.ndarray,
    ood_labels: np.ndarray,
    cls_correct: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Exact OSCR curve via sort + cumulative counts.

    Returns ``(fpr, ccr, aoscr)`` with one point per unique score plus
    a ``(0, 0)`` anchor. Tied scores collapse to the end of each run, so
    the curve is invariant to input order.
    """
    scores = np.asarray(scores, dtype=float)
    ood_labels = np.asarray(ood_labels).astype(int)
    cls_correct = np.asarray(cls_correct).astype(int)

    id_mask = ood_labels == 0
    ood_mask = ood_labels == 1
    n_id = int(id_mask.sum())
    n_ood = int(ood_mask.sum())

    if n_id == 0 or n_ood == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 0.0]), 0.0
    if scores.min() == scores.max():
        return np.array([0.0, 1.0]), np.array([0.0, 0.0]), 0.0

    order = np.argsort(scores, kind="stable")
    s_sorted = scores[order]
    is_id_correct = (id_mask & (cls_correct == 1)).astype(np.int64)[order]
    is_ood = ood_mask.astype(np.int64)[order]

    cum_ccr = np.cumsum(is_id_correct) / n_id
    cum_fpr = np.cumsum(is_ood) / n_ood

    n = s_sorted.size
    last_of_run = np.empty(n, dtype=bool)
    last_of_run[:-1] = s_sorted[1:] != s_sorted[:-1]
    last_of_run[-1] = True

    fpr = np.concatenate(([0.0], cum_fpr[last_of_run]))
    ccr = np.concatenate(([0.0], cum_ccr[last_of_run]))

    _trapz = getattr(np, "trapezoid", None) or np.trapz  # type: ignore[attr-defined]
    return fpr, ccr, float(_trapz(ccr, fpr))


def compute_aoscr(
    scores: np.ndarray,
    ood_labels: np.ndarray,
    class_predictions: np.ndarray,
    true_classes: np.ndarray,
    n_thresholds: int | None = None,
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
        n_thresholds: Deprecated and ignored. The implementation is now
            exact at every unique score. Passing it emits a warning.

    Returns:
        AOSCR in ``[0, 1]``.  Higher is better.  Returns ``0.0`` if either
        the ID or OOD class is empty.
    """
    if n_thresholds is not None:
        warnings.warn(
            "`n_thresholds` is deprecated and ignored; remove the argument.",
            DeprecationWarning,
            stacklevel=2,
        )
    scores = np.asarray(scores, dtype=float)
    class_predictions = np.asarray(class_predictions)
    true_classes = np.asarray(true_classes)
    cls_correct = (class_predictions == true_classes).astype(int)
    _, _, aoscr = _oscr_curve_points(scores, ood_labels, cls_correct)
    return aoscr


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


def aml_oscr_curve(
    novelty_scores: np.ndarray,
    ood_labels: np.ndarray,
    probs_known: np.ndarray,
    labels_known: np.ndarray,
    thresholds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Multi-label OSCR curve: macro-F1 vs novelty FPR.

    | Applies to   | Task                                  |
    |--------------|---------------------------------------|
    | Multi-label  | OSR (joint classify+reject, OS-MLC)   |

    Multi-label analogue of the Dhamija 2018 / Vaze 2022 OSCR curve.
    Sweeps a novelty threshold ``tau``. At each ``tau``:

      * a sample is **accepted** iff ``novelty_score <= tau``;
      * **FPR** = fraction of OOD samples (incorrectly) accepted;
      * **macro-F1** is the macro-averaged F1 over ``K`` known labels,
        computed on the **accepted ID subset** with per-label thresholds
        fixed in advance.

    The y-axis is macro-F1 (not multi-class top-1 accuracy as in OSCR),
    making the metric appropriate for multi-label closed-set evaluation
    under open-set rejection. The area under the curve is **AML-OSCR**.

    Args:
        novelty_scores: Per-image novelty scores, shape ``[N]``.
            Higher = more OOD.
        ood_labels: Binary OOD ground truth, shape ``[N]``. ``1`` = OOD,
            ``0`` = in-distribution.
        probs_known: Per-label sigmoid probabilities for the *K* known
            labels, shape ``[N, K]``. Only rows where ``ood_labels == 0``
            contribute to macro-F1.
        labels_known: Multi-hot ground truth over the *K* known labels,
            shape ``[N, K]``. OOD rows are ignored for macro-F1.
        thresholds: Per-label decision thresholds, shape ``[K]``. A label
            is predicted positive iff ``probs_known[:, k] >= thresholds[k]``.
            Fix these on a validation set, not on the test set.

    Returns:
        ``(fpr, macro_f1, aml_oscr)`` with arrays sorted by FPR ascending.
        Each curve has at most ``N + 1`` points (one per unique novelty
        score plus a ``(0, 0)`` anchor). ``aml_oscr`` is the trapezoidal
        area under ``(fpr, macro_f1)`` in ``[0, 1]``; higher is better.
        Returns ``(np.array([0, 1]), np.array([0, 0]), 0.0)`` if either
        the ID or OOD class is empty.
    """
    novelty_scores = np.asarray(novelty_scores, dtype=float)
    ood_labels = np.asarray(ood_labels).astype(int)
    probs_known = np.asarray(probs_known, dtype=float)
    labels_known = np.asarray(labels_known).astype(int)
    thresholds = np.asarray(thresholds, dtype=float)

    n = novelty_scores.shape[0]
    if probs_known.shape[0] != n or labels_known.shape[0] != n:
        raise ValueError(
            "aml_oscr_curve: probs_known/labels_known must have N rows "
            f"matching novelty_scores ({n}); got {probs_known.shape[0]} / "
            f"{labels_known.shape[0]}"
        )
    if probs_known.shape != labels_known.shape:
        raise ValueError(
            "aml_oscr_curve: probs_known and labels_known must have the "
            f"same shape; got {probs_known.shape} vs {labels_known.shape}"
        )
    K = probs_known.shape[1]
    if thresholds.shape != (K,):
        raise ValueError(
            f"aml_oscr_curve: thresholds must have shape [K]=({K},); "
            f"got {thresholds.shape}"
        )

    id_mask = ood_labels == 0
    n_id = int(id_mask.sum())
    n_ood = int(n - n_id)
    if n_id == 0 or n_ood == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 0.0]), 0.0
    if novelty_scores.min() == novelty_scores.max():
        return np.array([0.0, 1.0]), np.array([0.0, 0.0]), 0.0

    preds = (probs_known >= thresholds[None, :]).astype(np.int64)

    order = np.argsort(novelty_scores, kind="stable")
    s_sorted = novelty_scores[order]
    is_id = id_mask[order]
    is_ood = (~id_mask)[order].astype(np.int64)

    pred_acc = preds[order] * is_id[:, None].astype(np.int64)
    true_acc = labels_known[order] * is_id[:, None].astype(np.int64)
    tp_inc = (pred_acc & true_acc).astype(np.int64)
    fp_inc = (pred_acc & (1 - true_acc)).astype(np.int64)
    fn_inc = ((1 - pred_acc) & true_acc).astype(np.int64)

    cum_tp = np.cumsum(tp_inc, axis=0)
    cum_fp = np.cumsum(fp_inc, axis=0)
    cum_fn = np.cumsum(fn_inc, axis=0)
    cum_ood = np.cumsum(is_ood)

    last_of_run = np.empty(n, dtype=bool)
    last_of_run[:-1] = s_sorted[1:] != s_sorted[:-1]
    last_of_run[-1] = True

    tp_pts = cum_tp[last_of_run]
    fp_pts = cum_fp[last_of_run]
    fn_pts = cum_fn[last_of_run]
    fpr_pts = cum_ood[last_of_run] / n_ood

    denom = (2 * tp_pts + fp_pts + fn_pts).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        f1_per_label = np.where(denom > 0, (2 * tp_pts) / denom, 0.0)
    macro_f1_pts = f1_per_label.mean(axis=1)

    fpr = np.concatenate(([0.0], fpr_pts))
    macro_f1 = np.concatenate(([0.0], macro_f1_pts))

    _trapz = getattr(np, "trapezoid", None) or np.trapz  # type: ignore[attr-defined]
    return fpr, macro_f1, float(_trapz(macro_f1, fpr))


def compute_aml_oscr(
    novelty_scores: np.ndarray,
    ood_labels: np.ndarray,
    probs_known: np.ndarray,
    labels_known: np.ndarray,
    thresholds: np.ndarray,
    fpr_max: float | None = None,
) -> float:
    """Area under the AML-OSCR (multi-label OSCR) curve.

    | Applies to   | Task                                  |
    |--------------|---------------------------------------|
    | Multi-label  | OSR (joint classify+reject, OS-MLC)   |

    Wraps ``aml_oscr_curve`` and returns the area under the macro-F1 vs
    novelty-FPR curve. With ``fpr_max=None`` (default) the area is over
    the full ``FPR ∈ [0, 1]`` range and lies in ``[0, 1]``. Pass
    ``fpr_max < 1`` to integrate only the deployment-relevant
    high-rejection regime (e.g. ``fpr_max=0.2`` ↔ novelty TPR ≥ 0.8);
    the result is **not** rescaled to ``[0, 1]`` and should be reported
    alongside the cap.

    Args:
        novelty_scores: shape ``[N]``. Higher = more OOD.
        ood_labels: binary OOD ground truth, shape ``[N]``.
        probs_known: per-label probabilities, shape ``[N, K]``.
        labels_known: multi-hot ground truth, shape ``[N, K]``.
        thresholds: per-label decision thresholds, shape ``[K]``.
        fpr_max: if given, integrate only over ``FPR ∈ [0, fpr_max]``.

    Returns:
        AML-OSCR area. Higher is better.
    """
    fpr, macro_f1, full_area = aml_oscr_curve(
        novelty_scores, ood_labels, probs_known, labels_known, thresholds
    )
    if fpr_max is None:
        return full_area
    if not 0.0 < fpr_max <= 1.0:
        raise ValueError(
            f"compute_aml_oscr: fpr_max must be in (0, 1]; got {fpr_max}"
        )
    cap_idx = int(np.searchsorted(fpr, fpr_max, side="right"))
    if cap_idx == 0:
        return 0.0
    if cap_idx >= len(fpr):
        return full_area
    f0 = float(fpr[cap_idx - 1])
    x_hi = float(fpr[cap_idx])
    f1_lo = float(macro_f1[cap_idx - 1])
    f1_hi = float(macro_f1[cap_idx])
    if x_hi == f0:
        f1_at_cap = f1_hi
    else:
        f1_at_cap = f1_lo + (f1_hi - f1_lo) * (fpr_max - f0) / (x_hi - f0)
    fpr_cut = np.concatenate((fpr[:cap_idx], [fpr_max]))
    macro_f1_cut = np.concatenate((macro_f1[:cap_idx], [f1_at_cap]))
    _trapz = getattr(np, "trapezoid", None) or np.trapz  # type: ignore[attr-defined]
    return float(_trapz(macro_f1_cut, fpr_cut))


def per_novel_discovery_table(
    novelty_scores: np.ndarray,
    novel_label_vecs: np.ndarray,
    known_label_vecs: np.ndarray,
    threshold: float,
    novel_label_names: list[str] | None = None,
) -> dict:
    """Per-novel-label discovery rate, stratified by co-novel count and regime.

    | Applies to   | Task                                          |
    |--------------|-----------------------------------------------|
    | Multi-label  | OS-MLC diagnostic (which novelties recovered) |

    The model produces an image-level novelty score; this function
    measures, for each novel label ``ell``, the fraction of images
    carrying ``ell`` whose score exceeds ``threshold`` (i.e. flagged as
    novel). Each label is reported within strata that isolate the
    co-occurrence regime:

      * ``alone``       — only ``ell`` is positive among novel labels.
      * ``one_co``      — ``ell`` plus exactly one other novel label.
      * ``two_plus_co`` — ``ell`` plus two or more other novel labels.
      * ``pure``        — image is pure-novelty (zero positive known labels).
      * ``mixed``       — image is mixed-novelty (≥1 positive known label).
      * ``all``         — pooled (any image with ``ell`` positive).

    Args:
        novelty_scores: per-image novelty scores, shape ``[N]``.
            Higher = more OOD.
        novel_label_vecs: multi-hot novel-label matrix, shape
            ``[N, K_novel]``.
        known_label_vecs: multi-hot known-label matrix, shape
            ``[N, K_known]``. Used only to determine pure vs mixed
            regime per image.
        threshold: novelty operating point ``tau``. Image flagged
            novel iff ``score > tau``.
        novel_label_names: optional list of ``K_novel`` names used as
            output keys. Defaults to integer indices ``0 .. K_novel-1``.

    Returns:
        ``{label: {stratum: {"discovery": float, "n": int}}}``. A stratum
        with zero images for that label has ``discovery = nan`` and
        ``n = 0``.
    """
    novelty_scores = np.asarray(novelty_scores, dtype=float)
    novel_label_vecs = np.asarray(novel_label_vecs).astype(int)
    known_label_vecs = np.asarray(known_label_vecs).astype(int)

    n = novelty_scores.shape[0]
    if novel_label_vecs.shape[0] != n or known_label_vecs.shape[0] != n:
        raise ValueError(
            "per_novel_discovery_table: row counts must match "
            f"novelty_scores ({n}); got {novel_label_vecs.shape[0]} / "
            f"{known_label_vecs.shape[0]}"
        )

    K_novel = novel_label_vecs.shape[1]
    if novel_label_names is None:
        names: list = list(range(K_novel))
    else:
        if len(novel_label_names) != K_novel:
            raise ValueError(
                f"per_novel_discovery_table: novel_label_names has "
                f"{len(novel_label_names)} entries; expected K_novel={K_novel}"
            )
        names = list(novel_label_names)

    flagged = novelty_scores > threshold
    novel_count = novel_label_vecs.sum(axis=1)
    known_count = known_label_vecs.sum(axis=1)
    is_pure = (novel_count > 0) & (known_count == 0)
    is_mixed = (novel_count > 0) & (known_count > 0)

    out: dict = {}
    for k in range(K_novel):
        has_l = novel_label_vecs[:, k] == 1
        co_count = novel_count - novel_label_vecs[:, k]
        strata = {
            "all": has_l,
            "alone": has_l & (co_count == 0),
            "one_co": has_l & (co_count == 1),
            "two_plus_co": has_l & (co_count >= 2),
            "pure": has_l & is_pure,
            "mixed": has_l & is_mixed,
        }
        per_label: dict = {}
        for stratum, mask in strata.items():
            n_stratum = int(mask.sum())
            if n_stratum == 0:
                per_label[stratum] = {"discovery": float("nan"), "n": 0}
            else:
                per_label[stratum] = {
                    "discovery": float(flagged[mask].sum()) / n_stratum,
                    "n": n_stratum,
                }
        out[names[k]] = per_label
    return out


def compute_aoscr_multiclass(
    scores: np.ndarray,
    ood_labels: np.ndarray,
    preds: np.ndarray,
    y: np.ndarray,
    n_thresholds: int | None = None,
) -> float:
    """AOSCR for multi-class (single-label) OSR — convenience wrapper.

    | Applies to                | Task                  |
    |---------------------------|-----------------------|
    | Multi-class (single-label)| OSR (classify+reject) |

    Thin wrapper over ``compute_aoscr`` that accepts predictions in
    either form:

    - Integer class IDs, shape ``[N]`` — passed straight through.
    - Softmax / logit matrix, shape ``[N, K]`` — reduced via
      ``argmax(axis=1)`` before passing through.

    For multi-label OSR, use ``compute_aoscr`` directly with an
    exact-match indicator.

    Args:
        scores: Novelty scores, shape ``[N]``. Higher = more OOD.
        ood_labels: Binary OOD ground truth, shape ``[N]``.
        preds: Either integer class predictions ``[N]`` or a softmax /
            logit matrix ``[N, K]``.
        y: Integer ground-truth classes, shape ``[N]``.
        n_thresholds: Deprecated and ignored.

    Returns:
        AOSCR in ``[0, 1]``. Higher is better.
    """
    preds = np.asarray(preds)
    if preds.ndim == 2:
        class_predictions = preds.argmax(axis=1)
    elif preds.ndim == 1:
        class_predictions = preds
    else:
        raise ValueError(
            f"preds must be 1-D [N] or 2-D [N, K], got shape {preds.shape}"
        )
    return compute_aoscr(
        scores, ood_labels, class_predictions, np.asarray(y), n_thresholds=n_thresholds
    )

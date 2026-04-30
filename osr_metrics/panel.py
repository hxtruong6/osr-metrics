"""One-call publication metric panel.

``compute_panel`` runs every metric whose required inputs are present.
Use it when you want "the table" for a paper without remembering which
function applies to your setting.
"""
from __future__ import annotations

from typing import Any, Literal

import numpy as np

from .calibration import (
    brier_score,
    brier_score_multiclass,
    expected_calibration_error,
    expected_calibration_error_multiclass,
)
from .classification import (
    macro_auprc,
    macro_auprc_id_labels,
    per_label_auprc,
)
from .fourclass import compute_fourclass_metrics
from .multiclass import balanced_accuracy, macro_f1_multiclass, top1_accuracy
from .ood import (
    aupr_in,
    aupr_out,
    auroc,
    fpr_at_95tpr,
)
from .osr import compute_aoscr, compute_aoscr_multiclass
from .selective import aurc, eaurc, selective_risk_at_coverage

Setting = Literal["auto", "multiclass", "multilabel"]


def _infer_setting(
    *,
    preds: np.ndarray | None,
    label_vecs: np.ndarray | None,
    y: np.ndarray | None,
    probs: np.ndarray | None,
) -> Setting:
    if label_vecs is not None and getattr(label_vecs, "ndim", 1) == 2:
        return "multilabel"
    if y is not None and np.asarray(y).ndim == 1:
        return "multiclass"
    if preds is not None and np.asarray(preds).ndim in (1, 2):
        return "multiclass"
    if probs is not None and np.asarray(probs).ndim == 2 and y is not None:
        return "multiclass"
    return "multiclass"


def compute_panel(
    scores: np.ndarray | None = None,
    ood_labels: np.ndarray | None = None,
    *,
    preds: np.ndarray | None = None,
    y: np.ndarray | None = None,
    probs: np.ndarray | None = None,
    label_vecs: np.ndarray | None = None,
    label_names: list[str] | None = None,
    held_out_labels: list[str] | None = None,
    loss: np.ndarray | None = None,
    setting: Setting = "auto",
) -> dict[str, Any]:
    """Compute every applicable metric in one call.

    | Applies to | Task                       |
    |------------|----------------------------|
    | Any        | Publication metric panel   |

    Pass whichever inputs you have; the function infers what to compute.
    Missing inputs produce no key in the output (no errors). The
    ``setting`` argument disambiguates multi-class vs multi-label when
    both ``preds`` and ``label_vecs`` look plausible — leave as
    ``"auto"`` to infer from input shapes.

    Args:
        scores: Novelty scores ``[N]`` (higher = more OOD).
        ood_labels: Binary OOD ground truth ``[N]``.
        preds: Either integer predictions ``[N]`` (multi-class) or a
            multi-hot prediction matrix ``[N, K]`` (multi-label).
        y: Integer ground-truth class IDs ``[N]`` (multi-class).
        probs: Predicted probabilities. Shape ``[N, K]`` for multi-class
            softmax or multi-label sigmoid.
        label_vecs: Multi-hot ground-truth ``[N, K]`` (multi-label).
        label_names: K label names (multi-label).
        held_out_labels: Labels treated as held-out / unknown
            (multi-label).
        loss: Per-sample non-negative loss ``[N]`` for selective-prediction
            metrics (e.g. ``(y != y_pred).astype(float)`` for 0/1
            misclassification, or NLL / squared error). When provided
            together with ``scores``, the panel adds ``aurc``, ``eaurc``,
            and ``selective_risk@95`` to the output.
        setting: ``"auto"`` infers from shapes; ``"multiclass"`` or
            ``"multilabel"`` forces.

    Returns:
        Flat ``dict`` from metric name → scalar (or nested dict for
        ``compute_fourclass_metrics``).
    """
    if setting == "auto":
        setting = _infer_setting(
            preds=preds, label_vecs=label_vecs, y=y, probs=probs
        )

    out: dict[str, Any] = {"setting": setting}

    # --- OOD-detection block (task-agnostic) ---
    if scores is not None and ood_labels is not None:
        labels_arr = np.asarray(ood_labels)
        if labels_arr.sum() > 0 and (1 - labels_arr).sum() > 0:
            out["auroc"] = auroc(scores, ood_labels)
            out["fpr_at_95tpr"] = fpr_at_95tpr(scores, ood_labels)
            out["aupr_in"] = aupr_in(scores, ood_labels)
            out["aupr_out"] = aupr_out(scores, ood_labels)

    # --- OSR block ---
    if scores is not None and ood_labels is not None:
        if setting == "multiclass" and y is not None and (
            preds is not None or probs is not None
        ):
            preds_for_aoscr = preds if preds is not None else probs
            assert preds_for_aoscr is not None  # narrowed by guard above
            out["aoscr"] = compute_aoscr_multiclass(
                scores, ood_labels, preds_for_aoscr, y
            )
        elif setting == "multilabel" and preds is not None and label_vecs is not None:
            preds_arr = np.asarray(preds)
            labels_arr = np.asarray(label_vecs)
            exact_match = (preds_arr == labels_arr).all(axis=1).astype(int)
            out["aoscr"] = compute_aoscr(
                scores,
                ood_labels,
                exact_match,
                np.ones_like(exact_match),
            )

    # --- Closed-set classification ---
    if setting == "multiclass" and y is not None and (
        preds is not None or probs is not None
    ):
        preds_in = preds if preds is not None else probs
        assert preds_in is not None  # narrowed by guard above
        out["top1_accuracy"] = top1_accuracy(preds_in, y)
        out["macro_f1"] = macro_f1_multiclass(preds_in, y)
        out["balanced_accuracy"] = balanced_accuracy(preds_in, y)
    elif setting == "multilabel" and probs is not None and label_vecs is not None:
        out["macro_auprc"] = macro_auprc(probs, label_vecs)
        out["per_label_auprc"] = per_label_auprc(probs, label_vecs)
        if label_names is not None and held_out_labels is not None:
            out["macro_auprc_id_labels"] = macro_auprc_id_labels(
                probs, label_vecs, label_names, held_out_labels
            )

    # --- Calibration ---
    if probs is not None:
        if setting == "multiclass" and y is not None and np.asarray(probs).ndim == 2:
            out["ece"] = expected_calibration_error_multiclass(probs, y)
            out["brier"] = brier_score_multiclass(probs, y)
        elif setting == "multilabel" and label_vecs is not None:
            out["ece"] = expected_calibration_error(probs, label_vecs)
            out["brier"] = brier_score(probs, label_vecs)

    # --- Multi-label four-class breakdown ---
    if (
        setting == "multilabel"
        and scores is not None
        and label_vecs is not None
        and label_names is not None
        and held_out_labels is not None
    ):
        out["fourclass"] = compute_fourclass_metrics(
            scores, label_vecs, label_names, held_out_labels
        )

    # --- Selective prediction (task-agnostic) ---
    if loss is not None and scores is not None:
        out["aurc"] = aurc(scores, loss)
        out["eaurc"] = eaurc(scores, loss)
        out["selective_risk@95"] = selective_risk_at_coverage(
            scores, loss, coverage=0.95
        )

    return out

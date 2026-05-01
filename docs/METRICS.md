# Metric reference — what each number means

Alphabetical glossary of every public metric in `osr-metrics`. Each entry
is **interpretation-first**: what the number represents, what counts as
good or bad, when to report it, and when not to. For mathematical
definitions and call signatures see the function docstrings; for
"which metric should I use?" see [USAGE.md](USAGE.md); for shared
vocabulary (ID, OOD, near vs far novelty, …) see [CONCEPTS.md](CONCEPTS.md).

**Conventions assumed throughout.**
- Every OOD/novelty score: **higher = more OOD**. ID-positive metrics
  (`aupr_in`) flip the sign internally.
- "Multi-class" means single-label; "multi-label" means multi-hot.
- Range column is the value space; arrow shows which direction is
  better.

---

## Quick index

| Closed-set classification | OOD detection | OSR / OS-MLC | Calibration | Selective prediction | Statistics | Helpers |
|---|---|---|---|---|---|---|
| [`balanced_accuracy`](#balanced_accuracy) | [`aupr_in`](#aupr_in) | [`aml_oscr_curve` / `compute_aml_oscr`](#aml_oscr_curve--compute_aml_oscr) | [`brier_score`](#brier_score) | [`aurc`](#aurc) | [`bootstrap_ci`](#bootstrap_ci) | [`as_ood_scores`](#as_ood_scores) |
| [`f1_per_label`](#f1_per_label) | [`aupr_out`](#aupr_out) | [`compute_aoscr` / `oscr_curve`](#compute_aoscr--oscr_curve) | [`brier_score_multiclass`](#brier_score_multiclass) | [`eaurc`](#eaurc) | [`delong_test`](#delong_test) | [`build_fourclass_masks`](#build_fourclass_masks) |
| [`macro_auprc`](#macro_auprc) | [`auroc`](#auroc) | [`compute_aoscr_multiclass`](#compute_aoscr_multiclass) | [`expected_calibration_error`](#expected_calibration_error) | [`rc_curve`](#rc_curve) | [`paired_bootstrap_diff`](#paired_bootstrap_diff) | [`compute_fourclass_metrics`](#compute_fourclass_metrics) |
| [`macro_auprc_id_labels`](#macro_auprc_id_labels) | [`fpr_at_95tpr`](#fpr_at_95tpr) | [`compute_nf_rejection_at_tpr`](#compute_nf_rejection_at_tpr) | [`expected_calibration_error_multiclass`](#expected_calibration_error_multiclass) | [`selective_accuracy_at_coverage`](#selective_accuracy_at_coverage) | | [`compute_panel`](#compute_panel) |
| [`macro_f1_multiclass`](#macro_f1_multiclass) | [`fpr_at_tpr`](#fpr_at_tpr) | [`compute_rc_macro_f1`](#compute_rc_macro_f1) | | [`selective_risk_at_coverage`](#selective_risk_at_coverage) | | [`partition_ood_by_purity`](#partition_ood_by_purity) |
| [`macro_f1_with_thresholds`](#macro_f1_with_thresholds) | | [`per_novel_discovery_table`](#per_novel_discovery_table) | | [`warn_if_inverted_aurc`](#warn_if_inverted_aurc) | | [`warn_if_inverted_scores`](#warn_if_inverted_scores) |
| [`per_label_auprc`](#per_label_auprc) | | | | | | |
| [`top1_accuracy`](#top1_accuracy) | | | | | | |

---

## Alphabetical glossary

### `aml_oscr_curve` / `compute_aml_oscr`
**Multi-label OSCR — joint classify+reject curve for OS-MLC.** Range `[0, 1]`, ↑.
- **Means:** as you become less willing to accept novelty (lower FPR), how well does your closed-set multi-label classifier still do on the images you *do* accept? Macro-F1 over `K` known labels on accepted IDs vs novelty FPR. Area under that curve = AML-OSCR.
- **Impact:** this is your **headline operating curve** for an open-set multi-label deployment. A model that flags novelty well *and* keeps closed-set quality on retained IDs scores high. A model that achieves novelty TPR by indiscriminately rejecting everything has high closed-set drop and low AML-OSCR.
- **Use when:** reporting an OS-MLC model where both novelty rejection and known-label classification quality matter (medical imaging triage, content moderation with new tag classes).
- **Don't:** report only the area without quoting an operating point — for clinicians the (TPR, macro-F1) pair at the deployment threshold is more actionable. Use `fpr_max` to integrate only the high-rejection regime when the always-accept tail is irrelevant.

### `as_ood_scores`
**Score-direction adapter (helper, not a metric).**
- **Means:** rewrites a model's "confidence-like" output (where higher = more ID) into the library's convention (higher = more OOD). One-line boundary fix at the call site.
- **Use when:** importing scores from a model whose convention you can't change. Always wrap once at the boundary; never sprinkle sign flips through your eval code.

### `aupr_in`
**Area under PR curve, ID as positive.** Range `[0, 1]`, ↑. Baseline = ID prevalence.
- **Means:** if you treated the score as "this image is ID", how clean is the precision-recall trade-off? Pairs with `aupr_out` to summarize PR behavior on imbalanced ID/OOD splits.
- **Use when:** the deployed action is "accept as ID" and false-accepts are costly. AUROC alone hides class-imbalance effects; AUPR-In/Out together don't.
- **Don't:** report alone — always with `aupr_out` and the class prevalences. A high `aupr_in` on a 99% ID dataset is uninformative.

### `aupr_out`
**Area under PR curve, OOD as positive.** Range `[0, 1]`, ↑. Baseline = OOD prevalence.
- **Means:** if you treated the score as "this image is novel", how clean is the precision-recall trade-off?
- **Use when:** OOD is the rare class you want to surface (the typical novelty-detection setting). Often more discriminating than AUROC at high class imbalance.

### `aurc`
**Area under the Risk-Coverage curve.** Range `[0, 1]`, ↓.
- **Means:** average risk a selective classifier incurs across all coverage levels. A perfect ranker has AURC = baseline error × something tiny; a useless ranker reproduces the unconditional error at every coverage.
- **Impact:** single-number summary of "is my confidence/uncertainty score actually informative?" Lower means the model rejects the right examples first.
- **Use when:** you have any per-sample loss and a confidence/uncertainty signal — ask whether rejecting low-confidence samples actually buys you risk reduction.
- **Don't:** report alone for asymmetric workflows — pair with `selective_risk_at_coverage` at the operating point that matches your reject budget.

### `auroc`
**Area Under the ROC Curve.** Range `[0, 1]`, ↑. Baseline = 0.5 (random).
- **Means:** probability that a randomly drawn OOD sample scores higher than a randomly drawn ID sample. Ranking-based; threshold-independent.
- **Impact:** the universal sanity check. AUROC < 0.5 almost always means your score direction is inverted (`warn_if_inverted_scores` will flag it).
- **Use when:** comparing detectors at a glance, or as the headline number with `fpr_at_95tpr` for the deployment view.
- **Don't:** rely on alone under heavy class imbalance — AUROC saturates near 1 even when AUPR-Out is poor. Don't use as a per-method ranking when methods differ in calibration; use `oscr_curve`/`aml_oscr_curve` for joint quality.

### `balanced_accuracy`
**Macro-averaged recall across classes.** Range `[0, 1]`, ↑. Baseline = `1/K` (random over `K` balanced classes).
- **Means:** average per-class recall — what `top1_accuracy` would be if every class had equal weight. On long-tailed data, balanced accuracy says "do you actually classify the rare classes too?"
- **Use when:** classes are imbalanced and minority recall matters (medical, fraud, rare events).
- **Don't:** confuse with macro-F1 — balanced accuracy ignores precision. If false positives on minority classes hurt, prefer `macro_f1_multiclass`.

### `bootstrap_ci`
**Percentile bootstrap CI for any scalar metric.** Helper.
- **Means:** how stable is your reported number under resampling of the test set? Returns `(lower, mean, upper)` of the resample distribution.
- **Impact:** turns a single point estimate into an honest interval. Reviewers ask for this on any headline number.
- **Use when:** reporting any metric on a fixed test set, especially with `n < 5000`. Pass `stratify=True` for rare-positive problems.
- **Don't:** confuse with cross-validation variance — bootstrap CI is sampling noise on this dataset, not generalization variance.

### `brier_score`
**Mean squared error of probabilities (multi-label / binary).** Range `[0, 1]`, ↓.
- **Means:** how far your predicted probabilities sit from the truth. Decomposes into calibration + refinement — penalizes both miscalibration and lack of resolution.
- **Use when:** you want a single proper-scoring-rule number that punishes both overconfidence and underconfidence; common in clinical ML.
- **Don't:** mix with the multi-class form (Guo-2017 top-1 confidence). Mixing forms gives wrong answers.

### `brier_score_multiclass`
**Brier score on top-1 multi-class softmax (Guo 2017 form).** Range `[0, 1]`, ↓.
- **Means:** same idea as `brier_score` but on top-1 confidence vs top-1 correctness, matching the multi-class calibration literature.
- **Use when:** reporting calibration on a multi-class softmax classifier alongside `expected_calibration_error_multiclass`.

### `build_fourclass_masks`
**Partition images into ID-disease / No-Finding / Pure-OOD / Mixed-OOD (helper).**
- **Means:** mutually exclusive boolean masks for the four-leaf clinical taxonomy. Feeds `compute_fourclass_metrics`, `compute_rc_macro_f1`, and any per-regime analysis.
- **Use when:** any multi-label OSR analysis where the No-Finding image is meaningful (chest X-ray, histopathology with normal slides).

### `compute_aml_oscr`
See [`aml_oscr_curve` / `compute_aml_oscr`](#aml_oscr_curve--compute_aml_oscr).

### `compute_aoscr` / `oscr_curve`
**Open-Set Classification Rate (Dhamija 2018 / Vaze 2022).** Range `[0, 1]`, ↑.
- **Means:** can your model classify ID images correctly *and* reject OOD at the same time? The curve plots correct-classification rate on accepted IDs vs OOD-acceptance rate (FPR). AOSCR is the area.
- **Impact:** single number that prevents methods from gaming AUROC by destroying closed-set accuracy. The canonical multi-class joint metric.
- **Use when:** reporting multi-class OSR; pairs naturally with `top1_accuracy` (closed-set ceiling) and `auroc` (pure novelty detection).
- **Don't:** apply to multi-label without an exact-match reduction — for multi-label use `aml_oscr_curve` instead.

### `compute_aoscr_multiclass`
**Convenience wrapper for `compute_aoscr` on multi-class predictions.** Same range / direction.
- **Means:** identical to `compute_aoscr`, but accepts logits/softmax `[N, K]` directly (does the `argmax` for you).
- **Use when:** you have raw logits or softmax and don't want to write the `argmax` line.

### `compute_fourclass_metrics`
**AUROC + FPR@95 across five clinically meaningful four-class pairings.**
- **Means:** one call returns AUROC and FPR@95 for (ID+NF vs all OOD), (ID+NF vs Pure), (ID+NF vs Mixed), (ID-disease vs Mixed; near-OOD diagnostic), (NF vs Pure; healthy-vs-novel-only diagnostic), and the v15-compat (ID-disease vs all OOD).
- **Impact:** lets you see *which* sub-population a method struggles with — a method may rank well overall but collapse on Mixed-OOD.
- **Use when:** publication tables for chest-X-ray-style multi-label OSR.

### `compute_nf_rejection_at_tpr`
**Healthy-patient rejection rate at fixed ID-disease retention.** Range `[0, 1]`, ↑.
- **Means:** at the threshold that keeps `tpr=0.95` on real ID-disease images, what fraction of all-zero-label "No Finding" images get flagged for review?
- **Impact:** the false-alarm-on-healthy number a clinician will ask for. High = safe (healthy patients escalate to review). Low = unsafe (the model silently classifies a healthy patient as having a known disease).
- **Use when:** any multi-label OSR with a No-Finding regime in the test set.
- **Don't:** report without the calibration TPR — it's meaningless without the operating point.

### `compute_panel`
**One-call publication panel (helper).**
- **Means:** auto-detects multi-class vs multi-label from input shapes and computes every metric whose required inputs are present.
- **Use when:** the first pass on a new model — fastest way to get a wide-angle view. Then drop into individual metrics for the headline numbers and CIs.

### `compute_rc_macro_f1`
**Rejection-contagion macro-F1 (Regime B for OS-MLC).** Range `[0, 1]`, ↑. Delta `Δ ∈ [-1, 1]`, ↓.
- **Means:** macro-F1 over known labels restricted to **mixed-novelty images** (where a novel label co-occurs with known ones). The contagion `Δ = macro_f1(ID-only) − macro_f1(mixed)` measures how much the presence of a novel label degrades known-label prediction.
- **Impact:** quantifies "rejection contagion" — does the model suppress *all* labels when one is novel, or does it correctly keep the known-label predictions? A small `Δ` is the empirical evidence that classification and novelty rejection are decoupled.
- **Use when:** OS-MLC papers where the test set contains contaminated images.

### `delong_test`
**Paired AUROC z-test (DeLong 1988), O(n log n).** Returns z-statistic + p-value.
- **Means:** is method A's AUROC significantly different from method B's, on the *same* test set? Analytic asymptotic test — no resampling needed.
- **Impact:** the standard answer to "did your improvement actually beat the baseline?" Reviewers expect it for every AUROC delta.
- **Use when:** comparing two AUROCs on shared test data.
- **Don't:** use for non-AUROC metrics — use `paired_bootstrap_diff` instead. Don't use for unpaired (different test sets) comparisons.

### `eaurc`
**Excess AURC = AURC − optimal AURC.** Range `[0, 1]`, ↓.
- **Means:** how much risk you incur *beyond* a perfect ranker (one that orders samples by their actual loss). Subtracts the irreducible component.
- **Impact:** useful when comparing across datasets with different base rates — raw AURC isn't comparable; E-AURC is.
- **Use when:** cross-dataset selective-prediction comparisons.

### `expected_calibration_error`
**ECE (multi-label / binary, flattened).** Range `[0, 1]`, ↓.
- **Means:** average gap between predicted probability and observed positive rate, binned by predicted probability. "When the model says 0.8, is the true positive rate 0.8?"
- **Impact:** if calibrated, you can interpret probabilities as actionable risks. If not, downstream cost-sensitive decisions break.
- **Use when:** any multi-label or binary probability output that will feed a thresholded decision or be shown to a clinician.
- **Don't:** compare across different bin counts or weighting schemes — declare both. Don't apply to multi-class softmax with this function — use the `_multiclass` variant (this one will raise if you do).

### `expected_calibration_error_multiclass`
**ECE on top-1 multi-class confidence (Guo 2017 form).** Range `[0, 1]`, ↓.
- **Means:** the standard multi-class ECE — gap between top-1 confidence and top-1 correctness. The form quoted in nearly all multi-class calibration papers.
- **Use when:** multi-class softmax classifier, reporting calibration alongside top-1 accuracy.

### `f1_per_label`
**F1 score per label (multi-label diagnostic).** Each entry `[0, 1]`, ↑.
- **Means:** harmonic mean of precision and recall on each label independently. Surfaces which specific labels the model is bad at.
- **Use when:** debugging — macro-F1 hides per-label failures; this exposes them. Pair with `per_label_auprc` for threshold-free per-label diagnostics.

### `fpr_at_95tpr`
**False-positive rate at 95% true-positive rate.** Range `[0, 1]`, ↓.
- **Means:** at the threshold that catches 95% of OOD, how many ID samples did you wrongly reject? The single most-quoted operating-point number in OOD literature.
- **Impact:** lower = lower deployment cost (less manual review of legitimate ID samples).
- **Use when:** reporting alongside AUROC. The pair (AUROC, FPR@95) is standard in OOD papers (Hendrycks 2017 onward).
- **Don't:** confuse with FPR at 95% specificity — TPR=0.95 means 95% novelty-detection rate, not 95% ID-acceptance.

### `fpr_at_tpr`
**Generalized version of `fpr_at_95tpr`.**
- **Means:** same idea at any TPR. Use 0.80 or 0.99 when 0.95 isn't the deployment OP.

### `macro_auprc`
**Macro-averaged AUPRC across labels.** Range `[0, 1]`, ↑. Baseline = mean per-label prevalence.
- **Means:** per-label AUPRC averaged with equal weight per label. Threshold-free per-label classification quality summary.
- **Use when:** multi-label closed-set evaluation; standard alongside `macro_f1_with_thresholds`.
- **Don't:** confuse with `macro_auprc_id_labels` — see next.

### `macro_auprc_id_labels`
**Macro-AUPRC restricted to known (non-held-out) labels.** Range `[0, 1]`, ↑.
- **Means:** in leave-p-out OSR, held-out labels have zero positives in `test_id`, dragging per-label AUPRC to 0 and unfairly penalizing the macro. This metric excludes them.
- **Use when:** leave-p-out OSR experimental setups.

### `macro_f1_multiclass`
**Macro-averaged F1 over multi-class predictions.** Range `[0, 1]`, ↑.
- **Means:** average per-class F1 — equal weight per class regardless of frequency. Punishes both poor recall on rare classes and poor precision.
- **Use when:** imbalanced multi-class classification.

### `macro_f1_with_thresholds`
**Macro-F1 using per-label thresholds (multi-label).** Range `[0, 1]`, ↑.
- **Means:** same as macro-F1 but the threshold is *not* fixed at 0.5 — instead per-label thresholds (often F1-optimal on validation) are used. Closer to the deployment number than the threshold-free `macro_auprc`.
- **Use when:** multi-label deployment numbers, after fixing per-label thresholds on a held-out split.
- **Don't:** tune thresholds on the test set — the metric becomes optimistic.

### `oscr_curve`
See [`compute_aoscr` / `oscr_curve`](#compute_aoscr--oscr_curve).

### `paired_bootstrap_diff`
**Paired bootstrap CI + two-sided p on `metric(a) − metric(b)`.**
- **Means:** is the difference between two methods' scores on the same test set significant? Uses shared resample indices to remove the variance attributable to which test points were picked.
- **Impact:** the non-AUROC counterpart of `delong_test`. Required for principled deltas on macro-F1, AUPR, AOSCR, AML-OSCR, AURC.
- **Use when:** comparing any non-AUROC metric between two methods on shared data.
- **Don't:** report deltas without a CI/p — a 0.5-point macro-F1 difference may be within noise.

### `partition_ood_by_purity`
**Partition OOD images into pure-OOD vs mixed-OOD (helper).**
- **Means:** boolean masks splitting OOD images into "only held-out labels positive" vs "held-out + known labels co-positive". Feeds stratified AUROC and `compute_fourclass_metrics`.
- **Use when:** you need pure-vs-mixed novelty stratified reporting.

### `per_label_auprc`
**AUPRC computed independently per label.** Each entry `[0, 1]`, ↑.
- **Means:** threshold-free per-label classification quality. The diagnostic complement to `f1_per_label`.
- **Use when:** debugging a multi-label model — find the bottom-quartile labels and inspect their training data.

### `per_novel_discovery_table`
**Per-novel-label discovery rate, stratified by co-novel count and pure/mixed regime.** Each entry `[0, 1]`, ↑.
- **Means:** at a fixed novelty threshold, for each novel label `ℓ`, what fraction of images carrying `ℓ` get flagged? Broken down by `alone` / `one_co` / `two_plus_co` co-occurrence and by pure-novelty vs mixed-novelty regime.
- **Impact:** Regime-D diagnostic for OS-MLC papers. Answers "is the novelty score driven by one easy label, or does it actually find each novel label individually?" Surfaces hidden failures: a model with high overall novelty AUROC may achieve it by easily catching multi-novel-label images while missing every solo-novel-label case.
- **Use when:** OS-MLC evaluation where novel labels are individually identified in ground truth.
- **Don't:** confuse with per-novel-label *attribution* — the model still produces only an image-level score; per-label discovery is computed retroactively from ground truth.

### `rc_curve`
**Risk-coverage curve.** Underlying curve for `aurc` / `eaurc`.
- **Means:** as you accept more samples (higher coverage), how does average risk change? A useful confidence score keeps risk low at low coverage.
- **Use when:** plotting the selective-prediction trade-off, not just summarising it.

### `selective_accuracy_at_coverage`
**Accuracy on the top-`coverage` fraction of samples.** Range `[0, 1]`, ↑.
- **Means:** "if I accept the most-confident `c` fraction, what accuracy do I get?" Maps directly to a deployment SLA.
- **Use when:** "we can afford to defer 30% to humans — what's our accuracy on the rest?" type questions.

### `selective_risk_at_coverage`
**Mean loss on the top-`coverage` fraction of samples.** Range `[0, ∞)`, ↓.
- **Means:** the loss-side counterpart of accuracy-at-coverage; works with any per-sample loss (not just 0/1).
- **Use when:** loss is asymmetric (clinical false-negative cost ≫ false-positive) — accuracy-at-coverage hides it; risk-at-coverage exposes it.

### `top1_accuracy`
**Top-1 multi-class accuracy.** Range `[0, 1]`, ↑.
- **Means:** fraction of samples classified correctly. The closed-set ceiling that AOSCR converges to as OOD vanishes.
- **Use when:** multi-class classification — usually paired with `macro_f1_multiclass` or `balanced_accuracy` for imbalance-aware reporting.

### `warn_if_inverted_aurc`
**Heuristic warning for inverted confidence direction.** Helper.
- **Means:** AURC computed with confidence-inverted scores will look anomalously high — this surfaces the likely sign error.
- **Use when:** automatically guard against passing the wrong sign of confidence to the selective-prediction module.

### `warn_if_inverted_scores`
**Heuristic warning for inverted OOD score direction.** Helper.
- **Means:** if AUROC < 0.5 you almost certainly passed confidence (higher = more ID) where the library expects novelty (higher = more OOD). Wrap your call sites or use `as_ood_scores`.
- **Use when:** boundary-defense at any function that consumes scores from an external model.

---

## How to use this glossary

**Designing a methods section.** Pick one headline number per axis (closed-set quality, novelty detection, joint operating point), then pick one or two diagnostics per axis. Don't dump every metric — reviewers tune out.

**Debugging a model.** Run `compute_panel`, look for outliers, then drop into the per-label / per-stratum diagnostics (`f1_per_label`, `per_label_auprc`, `per_novel_discovery_table`, `compute_fourclass_metrics`).

**Reporting deltas.** Every claimed improvement gets a CI: `bootstrap_ci` for any single metric, `delong_test` for AUROC deltas, `paired_bootstrap_diff` for everything else.

**Picking operating points.** Joint metrics (`compute_aoscr`, `aml_oscr_curve`) summarize the curve; readers still want the (TPR, classification quality) pair at the deployment threshold quoted alongside.

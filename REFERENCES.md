# References

Bibliographic sources for the metrics implemented in `osr-metrics-py`.
Each metric in the codebase points to the short keys listed here (e.g.
`[Dhamija2018]`).  Project-original metrics are listed at the bottom.

---

## OOD detection (`osr_metrics/ood.py`)

### `auroc` — Area under the ROC curve
- **[Bradley1997]** Bradley, A. P. (1997). *The use of the area under the
  ROC curve in the evaluation of machine learning algorithms.* Pattern
  Recognition, 30(7), 1145–1159.
- **[Hendrycks2017]** Hendrycks, D. & Gimpel, K. (2017). *A Baseline for
  Detecting Misclassified and Out-of-Distribution Examples in Neural
  Networks.* ICLR. (Adopts AUROC as a primary OOD-detection metric.)

### `fpr_at_tpr`, `fpr_at_95tpr`
- **[Hendrycks2017]** (above) — defines the FPR-at-fixed-TPR convention.
- **[Liang2018]** Liang, S., Li, Y., & Srikant, R. (2018). *Enhancing the
  Reliability of Out-of-Distribution Image Detection in Neural Networks
  (ODIN).* ICLR. (Standardised FPR@95TPR reporting.)

### `aupr_in`, `aupr_out`
- **[Hendrycks2017]** (above) — defines AUPR-In and AUPR-Out for OOD.
- **[Davis2006]** Davis, J. & Goadrich, M. (2006). *The Relationship Between
  Precision-Recall and ROC Curves.* ICML.

### `oscr_curve`
- **[Dhamija2018]** Dhamija, A. R., Günther, M., & Boult, T. E. (2018).
  *Reducing Network Agnostophobia.* NeurIPS.
- **[Vaze2022]** Vaze, S., Han, K., Vedaldi, A., & Zisserman, A. (2022).
  *Open-Set Recognition: A Good Closed-Set Classifier Is All You Need.*
  ICLR. (Canonical FPR/CCR convention used here.)

### `bootstrap_ci`
- **[Efron1979]** Efron, B. (1979). *Bootstrap Methods: Another Look at the
  Jackknife.* Annals of Statistics, 7(1), 1–26.
- **[EfronTibshirani1993]** Efron, B. & Tibshirani, R. J. (1993). *An
  Introduction to the Bootstrap.* Chapman & Hall/CRC. (Percentile CI is
  Chapter 13.)

---

## Open-Set Recognition (`osr_metrics/osr.py`)

### `compute_aoscr`
- **[Dhamija2018]**, **[Vaze2022]** (see above).

### `compute_nf_rejection_at_tpr`
- Project-specific clinical metric; see "Project-original metrics" below.
- Conceptually related: **[Geifman2017]** Geifman, Y. & El-Yaniv, R.
  (2017). *Selective Classification for Deep Neural Networks.* NeurIPS.
- Calibration-at-fixed-TPR protocol is from **[Hendrycks2017]**.

---

## Calibration (`osr_metrics/calibration.py`)

### `expected_calibration_error`
- **[Naeini2015]** Naeini, M. P., Cooper, G. F., & Hauskrecht, M. (2015).
  *Obtaining Well Calibrated Probabilities Using Bayesian Binning.* AAAI.
  (Coins ECE.)
- **[Guo2017]** Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017).
  *On Calibration of Modern Neural Networks.* ICML. (Equal-width binning
  with `n_bins=15` is the convention used here.)

### `brier_score`
- **[Brier1950]** Brier, G. W. (1950). *Verification of Forecasts
  Expressed in Terms of Probability.* Monthly Weather Review, 78(1), 1–3.

---

## DeLong test (`osr_metrics/delong.py`)

### `delong_test`
- **[DeLong1988]** DeLong, E. R., DeLong, D. M., & Clarke-Pearson, D. L.
  (1988). *Comparing the Areas under Two or More Correlated Receiver
  Operating Characteristic Curves: A Nonparametric Approach.* Biometrics,
  44(3), 837–845.
- **[Sun2014]** Sun, X. & Xu, W. (2014). *Fast Implementation of DeLong's
  Algorithm for Comparing the Areas Under Correlated Receiver Operating
  Characteristic Curves.* IEEE Signal Processing Letters, 21(11), 1389–1393.
  (The O(n log n) midrank algorithm implemented here.)

---

## Classification (`osr_metrics/classification.py`)

### `macro_auprc`, `macro_auprc_id_labels`, `per_label_auprc`
- **[Davis2006]** (see above).
- **[Sorower2010]** Sorower, M. S. (2010). *A Literature Survey on
  Algorithms for Multi-Label Learning.* Technical report, Oregon State
  University. (Tutorial reference for macro-averaging in multi-label
  problems.)

### `macro_f1_with_thresholds`
- **[Lipton2014]** Lipton, Z. C., Elkan, C., & Narayanaswamy, B. (2014).
  *Optimal Thresholding of Classifiers to Maximize F1 Measure.* ECML PKDD.

### `f1_per_label`
- **[VanRijsbergen1979]** Van Rijsbergen, C. J. (1979). *Information
  Retrieval* (2nd ed.). Butterworth-Heinemann. (F-measure definition.)

---

## Four-class partitioning (`osr_metrics/fourclass.py`)

The four-way partition (ID-disease / No-Finding / Pure-OOD / Mixed-OOD)
and the pairings derived from it are project-specific. The "near-OOD vs.
far-OOD" framing they generalise comes from:

- **[Winkens2020]** Winkens, J., Bunel, R., Roy, A. G., Stanforth, R.,
  Natarajan, V., Ledsam, J. R., MacWilliams, P., Kohli, P., Karthikesalingam, A.,
  Kohl, S., Cemgil, T., Eslami, S. M. A., & Ronneberger, O. (2020).
  *Contrastive Training for Improved Out-of-Distribution Detection.*
  arXiv:2007.05566.
- **[Yang2022]** Yang, J., Wang, P., Zou, D., Zhou, Z., Ding, K., Peng, W.,
  Wang, H., Chen, G., Li, B., Sun, Y., Du, X., Zhou, K., Zhang, W.,
  Hendrycks, D., Li, Y., & Liu, Z. (2022). *OpenOOD: Benchmarking
  Generalized Out-of-Distribution Detection.* NeurIPS.

The AUROC/FPR primitives reused inside this module are cited under
`osr_metrics/ood.py` above.

---

## Selective prediction (`osr_metrics/selective.py`)

### `rc_curve`, `aurc`, `selective_*_at_coverage`
- Geifman, Y. & El-Yaniv, R. (2017). *Selective Classification for Deep Neural Networks.* NeurIPS. [arXiv:1705.08500](https://arxiv.org/abs/1705.08500). — defines the modern selective-classification setup; source for `rc_curve`, `aurc`, `selective_*_at_coverage`.

### `eaurc`
- Geifman, Y., Uziel, G. & El-Yaniv, R. (2019). *Bias-Reduced Uncertainty Estimation for Deep Neural Classifiers.* ICLR. [arXiv:1805.08206](https://arxiv.org/abs/1805.08206). — origin of E-AURC and the closed-form oracle `r + (1 − r)·ln(1 − r)`.

### Caveat on conflating selective classification with OOD detection
- Jaeger, P. et al. (2024). *Overcoming Common Flaws in the Evaluation of Selective Classification Systems.* NeurIPS. [arXiv:2407.01032](https://arxiv.org/abs/2407.01032). — modern restatement of the framework and warning about conflating selective classification with OOD detection.

---

## Project-original metrics

The following metrics have no external paper citation. They were
introduced in this codebase for chest-X-ray multi-label OSR evaluation.
Cite this package itself (see `CITATION.cff`) when reporting them.

- `partition_ood_by_purity` (`osr_metrics/ood.py`) — multi-label
  partition into Pure-OOD vs. Mixed-OOD. Inspired by the near-OOD
  framing of [Winkens2020] and [Yang2022], but the specific definition
  is project-internal.
- `compute_nf_rejection_at_tpr` (`osr_metrics/osr.py`) — fraction of
  No-Finding samples rejected at a fixed ID-disease retention rate.
  Selective-prediction adjacent ([Geifman2017]); calibration protocol
  follows [Hendrycks2017].
- `stability_score` (`osr_metrics/stability.py`) — `1 - MAD` over a
  yes-token probability trajectory. Loosely related to trajectory-based
  LM-uncertainty work (e.g. Kuhn, Gal, & Farquhar (2023), *Semantic
  Uncertainty*, ICLR), but the form here is original to this work.
- `build_fourclass_masks`, `compute_fourclass_metrics`,
  `auroc_mixed_vs_id_disease`, `auroc_nf_vs_pure`
  (`osr_metrics/fourclass.py`) — clinically-motivated 4-way partition
  and pairing-specific AUROC/FPR aggregations. Project-original.

---

## BibTeX

```bibtex
@article{Bradley1997,
  author  = {Bradley, Andrew P.},
  title   = {The use of the area under the {ROC} curve in the evaluation of machine learning algorithms},
  journal = {Pattern Recognition},
  volume  = {30},
  number  = {7},
  pages   = {1145--1159},
  year    = {1997},
}

@inproceedings{Hendrycks2017,
  author    = {Hendrycks, Dan and Gimpel, Kevin},
  title     = {A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2017},
}

@inproceedings{Liang2018,
  author    = {Liang, Shiyu and Li, Yixuan and Srikant, R.},
  title     = {Enhancing the Reliability of Out-of-Distribution Image Detection in Neural Networks},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2018},
}

@inproceedings{Davis2006,
  author    = {Davis, Jesse and Goadrich, Mark},
  title     = {The Relationship Between Precision-Recall and {ROC} Curves},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2006},
  pages     = {233--240},
}

@inproceedings{Dhamija2018,
  author    = {Dhamija, Akshay Raj and G\"unther, Manuel and Boult, Terrance E.},
  title     = {Reducing Network Agnostophobia},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2018},
}

@inproceedings{Vaze2022,
  author    = {Vaze, Sagar and Han, Kai and Vedaldi, Andrea and Zisserman, Andrew},
  title     = {Open-Set Recognition: A Good Closed-Set Classifier Is All You Need},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2022},
}

@article{Efron1979,
  author  = {Efron, Bradley},
  title   = {Bootstrap Methods: Another Look at the Jackknife},
  journal = {Annals of Statistics},
  volume  = {7},
  number  = {1},
  pages   = {1--26},
  year    = {1979},
}

@book{EfronTibshirani1993,
  author    = {Efron, Bradley and Tibshirani, Robert J.},
  title     = {An Introduction to the Bootstrap},
  publisher = {Chapman \& Hall/CRC},
  year      = {1993},
}

@inproceedings{Geifman2017,
  author    = {Geifman, Yonatan and El-Yaniv, Ran},
  title     = {Selective Classification for Deep Neural Networks},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2017},
}

@inproceedings{Naeini2015,
  author    = {Naeini, Mahdi Pakdaman and Cooper, Gregory F. and Hauskrecht, Milos},
  title     = {Obtaining Well Calibrated Probabilities Using {B}ayesian Binning},
  booktitle = {AAAI Conference on Artificial Intelligence},
  year      = {2015},
}

@inproceedings{Guo2017,
  author    = {Guo, Chuan and Pleiss, Geoff and Sun, Yu and Weinberger, Kilian Q.},
  title     = {On Calibration of Modern Neural Networks},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2017},
}

@article{Brier1950,
  author  = {Brier, Glenn W.},
  title   = {Verification of Forecasts Expressed in Terms of Probability},
  journal = {Monthly Weather Review},
  volume  = {78},
  number  = {1},
  pages   = {1--3},
  year    = {1950},
}

@article{DeLong1988,
  author  = {DeLong, Elizabeth R. and DeLong, David M. and Clarke-Pearson, Daniel L.},
  title   = {Comparing the Areas under Two or More Correlated Receiver Operating Characteristic Curves: {A} Nonparametric Approach},
  journal = {Biometrics},
  volume  = {44},
  number  = {3},
  pages   = {837--845},
  year    = {1988},
}

@article{Sun2014,
  author  = {Sun, Xu and Xu, Weichao},
  title   = {Fast Implementation of {DeLong}'s Algorithm for Comparing the Areas Under Correlated Receiver Operating Characteristic Curves},
  journal = {IEEE Signal Processing Letters},
  volume  = {21},
  number  = {11},
  pages   = {1389--1393},
  year    = {2014},
}

@techreport{Sorower2010,
  author      = {Sorower, Mohammad S.},
  title       = {A Literature Survey on Algorithms for Multi-Label Learning},
  institution = {Oregon State University},
  year        = {2010},
}

@inproceedings{Lipton2014,
  author    = {Lipton, Zachary C. and Elkan, Charles and Narayanaswamy, Balakrishnan},
  title     = {Optimal Thresholding of Classifiers to Maximize {F1} Measure},
  booktitle = {ECML PKDD},
  year      = {2014},
}

@book{VanRijsbergen1979,
  author    = {van Rijsbergen, C. J.},
  title     = {Information Retrieval},
  edition   = {2},
  publisher = {Butterworth-Heinemann},
  year      = {1979},
}

@article{Winkens2020,
  author  = {Winkens, Jim and Bunel, Rudy and Roy, Abhijit Guha and Stanforth, Robert and Natarajan, Vivek and Ledsam, Joseph R. and MacWilliams, Patricia and Kohli, Pushmeet and Karthikesalingam, Alan and Kohl, Simon and Cemgil, Taylan and Eslami, S. M. Ali and Ronneberger, Olaf},
  title   = {Contrastive Training for Improved Out-of-Distribution Detection},
  journal = {arXiv:2007.05566},
  year    = {2020},
}

@inproceedings{Yang2022,
  author    = {Yang, Jingkang and Wang, Pengyun and Zou, Dejian and Zhou, Zitang and Ding, Kunyuan and Peng, Wenxuan and Wang, Haoqi and Chen, Guangyao and Li, Bo and Sun, Yiyou and Du, Xuefeng and Zhou, Kaiyang and Zhang, Wayne and Hendrycks, Dan and Li, Yixuan and Liu, Ziwei},
  title     = {{OpenOOD}: Benchmarking Generalized Out-of-Distribution Detection},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2022},
}

@inproceedings{Geifman2019,
  author    = {Geifman, Yonatan and Uziel, Guy and El-Yaniv, Ran},
  title     = {Bias-Reduced Uncertainty Estimation for Deep Neural Classifiers},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2019},
}

@inproceedings{Jaeger2024,
  author    = {Jaeger, Paul F. and others},
  title     = {Overcoming Common Flaws in the Evaluation of Selective Classification Systems},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2024},
}
```

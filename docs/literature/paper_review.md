# Literature Review: the Two SECOM Course Papers (Verified Against the PDFs)

Written 2026-11-09 from the PDFs in `data/raw/` (`Paper_Kerdprasop (3).pdf`,
`Paper_Munirathinam (2).pdf`). Page numbers: PDF pages for Paper A, which has no
printed page numbers; printed journal pages (273–285) for Paper B.
This document **corrects** `paper_comparison_summary.md` where that summary disagrees with
the papers (Section 4).

---

## 1. The papers

| | **Paper A** | **Paper B** |
|---|---|---|
| Citation | K. Kerdprasop & N. Kerdprasop, "Feature Selection and Boosting Techniques to Improve Fault Detection Accuracy in the Semiconductor Manufacturing Process", IAENG, manuscript received Dec 2010 / revised Jan 2011 | S. Munirathinam & B. Ramadoss, "Predictive Models for Equipment Fault Detection in the Semiconductor Manufacturing Process", *IACSIT Int. J. of Engineering and Technology*, vol. 8, no. 4, Aug 2016, pp. 273–285, DOI 10.7763/IJET.2016.V8.898 |
| Called in the summary | "Study A" | "Study B (Ranthinam & Ramadoss)". The surname is **Munirathinam** |
| Tool | WEKA | WEKA and R (Python is in the keywords) |
| Framing | Data mining for fault detection | Explicitly **CRISP-DM** (Fig. 4), predictive maintenance |

## 2. What each paper actually does

### Paper A: Kerdprasop & Kerdprasop (2011)

| Step | What the paper does | Where |
|---|---|---|
| Data | 1,567 examples × 590 sensors; 104 fail (coded 1, positive), 1,463 pass (−1) | p. 2 |
| Cleaning | (1) remove features with a single value; (2) remove features with **more than 55% missing** → "remove 137 features … remaining 454 features" | pp. 2–3 |
| Feature selection | **Filters only:** chi-square, gain ratio, and their own **MeanDiff** (difference of fail-cluster and pass-cluster means, ranked). Plus **PCA** (feature extraction). Each keeps 168 features (about 95% of variance) | pp. 2–3 |
| Imputation | **Not described** | — |
| Imbalance ("case boosting") | **Duplicate the fail cases** in the training data until they equal the pass cases (45 → 1,096). This is random oversampling, *not* AdaBoost or any other ensemble boosting | p. 2 step 7, p. 4 |
| Models | Decision tree, naïve Bayes, logistic regression, k-NN (**k = 1**). Neural nets and SVM "cannot run to completion" for memory reasons | pp. 2–3 |
| Validation | 10-fold CV for the feature-selection comparison; a separate test set for the boosted models | p. 3 |
| Split | Test set: 468 (59 fail / 409 pass). Train: 1,099 (45 fail / 1,054 pass) | p. 4 |
| Metrics | TP rate (recall), FP rate, precision, F-measure | p. 3 |
| Result before boosting | MeanDiff + decision tree: FP rate 4.5%, precision 20.5%, recall 16%. Naïve Bayes + gain ratio: recall about 90% but FP rate about 80% | p. 6 |
| Result after boosting (test set) | Decision tree: TP 1.0, FP 0.161, precision 0.472, F 0.641 · k-NN: 0.983 / 0.24 / 0.372 / 0.54 · LR: 1.0 / 0.335 / 0.301 / 0.463 · NB: 0.746 / 0.352 / 0.234 / 0.356 | Table I, p. 4 |
| Interpretable model | The decision tree's root split is **S511 > 28.3784**, followed by S470, S16, S472, … | Fig. 10, p. 5 |

### Paper B: Munirathinam & Ramadoss (2016)

| Step | What the paper does | Where |
|---|---|---|
| Process model | CRISP-DM's six phases, explained one by one | pp. 274–275 |
| Data | 1,567 × 590; 104 fails; "a 1:14 proportion". It also says "1115 instances" and "discards 452 instances with null and missing values", which conflicts with its own 468 + 1,099 = 1,567 split | p. 275–276 |
| Scaling | Min-max to [−1, 1] | p. 276 |
| **Imputation** | **"Missing values were substituted with the class mean"** | p. 276 |
| Cleaning | Single-value features out; more than 55% missing out | p. 276 |
| Feature selection | (a) **SME knowledge**: 1,000 → 80+ variables (equipment data, not SECOM); (b) **correlation analysis**: drop predictors with \|r\| ≥ 0.8 using a deterministic rule (of the most-correlated pair, drop the one with the larger *average* correlation with all other predictors) → p = 28; (c) chi-square, PCA; (d) variable importance analysis (chi-square, decision trees, random forest); (e) gain ratio; (f) **AverageDiff** (the same idea as MeanDiff) | pp. 276–278 |
| Imbalance | Duplicate the fail cases to match the pass cases | p. 278 |
| Models | DT, NB, k-NN, LR, **ANN, SVM** (SVDD with kernel PCA); plus a **PD-kNN** fault detector (k-NN squared-distance threshold) | pp. 278–283 |
| Validation | 10-fold CV; a test set; also "3-fold cross-validation … repeated 2 times" | pp. 278–280 |
| Metrics | TP rate, FP rate, precision, F-measure, **MCC, ROC area, PRC area**, recall | p. 278 |
| Results | Fig. 21 / confusion matrices: **the same numbers as Paper A's Table I and Fig. 8** (see 3.3) | pp. 280–283 |

## 3. Critical assessment

### 3.1 Leakage: both papers fit preprocessing on the full dataset before splitting
- Both papers do cleaning and feature selection on all 1,567 rows **before** the train/test split (Paper A Fig. 1 and step 6; Paper B §VI). The thresholds and selected features have therefore "seen" the test rows.
- Paper B's **class-mean imputation uses the label to fill the features**. Every missing value tells the model which class the wafer belongs to. On the test set the label is unknown, so this cannot be deployed. It is target leakage and inflates the results.

### 3.2 The split is not stratified, although both papers say it is
- Both say "each data set maintains the same proportion of pass and fail cases". The numbers they report don't match that: test **59/468 = 12.6%** fail, train **45/1,099 = 4.1%** fail, against 6.6% overall.
- That doubles the fail share in the test set, which flatters precision.
- Duplicating 45 fail wafers about 24× and then using **1-NN** risks memorising those few cases.

### 3.3 Paper B is not independent evidence
Paper B reproduces Paper A's experiment:
- the same split (468: 59/409; 1,099: 45/1,054; 1,096 after duplication);
- identical confusion matrices (k-NN 58/1/98/311, DT 59/0/66/343, NB 44/15/144/265, LR 59/0/137/272);
- the same metric table;
- several sentences that are nearly word for word (e.g. "In the cleaning step, we remove 137 features …", the naïve Bayes "90% … 80%" passage).

Paper A is not cited for these. **Treat the two papers as one evidence source** for the classifier results. Paper B's own contributions are its CRISP-DM framing, the correlation-reduction rule, the extra metrics (MCC, ROC, PRC) and the PD-kNN / ANN / SVM descriptions. Paper B also contains internal errors:
- its appendix says "−1 corresponds to fail and +1 to a pass", but the UCI `.names` file and our data say −1 = pass and 1 = fail (104 ones);
- it labels precision as "Sensitivity/Precision";
- its instance counts are inconsistent.

### 3.4 The reported recall of 100% should not be expected
Several factors inflate the boosted results: preprocessing fitted on all the data, a test set with twice the real fail rate, a single random test split of 59 fails, and duplicated minority cases. Our leakage-free pipeline should expect **far lower** recall and precision. That gap would be a methodological result, not a failure of our pipeline.

### 3.5 Can the cleaning be reproduced?
- Applying "single value out, more than 55% missing out" to the full data gives **116 + 24 = 140 removed → 450 features**. Paper A reports 137 → 454.
- Our training-split result is also 140 → 450 (D-016, D-017), so fitting on train costs nothing here.
- The 3-column gap is unexplained. Possibly WEKA's "single value" test differs from variance = 0 (e.g. how all-missing-but-one columns are handled), or "454" counts the label attribute.

### 3.6 A cross-check with our own descriptive results
- Paper A's tree splits first on **S511**. If the paper's sensors are numbered from 1, S511 is our `feature_510`.
- `feature_510` is the 3rd-largest pass/fail mean shift in our descriptive report (+0.53 std, D-005).
- This is consistent, but the numbering assumption is unverified.

## 4. Corrections to `paper_comparison_summary.md`

| Summary claim | What the papers say | Verdict |
|---|---|---|
| Study B authors "Ranthinam & Ramadoss" | Munirathinam & Ramadoss | **Wrong name** |
| Study A models include "Ensemble Boosting (AdaBoost/XGBoost)" | DT, NB, LR, k-NN only. "Boosting" means duplicating fail cases (random oversampling). (XGBoost was published in 2014, after this 2011 paper) | **Wrong** |
| Study A feature selection: "Filter → Wrapper/Embedded → Boosting" | Filters (chi-square, gain ratio, MeanDiff) and PCA only. No wrapper or embedded method | **Wrong** |
| Study A: "KNN achieves high raw accuracy" | Accuracy is never reported. 1-NN works best on the cleaned data without further selection; after boosting k-NN has TP 0.983 but precision 0.372 | **Unsupported** |
| Study A: "Decision trees provide superior interpretability" | Yes: the MeanDiff + DT model is "the most appropriate" if comprehensibility matters | Correct |
| Study A: "feature selection + boosting essential for rare faults" | Roughly what the paper concludes (subject to the caveats in 3.1–3.4) | Correct, with caveats |
| Study B goal: "evaluate alternative distance metrics (adaptive Mahalanobis)" | Adaptive Mahalanobis appears only in **Paper A's related work** (Verdier & Ferreira) and in Paper B's reference list. Paper B's own goal is CRISP-DM predictive modelling with six classifiers and a PD-kNN detector | **Wrong attribution** |
| Study B models: KNN, DT, LR, NB | Plus **ANN and SVM** | Incomplete |
| Study B feature selection: "statistical variance/importance" | Correlation filter (\|r\| ≥ 0.8, deterministic average-correlation rule), chi-square, PCA, variable importance (RF/DT), gain ratio, AverageDiff, SME | Incomplete |
| Pipeline step 3A: "MICE with Decision Trees" | **Neither paper uses MICE.** Paper A doesn't describe imputation; Paper B uses class-mean imputation (target leakage) | Not from the papers. It's a methodological choice, and still a good one |
| Pipeline: split before preprocessing | Both papers preprocess **before** splitting | The summary is *better* than the papers here |
| Threshold 55% | Both papers: 55% | Correct |
| Metrics: precision, recall, F1 | Paper A: TP rate, FP rate, precision, F. Paper B adds MCC, ROC area, PRC area | Correct; Paper B adds more |

## 5. Consequences for this project

| Topic | Take from the papers | Do differently |
|---|---|---|
| Cleaning | Single-value removal and a 55% missing threshold. Ours (variance = 0, 60%) gives the **same 450 features** | Fit on the training split only (already done) |
| Split | — | A truly stratified split (6.62% / 6.69%) instead of 4.1% / 12.6% (already done) |
| Imputation | — | **Reject class-mean imputation** (target leakage). Candidates per D-021 |
| Correlation | Paper B's deterministic rule (drop the member of the pair with the larger average \|r\|) is an alternative to a random pick | Still deferred to Boruta (D-018); record it as an option in the SIP |
| Feature ranking | MeanDiff / AverageDiff ≈ our standardised pass/fail mean shift | Compute it on training folds only |
| Imbalance | Duplicating the minority class | Compare against SMOTE, applied **inside** CV folds only |
| Models | DT (interpretable), NB, LR, k-NN; Paper B adds ANN, SVM | Week 3 baseline is a Random Forest; add DT and LR as comparisons with the papers |
| Metrics | TP/FP rate, precision, F | Also MCC, PR-AUC, ROC-AUC, balanced accuracy |
| Benchmark | Table I numbers | Present them as an **optimistic upper reference**, not a target (3.4) |

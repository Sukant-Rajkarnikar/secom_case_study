# Scientific Diary — SECOM Data Understanding & Preparation

### D-001: Stratified 80/20 Train/Test Split & Test Sealing
- **Timestamp (UTC):** 2026-11-27 12:10
- **Decision:** Split 80/20 stratified on target (random_state=42) and sealed test_holdout.csv with SHA-256.
- **Rationale:** Preserves ~6.6% minority defect rate in train and test while preventing any test-set leakage.
- **Consequence:** Train: 1253 rows (83 defects); test holdout sealed and deleted from memory.

### D-002: Remove Zero-Variance Columns (Training Set Only)
- **Timestamp (UTC):** 2026-11-27 12:15
- **Decision:** Dropped 116 columns with exact variance == 0 on X_train.
- **Rationale:** Constant columns cannot separate pass from fail wafers.
- **Consequence:** Columns reduced from 590 to 474.

### D-003: Drop High-Missingness Columns (> 60% on Training Set)
- **Timestamp (UTC):** 2026-11-27 12:19
- **Decision:** Removed 24 columns with > 60% missing values on X_train_1.
- **Rationale:** Avoids relying heavily on imputed values in heavily sparse sensors while staying consistent with literature thresholds.
- **Consequence:** Columns reduced from 474 to 450; missing cells dropped from 33,091 to 11,211.

### D-004: Correlation Audit Only (No Pairwise Pruning)
- **Timestamp (UTC):** 2026-11-27 12:22
- **Decision:** Computed pairwise Pearson correlation on 450 training features without dropping columns.
- **Rationale:** Avoids arbitrary feature dropping before supervised feature selection.
- **Consequence:** Retained all 450 features; identified 608 pairs with |r| > 0.8.

### D-005: Outlier Treatment via 3σ Capping (Training Set Only)
- **Timestamp (UTC):** 2026-11-27 12:23
- **Decision:** Computed mean ± 3σ bounds on X_train_2 and capped 4,872 extreme cells.
- **Rationale:** Controls extreme scale distortion while keeping sample count intact and avoiding extra missingness.
- **Consequence:** Prepared X_train_3 with 1253 rows and 450 columns.



### D-006: Exact Duplicate Feature Audit
- **Timestamp (UTC):** 2026-11-27 12:52
- **Decision:** Audited and kept 450 columns; removed 0 exact duplicates.
- **Rationale:** Duplicate columns add no information and the audit is unsupervised.
- **Consequence:** Duplicate count and feature set remain training-only.

### D-007: k-NN Missing-Value Imputation
- **Timestamp (UTC):** 2026-11-27 12:52
- **Decision:** Fitted KNNImputer(n_neighbors=5, weights='distance') on training features.
- **Rationale:** k-NN preserves sensor relationships better than global mean/median imputation.
- **Consequence:** Missing cells changed from 11,211 to 0; artifact saved.

### D-008: Random-Forest Baseline
- **Timestamp (UTC):** 2026-11-27 12:52
- **Decision:** Fit RandomForestClassifier(n_estimators=300, random_state=42) and use 5-fold stratified CV.
- **Rationale:** Accuracy without minority recall is misleading for an imbalanced defect problem.
- **Consequence:** Training accuracy=1.0000; CV accuracy=0.9338; CV recall=0.0000.

### D-009: Feature-Selection Method Comparison
- **Timestamp (UTC):** 2026-11-27 12:52
- **Decision:** Created mutual-information and random-forest top-100 candidate sets; reserved Boruta for 10-seed stability.
- **Rationale:** Filter and embedded methods are fast comparators; Boruta provides stronger all-relevant evidence.
- **Consequence:** Candidate sets retained for downstream experiments; final selection depends on CV evidence.

### D-010: Normality Audit and Min-Max Scaling
- **Timestamp (UTC):** 2026-11-27 12:52
- **Decision:** Min-max-scaled the imputed training matrix after a D’Agostino-Pearson normality sample.
- **Rationale:** Scaling is required for distance-based pipelines and makes feature ranges comparable.
- **Consequence:** 47 of 450 feature samples passed p >= 0.05; scaled data saved.

### D-011: Imbalance-Treatment Comparison
- **Timestamp (UTC):** 2026-11-27 12:52
- **Decision:** Defined ROSE-style oversampling, SMOTE, ADASYN, and class-weight alternatives for nested CV.
- **Rationale:** Resampling must occur within each training fold to prevent leakage.
- **Consequence:** Comparison report is loaded when available; 3×5-fold rerun is reproducible from src/balancer_comparison.py.

### D-012: Leakage-Safe imblearn Pipeline
- **Timestamp (UTC):** 2026-11-27 12:52
- **Decision:** Built an imblearn pipeline with k-NN imputation, min-max scaling, SMOTE, and balanced random forest.
- **Rationale:** Each transformation is fitted inside each stratified CV training fold.
- **Consequence:** CV balanced accuracy=0.5096; defect recall=0.0243.

### D-013: Out-of-Fold Model Assessment
- **Timestamp (UTC):** 2026-11-27 12:52
- **Decision:** Used cross_val_predict to produce a leakage-safe confusion matrix and defect metrics.
- **Rationale:** The trade-off is explicit: Type I false alarms versus Type II missed defects.
- **Consequence:** OOF matrix [[TN=1164, FP=6], [FN=81, TP=2]].

### D-014: Model-Complexity Audit
- **Timestamp (UTC):** 2026-11-27 12:53
- **Decision:** Compared random-forest depths using balanced training accuracy and 5-fold CV.
- **Rationale:** The gap exposes over/under-fitting independently of raw accuracy.
- **Consequence:** Best CV depth observed: 3 (0.6133).

### D-015: Steps 3–5 State and Open Points
- **Timestamp (UTC):** 2026-11-27 12:53
- **Decision:** Saved the imputed/scaled training state and retained Boruta, balancing, threshold, and final-model choices as open decisions.
- **Rationale:** The sealed test set remains untouched, and no test-based tuning occurred.
- **Consequence:** Final state saved to prep_step3_5_state.json; test checksum verified=True.

### D-006: Exact Duplicate Feature Audit
- **Timestamp (UTC):** 2026-11-27 12:55
- **Decision:** Audited and kept 450 columns; removed 0 exact duplicates.
- **Rationale:** Duplicate columns add no information and the audit is unsupervised.
- **Consequence:** Duplicate count and feature set remain training-only.

### D-007: k-NN Missing-Value Imputation
- **Timestamp (UTC):** 2026-11-27 12:55
- **Decision:** Fitted KNNImputer(n_neighbors=5, weights='distance') on training features.
- **Rationale:** k-NN preserves sensor relationships better than global mean/median imputation.
- **Consequence:** Missing cells changed from 11,211 to 0; artifact saved.

### D-008: Random-Forest Baseline
- **Timestamp (UTC):** 2026-11-27 12:55
- **Decision:** Fit RandomForestClassifier(n_estimators=300, random_state=42) and use 5-fold stratified CV.
- **Rationale:** Accuracy without minority recall is misleading for an imbalanced defect problem.
- **Consequence:** Training accuracy=1.0000; CV accuracy=0.9338; CV recall=0.0000.

### D-009: Feature-Selection Method Comparison
- **Timestamp (UTC):** 2026-11-27 12:55
- **Decision:** Created mutual-information and random-forest top-100 candidate sets; reserved Boruta for 10-seed stability.
- **Rationale:** Filter and embedded methods are fast comparators; Boruta provides stronger all-relevant evidence.
- **Consequence:** Candidate sets retained for downstream experiments; final selection depends on CV evidence.

### D-010: Normality Audit and Min-Max Scaling
- **Timestamp (UTC):** 2026-11-27 12:56
- **Decision:** Min-max-scaled the imputed training matrix after a D’Agostino-Pearson normality sample.
- **Rationale:** Scaling is required for distance-based pipelines and makes feature ranges comparable.
- **Consequence:** 47 of 450 feature samples passed p >= 0.05; scaled data saved.

### D-011: Imbalance-Treatment Comparison
- **Timestamp (UTC):** 2026-11-27 12:56
- **Decision:** Defined ROSE-style oversampling, SMOTE, ADASYN, and class-weight alternatives for nested CV.
- **Rationale:** Resampling must occur within each training fold to prevent leakage.
- **Consequence:** Comparison report is loaded when available; 3×5-fold rerun is reproducible from src/balancer_comparison.py.

### D-012: Leakage-Safe imblearn Pipeline
- **Timestamp (UTC):** 2026-11-27 12:56
- **Decision:** Built an imblearn pipeline with k-NN imputation, min-max scaling, SMOTE, and balanced random forest.
- **Rationale:** Each transformation is fitted inside each stratified CV training fold.
- **Consequence:** CV balanced accuracy=0.5096; defect recall=0.0243.

### D-013: Out-of-Fold Model Assessment
- **Timestamp (UTC):** 2026-11-27 12:56
- **Decision:** Used cross_val_predict to produce a leakage-safe confusion matrix and defect metrics.
- **Rationale:** The trade-off is explicit: Type I false alarms versus Type II missed defects.
- **Consequence:** OOF matrix [[TN=1164, FP=6], [FN=81, TP=2]].

### D-014: Model-Complexity Audit
- **Timestamp (UTC):** 2026-11-27 12:57
- **Decision:** Compared random-forest depths using balanced training accuracy and 5-fold CV.
- **Rationale:** The gap exposes over/under-fitting independently of raw accuracy.
- **Consequence:** Best CV depth observed: 3 (0.6133).

### D-015: Steps 3–5 State and Open Points
- **Timestamp (UTC):** 2026-11-27 12:57
- **Decision:** Saved the imputed/scaled training state and retained Boruta, balancing, threshold, and final-model choices as open decisions.
- **Rationale:** The sealed test set remains untouched, and no test-based tuning occurred.
- **Consequence:** Final state saved to prep_step3_5_state.json; test checksum verified=True.

### D-006: Exact Duplicate Feature Audit
- **Timestamp (UTC):** 2026-11-27 13:49
- **Decision:** Audited and kept 450 columns; removed 0 exact duplicates.
- **Rationale:** Duplicate columns add no information and the audit is unsupervised.
- **Consequence:** Duplicate count and feature set remain training-only.

### D-006: Exact Duplicate Feature Audit
- **Timestamp (UTC):** 2026-11-27 13:52
- **Decision:** Audited and kept 450 columns; removed 0 exact duplicates.
- **Rationale:** Duplicate columns add no information and the audit is unsupervised.
- **Consequence:** Duplicate count and feature set remain training-only.

### D-006: Exact Duplicate Feature Audit
- **Timestamp (UTC):** 2026-11-27 13:53
- **Decision:** Audited and kept 450 columns; removed 0 exact duplicates.
- **Rationale:** Duplicate columns add no information and the audit is unsupervised.
- **Consequence:** Duplicate count and feature set remain training-only.

### D-006: Exact Duplicate Feature Audit
- **Timestamp (UTC):** 2026-11-27 13:53
- **Decision:** Audited and kept 450 columns; removed 0 exact duplicates.
- **Rationale:** Duplicate columns add no information and the audit is unsupervised.
- **Consequence:** Duplicate count and feature set remain training-only.

### D-006: Exact Duplicate Feature Audit
- **Timestamp (UTC):** 2026-11-27 13:53
- **Decision:** Audited and kept 450 columns; removed 0 exact duplicates.
- **Rationale:** Duplicate columns add no information and the audit is unsupervised.
- **Consequence:** Duplicate count and feature set remain training-only.

### D-006: Exact Duplicate Feature Audit
- **Timestamp (UTC):** 2026-11-27 13:55
- **Decision:** Audited and kept 450 columns; removed 0 exact duplicates.
- **Rationale:** Duplicate columns add no information and the audit is unsupervised.
- **Consequence:** Duplicate count and feature set remain training-only.

### D-007: k-NN Missing-Value Imputation
- **Timestamp (UTC):** 2026-11-27 13:56
- **Decision:** Fitted KNNImputer(n_neighbors=5, weights='distance') on training features.
- **Rationale:** k-NN preserves sensor relationships better than global mean/median imputation.
- **Consequence:** Missing cells changed from 11,211 to 0; artifact saved.

### D-008: Random-Forest Baseline
- **Timestamp (UTC):** 2026-11-27 13:57
- **Decision:** Fit RandomForestClassifier(n_estimators=300, random_state=42) and use 5-fold stratified CV.
- **Rationale:** Accuracy without minority recall is misleading for an imbalanced defect problem.
- **Consequence:** Training accuracy=1.0000; CV accuracy=0.9338; CV recall=0.0000.

### D-008: Random-Forest Baseline
- **Timestamp (UTC):** 2026-11-27 14:01
- **Decision:** Fit RandomForestClassifier(n_estimators=300, random_state=42) and use 5-fold stratified CV.
- **Rationale:** Accuracy without minority recall is misleading for an imbalanced defect problem.
- **Consequence:** Training accuracy=1.0000; CV accuracy=0.9338; CV recall=0.0000.

### D-009: Feature-Selection Method Comparison
- **Timestamp (UTC):** 2026-11-27 14:02
- **Decision:** Created mutual-information and random-forest top-100 candidate sets; reserved Boruta for 10-seed stability.
- **Rationale:** Filter and embedded methods are fast comparators; Boruta provides stronger all-relevant evidence.
- **Consequence:** Candidate sets retained for downstream experiments; final selection depends on CV evidence.

### D-010: Normality Audit and Min-Max Scaling
- **Timestamp (UTC):** 2026-11-27 14:02
- **Decision:** Min-max-scaled the imputed training matrix after a D’Agostino-Pearson normality sample.
- **Rationale:** Scaling is required for distance-based pipelines and makes feature ranges comparable.
- **Consequence:** 47 of 450 feature samples passed p >= 0.05; scaled data saved.

### D-011: Imbalance-Treatment Comparison
- **Timestamp (UTC):** 2026-11-27 14:02
- **Decision:** Defined ROSE-style oversampling, SMOTE, ADASYN, and class-weight alternatives for nested CV.
- **Rationale:** Resampling must occur within each training fold to prevent leakage.
- **Consequence:** Comparison report is loaded when available; 3×5-fold rerun is reproducible from src/balancer_comparison.py.

### D-012: Leakage-Safe imblearn Pipeline
- **Timestamp (UTC):** 2026-11-27 14:03
- **Decision:** Built an imblearn pipeline with k-NN imputation, min-max scaling, SMOTE, and balanced random forest.
- **Rationale:** Each transformation is fitted inside each stratified CV training fold.
- **Consequence:** CV balanced accuracy=0.5096; defect recall=0.0243.

### D-013: Out-of-Fold Model Assessment
- **Timestamp (UTC):** 2026-11-27 14:03
- **Decision:** Used cross_val_predict to produce a leakage-safe confusion matrix and defect metrics.
- **Rationale:** The trade-off is explicit: Type I false alarms versus Type II missed defects.
- **Consequence:** OOF matrix [[TN=1164, FP=6], [FN=81, TP=2]].

### D-014: Model-Complexity Audit
- **Timestamp (UTC):** 2026-11-27 14:03
- **Decision:** Compared random-forest depths using balanced training accuracy and 5-fold CV.
- **Rationale:** The gap exposes over/under-fitting independently of raw accuracy.
- **Consequence:** Best CV depth observed: 3 (0.6133).

### D-015: Steps 3–5 State and Open Points
- **Timestamp (UTC):** 2026-11-27 14:03
- **Decision:** Saved the imputed/scaled training state and retained Boruta, balancing, threshold, and final-model choices as open decisions.
- **Rationale:** The sealed test set remains untouched, and no test-based tuning occurred.
- **Consequence:** Final state saved to prep_step3_5_state.json; test checksum verified=True.
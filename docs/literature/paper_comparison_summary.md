# Comparative Analysis & Workflow Summary: Semiconductor Equipment Fault Detection

> Reference document provided by the project team on 2026-11-09, reproduced as given.
> How it maps onto this project's pipeline, and where we deviate, is recorded in
> `docs/scientific_diary.md` (entry D-021).
>
> **Checked against the PDFs on 2026-11-09: several claims below are wrong** (author name,
> "AdaBoost/XGBoost", wrapper/embedded selection, the Mahalanobis attribution, MICE).
> See `paper_review.md` §4 before citing anything from this summary.

This document summarizes the core findings, data processing pipeline steps, and key machine learning concepts discussed regarding high-dimensional sensor data analysis (specifically the SECOM dataset) for equipment fault detection.

---

## 1. Paper Comparison Overview

| Feature / Aspect | Study A (Feature Selection & Boosting Focus) | Study B (Ranthinam & Ramadoss Focus) |
| :--- | :--- | :--- |
| **Primary Goal** | Stabilize performance amidst extreme class imbalance using combined feature selection and boosting. | Compare basic classification models and evaluate alternative distance metrics (e.g., adaptive Mahalanobis distance). |
| **Key Models Explored** | Decision Trees, K-Nearest Neighbors (KNN), Ensemble Boosting (e.g., AdaBoost/XGBoost). | KNN, Decision Trees, Logistic Regression, Naive Bayes. |
| **Feature Selection Strategy** | Multi-stage pipeline: Filter methods $\rightarrow$ Wrapper/Embedded methods $\rightarrow$ Boosting stabilization. | Standard feature subset selection based on statistical variance/importance. |
| **Core Finding** | KNN achieves high raw accuracy, but Decision Trees provide superior model interpretability. Combining feature selection with boosting is essential for detecting rare fault states. | Traditional SPC (Statistical Process Control) can be enhanced through ML techniques and tailored metrics for distance-based classification. |

---

## 2. Core Concepts & Definitions

### A. Classification Algorithms vs. Feature Selection Techniques
* **Classification Algorithms:** Models designed to predict categorical target variables (e.g., *Pass* vs. *Fail*). Examples include KNN, Logistic Regression, Naive Bayes, and Decision Trees.
* **Feature Selection Techniques:** Methods used to identify and select the most relevant parameters from high-dimensional space (e.g., 500+ sensor signals) to reduce noise, prevent overfitting, and lower computational cost.
  * **Filter Methods:** Evaluate features independently based on statistical properties (e.g., correlation, chi-square, ANOVA).
  * **Wrapper Methods:** Evaluate feature subsets by iteratively training and scoring a specific model (e.g., Recursive Feature Elimination).
  * **Embedded Methods:** Perform feature selection during model training itself (e.g., LASSO regularization, Decision Tree feature importance).

### B. Class Imbalance
* **Definition:** A scenario where one class significantly outnumbers another in the target variable (e.g., 95%+ "Pass" vs. $<5\%$ "Fail" in semiconductor manufacturing).
* **Identification:** Discovered during **Exploratory Data Analysis (EDA)** prior to data splitting.
* **Impact:** Raw accuracy becomes a misleading metric because a baseline model predicting only the majority class achieves high accuracy while failing to detect operational faults.

---

## 3. Standard End-to-End Data Pipeline

To maintain rigorous evaluation and prevent **Data Leakage**, the data pipeline must follow a strict sequential structure:

```
[ Raw Sensor Data ]
         │
         ▼
[ 1. Initial Data Cleaning & EDA ] ──► (Identify class imbalance & missingness patterns)
         │
         ▼
[ 2. Stratified 80/20 Train/Test Split ]
         │
         ├──► [ 20% Test Set ] (Held out untouched until final model evaluation)
         │
         ▼
[ 3. Training Set Processing (80%) ]
         │
         ├──► A. Advanced Imputation (e.g., MICE with Decision Trees)
         │      └─ Preserves sensor correlations & statistical distribution ($Mean$, $Variance$).
         │
         ├──► B. Feature Selection (Filters / Wrappers / Embedded)
         │      └─ Reduces 500+ features down to critical predictive parameters.
         │
         └──► C. Model Training & Class Balancing (Boosting / Oversampling)
                 └─ Trains models (KNN, Decision Trees, Ensembles) on selected features.
         │
         ▼
[ 4. Validation & Final Testing ] ──► (Evaluate using Precision, Recall, F1-Score on Test Set)
```

---

## 4. Imputation Strategy Validation

When replacing missing values in high-dimensional sensor data, validation occurs through two main checks:

1. **Statistical Consistency:**
   * Verify that summary statistics (mean $\mu$, variance $\sigma^2$, distribution shape) before and after imputation remain aligned without artificial variance shrinkage.
2. **Downstream Model Performance:**
   * The ultimate metric of success: evaluate whether the imputed data improves downstream metrics (Precision, Recall, ROC-AUC) on the unseen validation/test set compared to standard median or mean filling.

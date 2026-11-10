# SECOM build plan (derived from the Students Manual)

Source: `data/raw/Linda_SECOM Case Study - Students Manual (1).docx` (read 2026-11-10).
Section numbers (§) refer to the manual's Implementation Guide. Diary IDs (D-NNN) refer to
`docs/scientific_diary.md`. Status: **Done**, **Partial**, **To do**.

## Fixed rules (from the manual; not open decisions)

1. Split 80/20 stratified first; test set untouched until final evaluation (§3.4, §15.1).
2. Every fitted parameter (thresholds, 3σ bounds, imputer, selector, scaler) is learned on train only, then applied unchanged to test.
3. No mean imputation. No IQR outlier detection; use 3σ.
4. kNN imputation is the required starting point (k = 5, adjustable); features must be scaled for the distance calculation.
5. Balancing (SMOTE/ROSE) on training rows only, never on test.
6. Select on cross-validation score, never on test performance.
7. Sensitivity/recall over accuracy; final choice by cost: **Type I = 100, Type II = 400**.
8. Every preprocessing decision gets a SIP row (options | pros/cons | decision | consequence with before/after counts).

## Deliverables

| Deliverable | Requirement | Status |
|---|---|---|
| Python script | One script, runs end to end in VS Code with no manual steps; **downloads the data at runtime** (ucimlrepo id=179 or the UCI URL); installs missing packages automatically; header comment on every function (purpose, inputs, outputs) | To do |
| Scientific notebook | Google Docs template, CRISP-DM structure, scientific language (no "we"/"I"), SIP tables, all Q&A tables filled | Partial (diary + notebook 01 hold the content) |
| Presentation | MPMD PowerPoint master, PDF emailed before the talk, before→after scheme per step, one big-picture pipeline diagram | To do |
| Outputs | `results/figures/` (histograms, Pareto, heatmap, ROC/PR), `results/tables/` (SIP, confusion matrices, metrics) | Partial (currently under `reports/`) |

## Build steps

### Phase A: Data understanding (§2, week 2)
| # | Step | Status |
|---|---|---|
| A1 | Load both files at runtime, name columns `sensor_0…589`, check 1567 rows in each, keep X and y separate | Partial: loader reads `data/raw`, so the runtime download is still missing (D-002) |
| A2 | Shape, dtypes, duplicate rows | Done (D-004) |
| A3 | Class distribution: counts, %, tier (moderate imbalance, ~6.6%) | Done (D-013) |
| A4 | Feature scales and skew; histograms of flagged features | Done (D-004, D-024) |
| A5 | Missingness overall, by feature, by class | Done (D-009, D-014) |
| A6 | Figures: class bar chart, missingness bar chart, histograms → `results/figures/` | Partial (move/copy) |
| A7 | §2.7 observations, SIP 2.8 (target coding −1/1 vs 0/1; ucimlrepo vs direct download), Q&A 2.9 (last answer still one line) | Partial |

### Phase B: Split (§3)
| B1 | Stratified 80/20, `random_state=42`, class % table for original/train/test, test sealed | Done (D-015) |
| B2 | Notebook text: aim of hold-out and CV; quantitative argument for why a random split fails with ~104 defects | Partial (D-026, D-028 have the numbers) |

### Phase C: Rough reduction (§4)
| C1 | Zero-variance removal on X_train (590 → 474) | Done (D-016) |
| C2 | Duplicate columns **checked on X_train** + SIP row | Done (D-033: 0 among 474 varying / 450 kept); SIP row → notebook |
| C3 | Missingness histogram + Pareto; 60% threshold justified (474 → 450) | Done (D-017) |
| C4 | Correlation heatmap, no removal | Done (D-018) |
| C5 | Reduction summary table + SIP 4.8 | Done |

### Phase D: Outliers (§5)
| D1 | 3σ bounds per feature from train; flagged counts; top features | Done (D-019) |
| D2 | All three options described; cap chosen; before/after boxplot; SIP 5.6 | Done (D-019) |
| D3 | Make the outlier treatment a **switchable transformer** (none / cap / remove→NaN) so the week-7 grid search can vary it | Done (D-035: `OutlierTreatment` cap / remove / none, σ adjustable) |

### Phase E: Imputation (§6, week 3)
| E1 | Recheck NaN count after outlier step | Done |
| E2 | Mean imputation ruled out with SECOM-specific reasons | Done (D-021) |
| E3 | kNN k = 5 with scaling (`ScaledImputer`), k adjustable; 0 NaN after; before/after histogram of a heavily missing feature | Done (D-025, D-027) |
| E4 | **Decide the imputer.** D-029 proposes median + flags; the manual expects kNN as the main method, with median only as an optional comparison. Recommendation: kNN (scaled) as the official method, median kept as the documented alternative, since the CV gap is 0.006 PR-AUC and not significant | Done (D-032: k-NN) |
| E5 | First random forest model; report the training accuracy and say why it is optimistic | Done (D-033: train acc 1.000 vs CV 0.934, recall 0) |

### Phase F: Feature selection (§7, week 4)
| F1 | Explain reduction vs. selection (with a figure); filter / wrapper / embedded families | To do |
| F2 | **Boruta** (`boruta` package, BorutaPy with RF) on train only → about 15–20 features incl. tentative ones; explain shadow features | Done (D-033: 13 features, 10-seed stability); theory text → notebook |
| F3 | Optional cross-check: ANOVA F-test + RF importance; report overlap with Boruta | Done (D-033: 7/13, 12/13) |
| F4 | §7.6: CV of all features vs selected set (mean and SD) | Done (D-033) |
| F5 | SIP 7.7 (method, N, tentative handling) | Done in diary (D-033); → notebook |

### Phase G: Scaling (§8, week 5)
| G1 | Normality step only if the model needs it (RF: skip, documented; LR/SVM: Yeo-Johnson optional) | Done (D-035: 445/450 non-normal; RF → no transform; Yeo-Johnson kept for LR/SVM) |
| G2 | Min-max scaler (week 5) fitted on train; DataFrame with names kept; SIP 8.5 | Done (D-035: after Boruta, train range [0, 1], names kept); SIP → notebook |

### Phase H: Imbalance (§9)
| H1 | Train class counts + majority-class baseline accuracy | Done (D-035: 1,170 / 83; baseline accuracy 0.934) |
| H2 | SMOTE (ROSE is R-only; note it), compare with class weighting and ADASYN; counts before/after; SIP 9.6 | Done (D-035: ROSE-style sensitivity 0.289, SMOTE 0.145 at threshold 0.5); SIP → notebook |

### Phase I: Pipeline (§10)
| I1 | `imblearn.pipeline.Pipeline`: imputer → selector → scaler → SMOTE → model (rows A–D stay outside, fitted once). Note: §10.3 puts the scaler before selection and §17.1 after it; choose one order and document it | Done (D-035: selector before scaler, `build_pipeline`) |
| I2 | Shape check, reproducibility check (identical predictions), row-count leakage print | Done (D-035: `pipeline_checks.json`) |
| I3 | Single `RANDOM_STATE = 42` for split, SMOTE, model, CV splitter | Done (`RANDOM_STATE = 42` everywhere) |
| I4 | Q&A 10.6 (five questions) | To do |

### Phase J: Models and tuning (§11–12, weeks 5–6)
| J1 | Dummy classifier (recall ≈ 0) as baseline | To do |
| J2 | Candidates: RF (main), logistic regression, optionally SVM / gradient boosting; same pipeline, `StratifiedKFold(5, shuffle, 42)`; mean ± SD table; SIP 11.4 | To do |
| J3 | Grid search over outlier treatment × kNN k × balancer × RF params (`n_estimators`, `max_depth`, ...); run selection before the heavy search; log every trial (methods + params) | To do |
| J4 | Top-10 results and stability; note parameters that hit a grid edge; visualise trials | To do |

### Phase K: Evaluation and cost (§13–14, weeks 4, 6, 7)
| K1 | Define success = detecting a faulty wafer (positive = 1); explain TP/TN/FP/FN and Type I/II | To do |
| K2 | Accuracy vs baseline, sensitivity, specificity, precision, F1, F2, ROC-AUC, PR curve | Partial (D-035: confusion matrices + metrics at threshold 0.5 on CV; curves to do) |
| K3 | Precision–recall trade-off; threshold choice on **CV predictions** (not test) | To do |
| K4 | Cost = 100·FP + 400·FN; pick model + threshold with the lowest cost; sensitivity analysis (300/500, and the manual's €200/€5000 variant) | To do |
| K5 | Q&A 13.6; SIP 14.5 | To do |

### Phase L: Final test evaluation (§15, week 7)
| L1 | `verify_sealed` → refit the best pipeline on all of train → transform-only on test (D-016/D-017 lists, D-019 bounds) | To do |
| L2 | Test confusion matrix + all metrics + total cost; bootstrap CI (D-028) | To do |
| L3 | Train vs test side by side; size of the gap vs model complexity | To do |

### Phase M: Wrap-up (§16–17)
| M1 | Stage-by-stage before/after summary; strengths, limitations, next steps | To do |
| M2 | Big-picture diagram: which steps are fit on train, which are applied to test | To do |
| M3 | Reproducibility: pinned `requirements.txt`, data source + access date, `joblib` dump of the fitted pipeline, relative paths, clean "run all" | To do |
| M4 | End-to-end Q&A (framing, big picture, CRISP-DM loop example, leakage, 590→N, imbalance, cost) | To do |
| M5 | Roll-up checklist §17.3 ticked; script, notebook, slides cross-checked | To do |

## Conflicts inside the manual (team to confirm)

| Topic | Manual says | Plan uses |
|---|---|---|
| Cost weights | Weeks 6–7, §16.2, §17.3: 100 / 400. §14.2: €200 / €5000 | 100 / 400 as primary; €200/€5000 in the sensitivity analysis |
| Boruta | Week 4, §17.1, final recommendations: implement it, target 15–20 features. §7.3: optional | Boruta as the main method |
| Scaling vs selection order | §10.3: scaler before selector. §17.1: selector before scaler | Selector before scaler (min-max on the final features); kNN does its own internal scaling |
| Outlier step in grid search | §10.2: outside the pipeline. Week 7: grid over outlier treatments | Switchable transformer inside the pipeline (D3) |
| Imputer | Manual: kNN required. Diary D-029: median + flags proposed | See E4 |

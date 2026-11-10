# SECOM scaffold overview

Snapshot of the project scaffold as of 2026-11-10 (branch `scaffold/secom-foundation`).
It covers the layout, the modules, the data flow, the commands, and what is done and what is still to build.
Details on any decision are in [scientific_diary.md](scientific_diary.md) (D-NNN IDs), and the
step-by-step plan is in [build_plan.md](build_plan.md).

## Purpose

The project is a CRISP-DM case study on the UCI SECOM semiconductor dataset: 1567 wafers, 590 sensor
features, and a pass/fail label with about 6.6% defects (about 104 fails). The goal is to detect
faulty wafers, with models chosen by cost (Type I = 100, Type II = 400). The test set is kept
sealed from every preprocessing and modelling decision.

## Fixed rules

1. Stratified 80/20 split first (`random_state=42`). The test set is untouched until final evaluation.
2. Every fitted parameter (drop lists, 3σ bounds, imputer, selector, scaler) is learned on train
   only and then applied unchanged to test.
3. Outliers use the 3σ rule, not IQR. The missing-value drop threshold is 60%. Mean imputation is not used.
4. Every preprocessing step gets a Step Impact Protocol (SIP) table: options, pros/cons, decision,
   and the counts before and after.
5. Models are selected on cross-validation scores and never on test performance.
6. Every change is logged in the scientific diary.

## Directory layout

```
SECOM/
├── pyproject.toml            # package secom-data-mining, Python ≥3.10, ruff + pytest config
├── README.md                 # setup and usage of every command
├── src/secom_analysis/       # the package (see Modules)
├── tests/                    # pytest suite (33 tests)
├── notebooks/
│   └── 01_data_understanding_and_preparation.ipynb
├── scripts/db.sh             # local PostgreSQL 16 cluster: init/start/stop/status/psql
├── docs/
│   ├── scaffold.md           # this file
│   ├── build_plan.md         # course manual → phases A–M with status
│   ├── scientific_diary.md   # append-only decision log (D-000 … D-035)
│   └── literature/           # paper review + comparison summary
├── data/                     # git-ignored
│   ├── raw/                  # secom.data, secom_labels.data, secom.names, manual, papers
│   └── processed/            # train/test splits, sealed test + sha256, prep state, imputed sets
├── models/imputers/          # fitted imputers (.joblib), git-ignored
└── reports/                  # generated outputs, git-ignored
    ├── health/  imputation/  imputation_cv/  split_study/  figures/
```

## Modules (`src/secom_analysis/`)

| Module | Role | Main entry points |
|---|---|---|
| `data.py` | Load features and labels (finds a single data sub-folder, keeps timestamps), stratified split | `load_secom`, `load_labels`, `stratified_split` |
| `eda.py` | Week 1–2 descriptive stats, constant features, missing-threshold filter, plots, Markdown report | `descriptive_statistics`, `column_summary`, `save_eda_overview` |
| `cli.py` | Runs the initial descriptive and split workflow | `python -m secom_analysis.cli` |
| `health.py` | PASS/WARN/FAIL health checks and profiles for the raw files, labels, features, splits and train/test drift (KS test) | `run_health_checks`, `python -m secom_analysis.health` |
| `report.py` | Standalone HTML descriptive report built from the health outputs | `write_descriptive_report` |
| `staging.py` | Loads the untouched source files into Postgres schema `secom_raw_staging` | `python -m secom_analysis.staging` |
| `preparation.py` | Saves the split, verifies the test seal (sha256), handles zero variance, missing %, the Pareto chart, 3σ bounds, cap/remove outliers and SIP tables | `save_split`, `verify_sealed`, `three_sigma_bounds`, `cap_outliers`, `sip_table` |
| `impute.py` | Fits 5 imputation strategies on train and compares them (masked reconstruction, distribution consistency, correlation distortion) | `python -m secom_analysis.impute` |
| `impute_viz.py` | Visual comparison of the strategies (distributions, relationships, variance heatmap, error by column) | `python -m secom_analysis.impute_viz` |
| `impute_cv.py` | Leakage-free downstream CV (3×5-fold, imputer fitted per fold), corrected t-test, time-ordered check | `python -m secom_analysis.impute_cv` |
| `production.py` | Loads the imputed sets into `secom_production`, keeps the strategy registry, selects a strategy, records CV results | `python -m secom_analysis.production --select/--record-cv` |
| `split_study.py` | Compares split and validation schemes: hold-outs, repeated CV, time hold-out, forward-chaining, leave-one-month-out | `python -m secom_analysis.split_study` |
| `selection.py` | Duplicate check on train, first random forest, Boruta stability selection (10 seeds, 450 → 13), ANOVA / RF-importance cross-checks, §7.6 CV | `python -m secom_analysis.selection` |
| `pipeline.py` | Switchable 3σ outlier step, normality check, min-max scaling, balancers (class weights / ROSE-style / SMOTE / ADASYN), full `imblearn` pipeline, first confusion matrices and cost | `build_pipeline`, `python -m secom_analysis.pipeline` |

Imputation strategies: `median`, `median_flags`, `knn` (scaled, k=5), `mice_bayesridge`, `mice_extratrees`.

## Data flow

```
UCI files (data/raw)
   │  staging.py ──────────────► Postgres secom_raw_staging (measurements, labels, source_files, load_log)
   ▼
load_secom → stratified 80/20 split ──► test_holdout.csv (+ .sha256)  [SEALED]
   │ train
   ▼
zero-variance drop (590 → 474) → missing > 60% drop (474 → 450) → correlation heatmap (no removal)
   → 3σ capping (bounds from train)  ──► train_prepared.csv, preparation_state.json, three_sigma_bounds.csv
   ▼
impute.py: 5 strategies ──► data/processed/imputed/, models/imputers/*.joblib
   │                       └► Postgres secom_production (train_prepared, train_imputed_<s>, imputation_strategy, view train_selected)
   ▼
impute_cv.py: downstream CV ──► registry (production --record-cv) ──► production --select <s>
```

## Commands

```bash
python -m pip install -e ".[dev]"                                  # setup
python -m secom_analysis.cli --data-dir data/raw --output-dir reports
python -m secom_analysis.health --data-dir data/raw --output-dir reports/health
scripts/db.sh init | start | stop | status | psql                  # Postgres on localhost:5433
python -m secom_analysis.staging --data-dir data/raw
python -m secom_analysis.impute
python -m secom_analysis.impute_cv                                 # ~23 min
python -m secom_analysis.production --record-cv
python -m secom_analysis.production --select <strategy>
python -m secom_analysis.selection                                 # ~13 min, Boruta
python -m secom_analysis.pipeline                                  # ~8 min, balancers + confusion matrices
python -m pytest -q                                                # 33 passed
```

## Key findings so far

- **Class imbalance:** about 6.6% fails. The target is coded −1 pass / 1 fail (D-013).
- **Reduction on train:** 116 zero-variance columns were removed, then 24 more above 60% missing, which leaves 450 features (D-016, D-017).
  Every threshold from 49.1% to 61.3% gives the same result.
- **Correlation:** 608 pairs have |r| > 0.8. They are only visualised for now, and removal is left to Boruta (D-018).
- **Outliers:** 3σ capping at the train bounds (D-019).
- **Split:** the stratified hold-out is kept. A single 80/20 split varies by about ±0.07 PR-AUC, and time-ordered
  validation scores about 0.07 PR-AUC lower because the defect rate drifts (D-026, D-028).
- **Imputation:** no strategy beats median in downstream CV (D-029). The kNN vs median gap is about 0.006 PR-AUC and
  is not significant.

## Status by phase (from build_plan.md)

| Phase | Topic | Status |
|---|---|---|
| A | Data understanding | Mostly done. The script still needs to download the data at runtime |
| B | Stratified split, test sealed | Done |
| C | Rough reduction | Done. Duplicate check on X_train found 0 duplicates (D-033) |
| D | Outliers (3σ) | Done. Switchable `OutlierTreatment` (cap / remove / none) inside the pipeline (D-035) |
| E | Imputation | Done. Official imputer is k-NN, k = 5, scaled (D-032). First RF: training accuracy 1.000 vs CV 0.934, recall 0 (D-033) |
| F | Boruta feature selection | Done. 450 → 13 features with 10-seed stability (D-033). Theory text still to go into the notebook |
| G | Normality + min-max scaling | Done. No transform for RF; min-max after Boruta (D-035) |
| H | Imbalance (ROSE-style / SMOTE / ADASYN / class weights) | Done. Compared under CV (D-035) |
| I | `imblearn` pipeline | Done. `build_pipeline`, checks pass (D-035). Q&A 10.6 still to go into the notebook |
| J | Models and grid search | To do |
| K | Evaluation and cost-based threshold | Partial. First confusion matrices at threshold 0.5 (D-035). Threshold choice to do |
| L | Final test evaluation | To do |
| M | Wrap-up, reproducibility, deliverables | To do |

## Open decisions for the team

- Cost weights: 100/400 (primary) or €200/€5000 (sensitivity)?
- Is Boruta required (the plan says yes, with a target of 15–20 features)?

## Deliverables still to build

- A single end-to-end script that downloads the data, installs missing packages, and has a header comment on every function.
- The scientific notebook (Google Docs template, CRISP-DM structure, SIP and Q&A tables).
- The presentation (MPMD master, PDF sent before the talk, one big-picture pipeline diagram).
- Outputs reorganised into `results/figures/` and `results/tables/`.

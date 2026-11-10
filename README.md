# SECOM Data Mining Project

Scaffold for the SECOM semiconductor manufacturing case study. The project follows the
Advanced CRISP-DM workflow and keeps the held-out test set isolated from preprocessing
decisions and model fitting.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Download the SECOM files from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/179/secom)
and place them in `data/raw/` as:

- `secom.data`
- `secom.names`
- `secom_labels.data`

## Usage

Run the initial descriptive-statistics and stratified-split workflow:

```bash
python -m secom_analysis.cli --data-dir data/raw --output-dir reports
```

The command writes to `reports/`: `descriptive_report.md` (shape, dtypes, nulls, unique
values, value ranges), `column_summary.csv` (one row per column), `dataset_overview.csv`,
`descriptive_statistics.json`, `eda_overview.png` and `missingness_histogram.png`. Raw data,
generated reports, notebooks checkpoints, and virtual environments are ignored by Git.

## Local PostgreSQL & raw staging

A project-local PostgreSQL 16 cluster lives in `.pgdata/` (git-ignored, no sudo needed) and
listens on `localhost:5433`. The connection string is written to `.env` as
`SECOM_DATABASE_URL`.

```bash
scripts/db.sh init      # one-time: create cluster, `secom` database and .env
scripts/db.sh start     # after a reboot (also: stop, status)
scripts/db.sh psql      # interactive psql on the `secom` database
python -m secom_analysis.staging --data-dir data/raw   # (re)load raw files
```

Schema `secom_raw_staging` holds the untouched source data (re-running truncates and reloads
in one transaction):

| table | contents |
|---|---|
| `measurements` | one row per `secom.data` line: `source_line`, `feature_000`..`feature_589` (`NaN` → `NULL`) |
| `labels` | `source_line`, raw `label` (-1 pass / 1 fail), `timestamp_raw`, parsed `test_timestamp` |
| `source_files` | file name, SHA-256, size, line count; full text of `secom.names` |
| `load_log` | one row per load run |

Join measurements and labels on `source_line`.

## Data health & descriptive report

Profile and health-check every dataset (raw files, labels, full feature matrix, train and
test splits, and train/test drift):

```bash
python -m secom_analysis.health --data-dir data/raw --output-dir reports/health
```

The loader also finds the files in a single sub-folder (e.g. `data/raw/secom (1)/`).
Outputs in `reports/health/`:

- `descriptive_report.html` — standalone visual report (open in a browser)
- `health_report.md` / `health_summary.json` / `health_checks.csv` — every check with PASS/WARN/FAIL
- `features_profile.csv`, `train_profile.csv`, `test_profile.csv` — per-feature descriptive statistics
- `split_drift_profile.csv` — per-feature KS test and missingness gap, train vs test

The command exits non-zero on any FAIL (add `--fail-on-warn` to also fail on WARN).

## Imputation candidates & production schema

```bash
python -m secom_analysis.impute                       # fit 5 strategies on train, compare, load DB
python -m secom_analysis.production --select <name>   # later: promote one strategy
```

Strategies: `median`, `median_flags`, `knn`, `mice_bayesridge`, `mice_extratrees`, all fitted on
the training split only (the test set is never read). Outputs:

- `data/processed/imputed/train_<strategy>.csv`: one complete dataset per strategy
- `models/imputers/<strategy>.joblib`: fitted imputers, to transform the test set at final evaluation
- `reports/imputation/`: `imputation_report.md`, `comparison.csv`, `imputation_comparison.png`,
  `imputed_distributions.png`, per-column `consistency_<strategy>.csv`, and the visual gallery
  `imputation_visuals.md` with five charts in `figures/` (redraw: `python -m secom_analysis.impute_viz`)

Postgres layers: `secom_raw_staging` (untouched source) → `secom_production`
(`train_prepared`, `train_imputed_<strategy>`, registry `imputation_strategy`, and the view
`train_selected` once a strategy is chosen). All tables join on `source_line`.

### Choosing a strategy (downstream CV)

```bash
python -m secom_analysis.impute_cv                 # ~23 min: 3x5-fold CV, imputer fitted per fold
python -m secom_analysis.production --record-cv    # store CV results in the registry
```

Outputs in `reports/imputation_cv/` (`cv_report.md`, `cv_pr_auc.png`, fold scores, time-ordered check).

## Notebook & scientific diary

- `notebooks/01_data_understanding_and_preparation.ipynb`: dataset background, target coding,
  missing-value audit, stratified 80/20 split (test set sealed in `data/processed/`), and rough
  feature reduction (zero variance, more than 60% missing, correlation heatmap, 3σ outlier
  capping), with a Step Impact Protocol (SIP) table after each step. Use the `SECOM (.venv)` kernel.
- `docs/scientific_diary.md`: append-only log of every change and decision (why, what
  happened, next steps), with a decision index and open items. **Add an entry with every change.**

## Project workflow

1. **Week 1:** load the data and calculate descriptive statistics.
2. **Week 2:** identify missing values and constant features, create a stratified 80/20
   train/test split, inspect missingness, visualize correlations, and document outlier
   decisions.
3. **Week 3:** add imputation and a baseline random-forest model using training data only.
4. **Week 4:** add Boruta-based feature selection, balancing, and defect-focused evaluation.

The test set is created once and saved separately. Do not use it to choose thresholds,
features, imputers, or model parameters.

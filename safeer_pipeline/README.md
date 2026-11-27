# Leakage-free pipeline and cost-based threshold (Safeer)

Covers T13 and T15–T17 (baseline model, Boruta, SMOTE, confusion matrix and metrics) and a first version of T24 (cost-based optimisation). Course methods only.

## What it does
- **Same data as master.** `step1` downloads SECOM from UCI at runtime, repeats master's stratified 80/20 split (seed 42) and checks the training file against master's SHA-256. The 20% test set is deleted straight away and never used.
- **The team pipeline as one `imblearn` Pipeline** (`team_pipeline.py`): drop constant and >60%-missing columns → 3σ capping → scaled kNN imputation (k=5, distance-weighted) → Boruta (confirmed + tentative) → min-max scaling → SMOTE → random forest (500 trees).
- **Every step is refitted inside each fold**, including Boruta and the column drops, so no scored wafer influences any choice. Boruta keeps 12–26 features depending on the fold (18, 15, 12, 26, 14), which matches notebook 02's per-fold counts.
- **Threshold chosen by cost, not fixed at 0.5** (`benchmark_core.py`): inside each outer fold, an inner 4-fold CV on the training rows only picks the threshold with the lowest cost (100 × false alarms + 400 × missed defects). The held-out fold is then scored once with that threshold.

## Results (stratified 5-fold CV on the 1,253 training wafers, 83 defective)

| Model | Total cost | Defects caught (of 83) | False alarms | ROC-AUC | PR-AUC | Thresholds chosen |
|---|---:|---:|---:|---|---|---|
| Always pass | 33,200 | 0 | 0 | 0.50 | 0.07 | – |
| Team pipeline + cost threshold | **32,300** | 5 | 11 | 0.69 (SD 0.08) | 0.17 (SD 0.03) | 0.54–0.95 |

Per-fold numbers: `results/team_pipeline_folds.csv`; Boruta features per fold: `results/team_pipeline_features.csv`.

## What this means for the presentation
- At the default 0.5 threshold the leakage-free pipeline does **not** beat always-pass (notebook 02: SMOTE 33,400, ROSE-style 34,900). Tuning the threshold on cost is what gets it below 33,200.
- The margin is small (900) and the model catches only 5 of 83 defects: the chosen thresholds are high, so it flags very few wafers.
- With these costs, flagging a wafer only pays off when the probability of a defect is above 100 / (100 + 400) = 20%. SMOTE trains on 50% defects, so its probabilities are too high and the tuned threshold drifts upward. Next steps for T24: compare against class weights (no synthetic rows) and against calibrated probabilities.
- A feature set chosen once on all training rows and then cross-validated on the same rows looks better than it is (notebook 02 §4.6: PR-AUC 0.224 fixed vs 0.162 refitted per fold). Any feature-count comparison (13 vs 25) should be redone with Boruta inside the folds.

## Run
```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python run_pipeline.py
```
`python src/step3_team_pipeline.py --check` reproduces notebook 02's Boruta counts on folds 0–3 first.

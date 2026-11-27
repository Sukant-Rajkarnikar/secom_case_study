"""Step 3: score the team's course pipeline with the shared protocol.

Purpose: leakage-free cross-validation of the team pipeline with a cost-based threshold.
Inputs:  data/processed from step 1.
Outputs: results/team_pipeline_folds.csv, results/team_pipeline_features.csv, printed summary.

Run with --check first: it reproduces the team's own Boruta feature counts on folds 0-3
(notebook 02 printed 18, 15, 12, 26), to prove this rebuild behaves like the team code.
"""

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_core import RESULTS, evaluate, hardware, load_training_data, summarise
from team_pipeline import build_team_pipeline, ColumnDropper, MANUAL_DECISIONS

TEAM_FEATURE_COUNTS = [18, 15, 12, 26]   # notebook 02, section 8, folds 1-4 (0-based 0-3)


def check_against_team():
    """Purpose: rebuild the team's exact setup (columns dropped once on all training rows,
    then capping, kNN and Boruta fitted per fold) and compare Boruta's feature counts."""
    X, y, fold = load_training_data()
    X = ColumnDropper().fit_transform(X)
    print(f"Columns after the team's drops on all training rows: {X.shape[1]} (team: 450)")
    front = build_team_pipeline()[1:4]          # outliers, imputer, Boruta
    for k, expected in enumerate(TEAM_FEATURE_COUNTS):
        t0 = time.perf_counter()
        n = len(front.fit(X[fold != k], y[fold != k])[-1].selected_)
        print(f"  fold {k}: {n} features (team: {expected}) "
              f"{'MATCH' if n == expected else 'DIFFERENT'}  {time.perf_counter() - t0:.0f}s")


def main():
    print(f"Hardware: {hardware()}")
    print("Team pipeline (inner threshold search: 4 fits per fold, then 1 final fit):")
    result = evaluate("team_pipeline", build_team_pipeline())
    means, totals = summarise(result)
    print("\nMean (SD) over 5 folds:")
    print(means.round(3).to_string())
    print("\nSummed over all 1,253 training wafers:")
    print(totals.round(1).to_string())
    print(f"\nManual decisions encoded: {len(MANUAL_DECISIONS)}")

    # Which sensors Boruta picked in each outer fold (refit on that fold's training rows).
    X, y, fold = load_training_data()
    rows = []
    for k in range(5):
        fitted = build_team_pipeline()[:4].fit(X[fold != k], y[fold != k])
        rows.append({"fold": k, "n_features": len(fitted[-1].selected_),
                     "features": " ".join(fitted[-1].selected_)})
    pd.DataFrame(rows).to_csv(RESULTS / "team_pipeline_features.csv", index=False)
    print("Features per outer fold:", [r["n_features"] for r in rows])


if __name__ == "__main__":
    check_against_team() if "--check" in sys.argv else main()

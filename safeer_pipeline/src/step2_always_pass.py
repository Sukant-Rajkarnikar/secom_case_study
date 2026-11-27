"""Step 2: the floor. A "model" that calls every wafer good.

Purpose: score the always-pass rule with the shared protocol, so every real model has a
         number it must beat. It also tests benchmark_core.py on a case whose answer is known.
Inputs:  data/processed from step 1.
Outputs: results/always_pass_folds.csv and a printed summary.
Expected: recall 0, no false alarms, cost = 400 x the defective wafers in each fold.
"""

import sys
from pathlib import Path

from sklearn.dummy import DummyClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))   # find benchmark_core.py next to us
from benchmark_core import evaluate, hardware, summarise

# DECISION: strategy="constant", constant=0 gives every wafer a defect probability of 0,
# so no threshold ever flags one: literally "always pass". No training data is used.
always_pass = DummyClassifier(strategy="constant", constant=0)

if __name__ == "__main__":
    print(f"Hardware: {hardware()}")
    print("Always pass:")
    result = evaluate("always_pass", always_pass)
    means, totals = summarise(result)
    print("\nMean (SD) over 5 folds:")
    print(means.round(3).to_string())
    print("\nSummed over all 1,253 training wafers:")
    print(totals[["tn", "fp", "fn", "tp", "cost"]].to_string())

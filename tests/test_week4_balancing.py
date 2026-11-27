import numpy as np
import pandas as pd

from secom_analysis.week4.balancing_and_eval import (
    COST_FN,
    COST_FP,
    compute_error_metrics,
    resampling_counts,
)


def test_resampling_counts_balances_minority_class() -> None:
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(size=(100, 4)), columns=["a", "b", "c", "d"])
    y = pd.Series([0] * 90 + [1] * 10, name="defective")
    counts = resampling_counts(X, y, seed=42)
    smote_row = counts[counts["balancer"] == "SMOTE"].iloc[0]
    rose_row = counts[counts["balancer"] == "ROSE-style"].iloc[0]
    assert smote_row["good_pass_0"] == 90
    assert smote_row["defective_fail_1"] == 90
    assert rose_row["defective_fail_1"] == 90


def test_compute_error_metrics_and_asymmetric_cost() -> None:
    y_true = np.array([0, 0, 0, 1, 1])
    proba = np.array([0.1, 0.8, 0.2, 0.9, 0.3])  # TN=2, FP=1, TP=1, FN=1
    m = compute_error_metrics(y_true, proba, threshold=0.5)
    assert (m["TP"], m["FP"], m["FN"], m["TN"]) == (1, 1, 1, 2)
    assert m["cost"] == COST_FP * 1 + COST_FN * 1
    assert np.isclose(m["sensitivity"], 0.5)
    assert np.isclose(m["precision"], 0.5)

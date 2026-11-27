import numpy as np

from secom_analysis.week6.tuning import best_threshold, costs_by_threshold, honest_cv_threshold_eval


def test_best_threshold_minimises_asymmetric_cost() -> None:
    y_true = np.array([0, 0, 0, 0, 1, 1])
    proba = np.array([0.05, 0.10, 0.12, 0.25, 0.30, 0.80])
    t = best_threshold(y_true, proba, cost_fp=100, cost_fn=400)
    # Catching both positives at t in (0.25, 0.30] yields 0 FP, 0 FN => cost 0
    assert 0.25 < t <= 0.30
    assert costs_by_threshold(y_true, proba).min() == 0


def test_honest_cv_threshold_eval_returns_expected_keys() -> None:
    rng = np.random.default_rng(42)
    y_true = np.array([0] * 40 + [1] * 10)
    proba = np.where(y_true == 1, rng.uniform(0.3, 0.9, size=50), rng.uniform(0.0, 0.5, size=50))
    fold_ids = np.arange(50) % 5
    res = honest_cv_threshold_eval(y_true, proba, fold_ids)
    assert res["TP"] + res["FN"] == 10
    assert res["TN"] + res["FP"] == 40
    assert res["cost_cv_threshold"] == 100 * res["FP"] + 400 * res["FN"]

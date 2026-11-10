import numpy as np
import pytest

from secom_analysis.tuning import (
    THRESHOLDS,
    best_threshold,
    costs_by_threshold,
    cv_threshold_cost,
    make_classifier,
)


def test_costs_by_threshold_counts_both_error_types() -> None:
    y = np.array([1, 0, 0, 1])
    proba = np.array([0.9, 0.6, 0.1, 0.3])
    cost = dict(zip(THRESHOLDS, costs_by_threshold(y, proba)))
    assert cost[0.5] == 100 * 1 + 400 * 1  # FP: 0.6; FN: 0.3
    assert cost[0.2] == 100 * 1  # catches both defects, one false alarm
    assert cost[0.95] == 400 * 2  # flags nothing


def test_best_threshold_prefers_lowest_cost_then_closest_to_half() -> None:
    y = np.array([1, 0, 0, 1])
    proba = np.array([0.9, 0.6, 0.1, 0.3])
    assert best_threshold(y, proba) == pytest.approx(0.3)  # 0.11–0.30 tie at cost 100


def test_cv_threshold_cost_never_uses_the_scored_fold() -> None:
    # Fold 1 alone would want a very low threshold; the other folds dictate 0.5-ish.
    y = np.array([1, 0, 1, 0, 1, 0])
    proba = np.array([0.8, 0.2, 0.9, 0.1, 0.05, 0.01])
    fold = np.array([0, 0, 0, 0, 1, 1])
    result = cv_threshold_cost(y, proba, fold)
    assert result["FN"] >= 1  # the fold-1 defect at 0.05 is missed: no peeking
    assert result["cost"] == 100 * result["FP"] + 400 * result["FN"]


def test_make_classifier_weights_only_for_class_weight() -> None:
    assert make_classifier("random_forest", "class_weight").class_weight == "balanced"
    assert make_classifier("random_forest", "smote").class_weight is None
    with pytest.raises(ValueError):
        make_classifier("svm", "none")

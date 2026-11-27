"""Week 3: Duplicate column check, scaled k-NN imputation, and initial Random Forest baseline."""

from .impute import (
    RANDOM_STATE,
    STRATEGIES,
    STRATEGY_BY_NAME,
    MedianFlagImputer,
    ScaledImputer,
    Strategy,
    duplicate_features,
    first_model,
    knn_imputer,
    load_prepared,
    pattern_mask,
    run,
)

__all__ = [
    "RANDOM_STATE",
    "STRATEGIES",
    "STRATEGY_BY_NAME",
    "MedianFlagImputer",
    "ScaledImputer",
    "Strategy",
    "duplicate_features",
    "first_model",
    "knn_imputer",
    "load_prepared",
    "pattern_mask",
    "run",
]

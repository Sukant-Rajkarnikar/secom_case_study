import pandas as pd

from secom_analysis.data import stratified_split
from secom_analysis.eda import constant_features, remove_features_above_missing_threshold


def test_stratified_split_preserves_class_prevalence() -> None:
    features = pd.DataFrame({"a": range(40), "b": range(40, 80)})
    target = pd.Series([0] * 32 + [1] * 8)

    train_x, test_x, train_y, test_y = stratified_split(
        features, target, test_size=0.2, random_state=42
    )

    assert len(train_x) == 32
    assert len(test_x) == 8
    assert train_y.sum() == 6
    assert test_y.sum() == 2
    assert abs(train_y.mean() - target.mean()) < 0.1
    assert abs(test_y.mean() - target.mean()) < 0.1


def test_constant_and_missing_feature_reduction() -> None:
    features = pd.DataFrame(
        {"constant": [1, 1, 1], "mostly_missing": [None, None, 1], "kept": [1, 2, 3]}
    )

    assert constant_features(features) == ["constant"]
    reduced, removed = remove_features_above_missing_threshold(features, 60)
    assert removed == ["mostly_missing"]
    assert list(reduced.columns) == ["constant", "kept"]

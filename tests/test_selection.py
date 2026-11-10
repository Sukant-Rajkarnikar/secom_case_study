import numpy as np
import pandas as pd

from secom_analysis.selection import (
    BorutaSelector,
    ColumnSubset,
    duplicate_features,
    stability_selection,
)


def test_duplicate_features_respects_missing_pattern() -> None:
    frame = pd.DataFrame(
        {
            "a": [1.0, 2.0, np.nan],
            "b": [1.0, 2.0, np.nan],  # exact copy of a, NaN included
            "c": [1.0, 2.0, 3.0],  # same observed values, different NaN pattern
        }
    )
    result = duplicate_features(frame)
    assert result.to_dict("records") == [{"duplicate": "b", "copy_of": "a"}]


def test_stability_selection_keeps_features_at_threshold() -> None:
    runs = pd.DataFrame(
        {
            "s0": ["confirmed", "tentative", "rejected"],
            "s1": ["confirmed", "rejected", "rejected"],
            "s2": ["confirmed", "tentative", "confirmed"],
            "s3": ["confirmed", "rejected", "rejected"],
        },
        index=["x", "y", "z"],
    )
    table = stability_selection(runs, keep_fraction=0.5)
    assert table.loc["x", "frequency"] == 1.0
    assert table.loc["y", "kept"]  # tentative twice in four runs = 50%
    assert not table.loc["z", "kept"]


def test_boruta_selector_finds_informative_feature() -> None:
    rng = np.random.default_rng(0)
    noise = rng.normal(size=(300, 8))
    target = (rng.random(300) < 0.3).astype(int)
    signal = target * 3 + rng.normal(scale=0.5, size=300)
    features = pd.DataFrame(np.column_stack([signal, noise]), columns=[f"f{i}" for i in range(9)])
    selector = BorutaSelector(seed=0, max_iter=30).fit(features, target)
    assert selector.support_[0]
    assert list(selector.transform(features).columns)[0] == "f0"


def test_column_subset_works_on_arrays() -> None:
    values = np.arange(12).reshape(3, 4)
    assert ColumnSubset([3, 1]).fit(values).transform(values).tolist() == [[3, 1], [7, 5], [11, 9]]

import numpy as np
import pandas as pd

from secom_analysis.preparation import (
    cap_outliers,
    flag_outliers,
    load_split,
    missing_threshold_pareto,
    remove_outliers,
    save_split,
    three_sigma_bounds,
    verify_sealed,
    zero_variance_features,
)


def test_zero_variance_uses_exact_zero_and_ignores_nan() -> None:
    frame = pd.DataFrame(
        {"const": [5.0, 5.0, np.nan], "tiny": [1.0, 1.0 + 1e-9, 1.0], "normal": [1.0, 2.0, 3.0]}
    )
    variance, zero = zero_variance_features(frame)
    assert zero == ["const"]
    assert variance["tiny"] > 0


def test_missing_pareto_counts_columns_strictly_above_threshold() -> None:
    missing = pd.Series({"a": 0.0, "b": 60.0, "c": 61.3, "d": 90.0})
    pareto = missing_threshold_pareto(missing, np.array([0, 60, 90]))
    assert pareto["columns_removed"].tolist() == [3, 2, 0]


def test_three_sigma_capping_and_removal_keep_nan_and_bounds() -> None:
    values = [0.0] * 20 + [100.0, np.nan]
    frame = pd.DataFrame({"x": values})
    bounds = three_sigma_bounds(frame)
    assert flag_outliers(frame, bounds)["x"].sum() == 1
    capped = cap_outliers(frame, bounds)
    assert capped["x"].max() == bounds.loc["x", "upper"]
    assert capped["x"].isna().sum() == 1
    assert remove_outliers(frame, bounds)["x"].isna().sum() == 2


def test_sealed_split_roundtrip_and_tamper_detection(tmp_path) -> None:
    features = pd.DataFrame({"f": [1.5, np.nan]}, index=[4, 9])
    target = pd.Series([0, 1], index=[4, 9])
    path = tmp_path / "holdout.csv"
    save_split(features, target, path, seal=True)
    loaded_x, loaded_y = load_split(path)
    assert loaded_x.index.tolist() == [4, 9]
    assert loaded_y.tolist() == [0, 1]
    assert verify_sealed(path)
    path.write_text(path.read_text() + "\n")
    assert not verify_sealed(path)

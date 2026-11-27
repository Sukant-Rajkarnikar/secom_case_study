import numpy as np
import pandas as pd

from secom_analysis.week5.pipeline_and_error import (
    FrameMinMaxScaler,
    OutlierTreatment,
    build_pipeline,
)


def test_outlier_treatment_cap_and_remove() -> None:
    df = pd.DataFrame({"a": [0.0] * 20 + [100.0], "b": [1.0] * 21})
    capped = OutlierTreatment("cap", sigma=3.0).fit_transform(df)
    removed = OutlierTreatment("remove", sigma=3.0).fit_transform(df)
    assert capped["a"].max() < 100.0
    assert capped.isna().sum().sum() == 0
    assert removed["a"].isna().sum() == 1


def test_frame_minmax_scaler_keeps_names_and_bounds() -> None:
    df = pd.DataFrame({"s1": [-10.0, 0.0, 10.0], "s2": [100.0, 200.0, 300.0]})
    out = FrameMinMaxScaler((0.0, 1.0)).fit_transform(df)
    assert list(out.columns) == ["s1", "s2"]
    assert np.isclose(out.min().min(), 0.0)
    assert np.isclose(out.max().max(), 1.0)


def test_build_pipeline_fits_and_predicts_without_leaking_sampler() -> None:
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(size=(80, 5)), columns=[f"f{i}" for i in range(5)])
    y = pd.Series(([0] * 5 + [1]) * 13 + [0, 0])  # interleaved both classes in train & val
    pipe = build_pipeline(
        outlier="cap", n_neighbors=3, selector=["f0", "f1", "f2"], balancer="smote",
        model_params={"n_estimators": 20}, seed=42,
    )
    pipe.fit(X.iloc[:60], y.iloc[:60])
    assert pipe[:5].transform(X.iloc[:60]).shape[1] == 3
    proba = pipe.predict_proba(X.iloc[60:])[:, 1]
    assert len(proba) == 20
    assert np.all((proba >= 0.0) & (proba <= 1.0))

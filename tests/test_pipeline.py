import numpy as np
import pandas as pd
import pytest

from secom_analysis.pipeline import (
    BALANCERS,
    OutlierTreatment,
    build_pipeline,
    error_metrics,
    resampling_counts,
)


def _data(n: int = 200, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    features = pd.DataFrame(rng.normal(size=(n, 6)), columns=[f"f{i}" for i in range(6)])
    target = pd.Series((rng.random(n) < 0.1).astype(int))
    features.loc[target == 1, "f0"] += 2.0
    features.iloc[::7, 2] = np.nan
    return features, target


def test_outlier_treatment_modes() -> None:
    train = pd.DataFrame({"a": np.r_[np.zeros(50), 1.0, -1.0] * 1.0})
    new = pd.DataFrame({"a": [100.0, np.nan, 0.0]})
    upper = OutlierTreatment("cap").fit(train).bounds_.loc["a", "upper"]
    assert OutlierTreatment("cap").fit(train).transform(new)["a"].iloc[0] == pytest.approx(upper)
    removed = OutlierTreatment("remove").fit(train).transform(new)["a"]
    assert np.isnan(removed.iloc[0]) and np.isnan(removed.iloc[1]) and removed.iloc[2] == 0
    assert OutlierTreatment("none").fit(train).transform(new)["a"].iloc[0] == 100.0


def test_error_metrics_cost_and_rates() -> None:
    y = np.array([1, 1, 0, 0, 0])
    proba = np.array([0.9, 0.1, 0.8, 0.2, 0.1])  # TP=1, FN=1, FP=1, TN=2
    result = error_metrics(y, proba)
    assert (result["TP"], result["FN"], result["FP"], result["TN"]) == (1, 1, 1, 2)
    assert result["cost"] == 100 * 1 + 400 * 1
    assert result["sensitivity"] == 0.5 and result["specificity"] == pytest.approx(2 / 3)


@pytest.mark.parametrize("balancer", BALANCERS)
def test_pipeline_samples_only_in_fit_and_is_reproducible(balancer: str) -> None:
    features, target = _data()
    train, valid = features.iloc[:150], features.iloc[150:]
    first = build_pipeline(selector="all", balancer=balancer, n_estimators=20).fit(train, target.iloc[:150])
    proba = first.predict_proba(valid)[:, 1]
    assert len(proba) == len(valid)  # the sampler never adds rows at predict time
    again = build_pipeline(selector="all", balancer=balancer, n_estimators=20).fit(train, target.iloc[:150])
    assert np.array_equal(proba, again.predict_proba(valid)[:, 1])


def test_build_pipeline_rejects_unknown_selector() -> None:
    with pytest.raises(ValueError):
        build_pipeline(selector="Boruta")


def test_outlier_treatment_rejects_arrays() -> None:
    with pytest.raises(TypeError):
        OutlierTreatment().fit(np.zeros((3, 2)))


def test_resampling_counts_balance_the_classes() -> None:
    features, target = _data()
    complete = features.fillna(0)
    counts = resampling_counts(complete, target).set_index("balancer")
    assert counts.loc["none", "defective"] == target.sum()
    for name in ("rose", "smote"):
        assert counts.loc[name, "defective"] == counts.loc[name, "good"]

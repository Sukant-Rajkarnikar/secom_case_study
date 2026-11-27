import numpy as np
import pandas as pd

from secom_analysis.week4.feature_selection import BorutaSelector, cross_check_rankings


def test_boruta_selector_identifies_signal_and_keeps_dataframe_columns() -> None:
    rng = np.random.default_rng(42)
    n_rows = 120
    y = pd.Series(rng.binomial(1, 0.3, size=n_rows), name="defective")
    data = {f"noise_{i}": rng.normal(size=n_rows) for i in range(6)}
    data["signal_a"] = y * 3.0 + rng.normal(scale=0.5, size=n_rows)
    data["signal_b"] = -y * 2.5 + rng.normal(scale=0.5, size=n_rows)
    X = pd.DataFrame(data)

    selector = BorutaSelector(seed=42, max_iter=20, keep_tentative=True).fit(X, y)
    transformed = selector.transform(X)

    assert isinstance(transformed, pd.DataFrame)
    assert "signal_a" in selector.selected_features_
    assert "signal_b" in selector.selected_features_
    assert list(transformed.columns) == selector.selected_features_

    ranks = cross_check_rankings(X, y, selector.selected_features_, seed=42)
    assert set(["anova_rank", "rf_rank", "boruta_kept"]).issubset(ranks.columns)

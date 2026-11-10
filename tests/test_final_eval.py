import numpy as np

from secom_analysis.final_eval import bootstrap_intervals


def test_bootstrap_intervals_cover_the_point_estimate() -> None:
    rng = np.random.default_rng(0)
    y = np.r_[np.ones(20), np.zeros(280)].astype(int)
    proba = np.clip(0.3 * y + rng.normal(0.2, 0.15, len(y)), 0, 1)
    table = bootstrap_intervals(y, proba, threshold=0.35, n=300)
    point_sensitivity = ((proba >= 0.35) & (y == 1)).sum() / 20
    assert table.loc["sensitivity", "low"] <= point_sensitivity <= table.loc["sensitivity", "high"]
    assert (table["low"] <= table["high"]).all()

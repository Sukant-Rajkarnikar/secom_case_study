import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

from secom_analysis.impute import MedianFlagImputer, ScaledImputer, pattern_mask


def _frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(rng.normal(size=(40, 3)) * [1, 100, 0.01], columns=["a", "b", "c"])
    frame.loc[:9, ["a", "b"]] = np.nan  # a block missing together
    frame.loc[20, "c"] = np.nan
    return frame


def test_scaled_imputer_keeps_observed_values_and_stays_in_range() -> None:
    frame = _frame()
    out = ScaledImputer(KNNImputer(n_neighbors=3)).fit(frame).transform(frame)
    observed = frame.notna().to_numpy()
    assert not np.isnan(out).any()
    assert np.allclose(out[observed], frame.to_numpy()[observed])
    assert (out.min(axis=0) >= frame.min().to_numpy() - 1e-12).all()
    assert (out.max(axis=0) <= frame.max().to_numpy() + 1e-12).all()


def test_median_flags_share_one_flag_per_missing_block() -> None:
    frame = _frame()
    imputer = MedianFlagImputer(min_flag_pct=5.0).fit(frame)
    out = imputer.transform(frame)
    assert imputer.blocks_ == {"missing_flag_01": ["a", "b"]}  # c: 1 of 40 rows < 5%
    assert out["missing_flag_01"].sum() == 10
    assert not out.isna().any().any()


def test_pattern_mask_only_hides_observed_cells() -> None:
    frame = _frame()
    hide = pattern_mask(frame, seed=1)
    assert hide.any()
    assert not (hide & frame.isna().to_numpy()).any()

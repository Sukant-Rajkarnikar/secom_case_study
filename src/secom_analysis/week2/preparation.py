"""Data-preparation steps for the SECOM training split (CRISP-DM: Data Preparation).

Every function here is *fitted on the training split only*. Anything that must later be
applied to the test split (dropped columns, outlier bounds) is returned explicitly so it
can be saved and re-applied without peeking at the test data.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------------------
# Hold-out handling
# --------------------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_split(features: pd.DataFrame, target: pd.Series, path: Path, *, seal: bool = False) -> str:
    """Write a split to CSV with its original row id (``source_line``, 1-based).

    With ``seal=True`` a ``.sha256`` file is written next to it, so later code can prove
    the hold-out set was never modified.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = features.copy()
    frame.insert(0, "source_line", features.index + 1)
    frame["defective"] = target.loc[features.index].to_numpy()
    frame.to_csv(path, index=False, float_format="%.10g")
    digest = _sha256(path)
    if seal:
        path.with_suffix(path.suffix + ".sha256").write_text(
            f"{digest}  {path.name}\n", encoding="utf-8"
        )
    return digest


def verify_sealed(path: Path) -> bool:
    """Return True when ``path`` still matches the hash written by ``save_split(seal=True)``."""
    expected = path.with_suffix(path.suffix + ".sha256").read_text(encoding="utf-8").split()[0]
    return _sha256(path) == expected


def load_split(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Read a split written by :func:`save_split`; the index is restored to row position."""
    frame = pd.read_csv(path)
    frame.index = frame.pop("source_line") - 1
    target = frame.pop("defective").astype("int8")
    return frame, target


# --------------------------------------------------------------------------------------
# Rough feature reduction
# --------------------------------------------------------------------------------------


def zero_variance_features(features: pd.DataFrame) -> tuple[pd.Series, list[str]]:
    """Return per-column variance and the columns whose variance is *exactly* zero.

    Columns where the variance is undefined (fewer than two observed values) are also
    returned, because they carry no information either; the caller sees them in the
    variance series as NaN.
    """
    variance = features.var(axis=0, skipna=True)
    zero = variance.index[variance.eq(0) | variance.isna()].tolist()
    return variance, zero


def missing_percentage(features: pd.DataFrame) -> pd.Series:
    """Percentage of missing values per column."""
    return features.isna().mean().mul(100).rename("missing_pct")


def missing_threshold_pareto(
    missing_pct: pd.Series, thresholds: np.ndarray | None = None
) -> pd.DataFrame:
    """Columns removed / kept if every column with missing % *above* a threshold is dropped."""
    if thresholds is None:
        thresholds = np.arange(0, 101, 1)
    removed = [int(missing_pct.gt(t).sum()) for t in thresholds]
    return pd.DataFrame(
        {
            "threshold_pct": thresholds,
            "columns_removed": removed,
            "columns_kept": [len(missing_pct) - r for r in removed],
        }
    )


# --------------------------------------------------------------------------------------
# Outliers: 3-sigma rule
# --------------------------------------------------------------------------------------


def three_sigma_bounds(features: pd.DataFrame, k: float = 3.0) -> pd.DataFrame:
    """Per-column mean, standard deviation and the mean +/- k*std boundaries."""
    mean = features.mean()
    std = features.std()
    return pd.DataFrame(
        {"mean": mean, "std": std, "lower": mean - k * std, "upper": mean + k * std}
    )


def flag_outliers(features: pd.DataFrame, bounds: pd.DataFrame) -> pd.DataFrame:
    """Boolean frame: True where an observed value lies outside its column's bounds.

    Missing values are never flagged (comparisons with NaN are False).
    """
    bounds = bounds.loc[features.columns]
    return features.lt(bounds["lower"], axis=1) | features.gt(bounds["upper"], axis=1)


def cap_outliers(features: pd.DataFrame, bounds: pd.DataFrame) -> pd.DataFrame:
    """Replace values beyond the bounds by the bound itself; NaN stays NaN."""
    bounds = bounds.loc[features.columns]
    return features.clip(lower=bounds["lower"], upper=bounds["upper"], axis=1)


def remove_outliers(features: pd.DataFrame, bounds: pd.DataFrame) -> pd.DataFrame:
    """Replace values beyond the bounds by NaN (the "remove and leave gaps" option)."""
    return features.mask(flag_outliers(features, bounds))


# --------------------------------------------------------------------------------------
# Documentation helpers
# --------------------------------------------------------------------------------------


def sip_table(rows: list[tuple[str, str, str, str]]):
    """Step Impact Protocol table: option, pro/con, decision, consequence."""
    table = pd.DataFrame(
        rows,
        columns=[
            "Options / approaches",
            "Pro and con",
            "Decision made",
            "Consequence",
        ],
    )
    return (
        table.style.hide(axis="index")
        .set_properties(
            **{"text-align": "left", "white-space": "pre-wrap", "vertical-align": "top"}
        )
        .set_table_styles([{"selector": "th", "props": [("text-align", "left")]}])
    )


def save_state(path: Path, state: dict) -> None:
    """Persist the fitted preparation state (dropped columns, thresholds, file hashes)."""
    path.write_text(json.dumps(state, indent=2, default=str) + "\n", encoding="utf-8")

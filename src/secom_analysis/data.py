"""Data loading and reproducible train/test splitting for SECOM."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def load_secom(data_dir: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load SECOM measurements and labels from the UCI file layout."""
    measurements_path = data_dir / "secom.data"
    labels_path = data_dir / "secom_labels.data"
    missing = [path for path in (measurements_path, labels_path) if not path.exists()]
    if missing:
        names = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing SECOM file(s): {names}")

    features = pd.read_csv(measurements_path, sep=r"\s+", header=None, na_values=["NaN"])
    labels = pd.read_csv(labels_path, sep=r"\s+", header=None, names=["label", "timestamp"])
    if len(features) != len(labels):
        raise ValueError("SECOM measurements and labels have different row counts")

    target = labels["label"].map({-1: 0, 1: 1}).rename("defective")
    if target.isna().any():
        raise ValueError("SECOM labels must contain only -1 and 1")
    features.columns = [f"feature_{index:03d}" for index in range(features.shape[1])]
    return features, target.astype("int8")


def stratified_split(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create a reproducible stratified split while preserving defect prevalence."""
    return train_test_split(
        features,
        target,
        test_size=test_size,
        stratify=target,
        random_state=random_state,
    )

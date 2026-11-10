"""Data loading and reproducible train/test splitting for SECOM."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

MEASUREMENTS_FILE = "secom.data"
LABELS_FILE = "secom_labels.data"
TIMESTAMP_FORMAT = "%d/%m/%Y %H:%M:%S"


def resolve_data_dir(data_dir: Path) -> Path:
    """Return the directory holding the SECOM files, searching one level down if needed.

    The UCI download often unpacks into a sub-folder such as ``data/raw/secom (1)/``.
    """
    if (data_dir / MEASUREMENTS_FILE).exists():
        return data_dir
    candidates = sorted(path.parent for path in data_dir.glob(f"*/{MEASUREMENTS_FILE}"))
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        found = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(f"Multiple SECOM folders under {data_dir}: {found}")
    return data_dir


def load_labels(data_dir: Path) -> pd.DataFrame:
    """Load the raw label file: ``label`` (-1 pass / 1 fail) and parsed ``timestamp``."""
    labels_path = resolve_data_dir(data_dir) / LABELS_FILE
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing SECOM file(s): {labels_path}")
    labels = pd.read_csv(labels_path, sep=r"\s+", header=None, names=["label", "timestamp_raw"])
    labels["timestamp"] = pd.to_datetime(
        labels["timestamp_raw"], format=TIMESTAMP_FORMAT, errors="coerce"
    )
    return labels


def load_secom(data_dir: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load SECOM measurements and labels from the UCI file layout."""
    data_dir = resolve_data_dir(data_dir)
    measurements_path = data_dir / MEASUREMENTS_FILE
    labels_path = data_dir / LABELS_FILE
    missing = [path for path in (measurements_path, labels_path) if not path.exists()]
    if missing:
        names = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing SECOM file(s): {names}")

    features = pd.read_csv(measurements_path, sep=r"\s+", header=None, na_values=["NaN"])
    labels = load_labels(data_dir)
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

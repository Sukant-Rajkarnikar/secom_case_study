"""Step 1: download SECOM, rebuild the team's 80/20 split, and fix the 5 CV folds.

Purpose: give every model in the benchmark the exact same training wafers and the
         exact same folds, matching the team project on master.
Inputs:  nothing (SECOM is downloaded from UCI at runtime).
Outputs: data/processed/train_raw.csv  (1,253 training wafers, label in column "defective")
         data/processed/folds.csv      (fold number 0-4 for each training wafer)
The 20% test set is created only so the split matches master, then deleted at once.
It is never written to disk and never used.
"""

import hashlib
import ssl
import urllib.request
from pathlib import Path

import certifi
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
UCI = "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/"
FILES = ["secom.data", "secom_labels.data"]

RANDOM_STATE = 42   # course rule: one seed for split, folds, sampler and models
TEST_SIZE = 0.20
N_FOLDS = 5
# First 16 characters of the SHA-256 of master's train_raw.csv (notebook 01, section 4.4).
MASTER_TRAIN_SHA_PREFIX = "10c997519a31f314"


def download_secom():
    """Purpose: fetch the two SECOM files from UCI unless they are already here.
    Inputs: none. Outputs: files in data/raw/."""
    RAW.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        target = RAW / name
        if not target.exists():
            print(f"Downloading {name} ...")
            # certifi supplies trusted certificates (the macOS python.org build has none).
            context = ssl.create_default_context(cafile=certifi.where())
            with urllib.request.urlopen(UCI + name, context=context, timeout=60) as response:
                target.write_bytes(response.read())


def load_secom():
    """Purpose: load features and labels the same way master does.
    Inputs: data/raw files. Outputs: X (1,567 x 590, NaN = missing), y (1 = defective, 0 = pass)."""
    X = pd.read_csv(RAW / "secom.data", sep=r"\s+", header=None, na_values=["NaN"])
    labels = pd.read_csv(RAW / "secom_labels.data", sep=r"\s+", header=None,
                         names=["label", "timestamp_raw"])
    y = labels["label"].map({-1: 0, 1: 1}).rename("defective").astype("int8")
    X.columns = [f"feature_{i:03d}" for i in range(X.shape[1])]
    return X, y


def save_train(X_train, y_train, path):
    """Purpose: write the training split in master's exact CSV format, so the files can be
    compared by fingerprint. Inputs: training X, y, path. Outputs: the file's SHA-256."""
    frame = X_train.copy()
    frame.insert(0, "source_line", X_train.index + 1)   # original row number, 1-based
    frame["defective"] = y_train.loc[X_train.index].to_numpy()
    frame.to_csv(path, index=False, float_format="%.10g")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    download_secom()
    X, y = load_secom()
    print(f"Loaded {X.shape[0]:,} wafers x {X.shape[1]} sensors, "
          f"{int(y.sum())} defective ({y.mean():.2%})")

    # DECISION: stratify=y keeps the defect rate the same in train and test.
    # Same call, same seed as master, so it gives the same rows.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    del X_test, y_test, X, y   # test set gone: never saved, never looked at

    PROCESSED.mkdir(parents=True, exist_ok=True)
    digest = save_train(X_train, y_train, PROCESSED / "train_raw.csv")
    print(f"Training split: {len(y_train):,} wafers, {int(y_train.sum())} defective")
    match = digest.startswith(MASTER_TRAIN_SHA_PREFIX)
    print(f"Fingerprint {digest[:16]} vs master {MASTER_TRAIN_SHA_PREFIX}: "
          f"{'IDENTICAL' if match else 'DIFFERENT'}")

    # DECISION: one fixed set of 5 stratified folds, shared by every model, so each model
    # is scored on exactly the same wafers. Same splitter and seed as master's notebook 02.
    folds = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_id = pd.Series(-1, index=X_train.index, name="fold")
    for k, (_, valid_rows) in enumerate(folds.split(X_train, y_train)):
        fold_id.iloc[valid_rows] = k
    out = pd.DataFrame({"source_line": X_train.index + 1, "fold": fold_id.to_numpy()})
    out.to_csv(PROCESSED / "folds.csv", index=False)

    summary = (pd.DataFrame({"fold": fold_id, "defective": y_train})
               .groupby("fold")["defective"].agg(wafers="size", defective="sum"))
    print("\nWafers held out in each fold:")
    print(summary.to_string())


if __name__ == "__main__":
    main()

"""Shared scoring code: every model (always-pass and the team pipeline) is scored the same way.

Protocol, per outer fold k (5 folds from step 1):
  1. Training rows = the other 4 folds. Fold k is not touched until step 4.
  2. Threshold: inner stratified 4-fold CV on the training rows only gives out-of-fold
     probabilities; the threshold with the lowest cost on those is chosen.
  3. Refit the model on all training rows of the outer fold.
  4. Predict probabilities for fold k, apply the threshold, record metrics and time.
Because the threshold is chosen without fold k, the cost on fold k is an honest estimate.
"""

import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (average_precision_score, confusion_matrix, fbeta_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"

RANDOM_STATE = 42
COST_FP, COST_FN = 100, 400        # false alarm on a good wafer / missed defective wafer
INNER_FOLDS = 4
THRESHOLDS = np.round(np.arange(0.01, 1.00, 0.01), 2)


def load_training_data():
    """Purpose: load step 1's training split and fold numbers.
    Inputs: data/processed files. Outputs: X (DataFrame, NaN kept), y (0/1), fold (0-4)."""
    train = pd.read_csv(PROCESSED / "train_raw.csv")
    folds = pd.read_csv(PROCESSED / "folds.csv")
    assert (train["source_line"].to_numpy() == folds["source_line"].to_numpy()).all()
    y = train.pop("defective").astype(int)
    X = train.drop(columns="source_line")
    return X, y, folds["fold"].to_numpy()


def cost_at(y_true, proba, threshold):
    """Purpose: total cost 100*FP + 400*FN when wafers with proba >= threshold are flagged."""
    flagged = proba >= threshold
    fp = int((flagged & (y_true == 0)).sum())
    fn = int((~flagged & (y_true == 1)).sum())
    return COST_FP * fp + COST_FN * fn


def best_threshold(y_true, proba):
    """Purpose: threshold with the lowest cost; ties go to the one closest to 0.5 (as master).
    Inputs: labels and probabilities of training rows only. Outputs: one threshold."""
    costs = np.array([cost_at(y_true, proba, t) for t in THRESHOLDS])
    tied = THRESHOLDS[costs == costs.min()]
    return float(tied[np.argmin(np.abs(tied - 0.5))])


def inner_oof_proba(model, X, y):
    """Purpose: out-of-fold probabilities on the outer training rows, from a fresh copy of
    the model per inner fold. Used only to choose the threshold."""
    inner = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros(len(y))
    for tr, va in inner.split(X, y):
        m = clone(model).fit(X.iloc[tr], y.iloc[tr])
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
    return oof


def fold_metrics(y_true, proba, threshold):
    """Purpose: every reported metric for one held-out fold.
    Precision is NaN when no wafer is flagged (it is undefined, not 0)."""
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "recall": recall_score(y_true, pred, zero_division=0),
        "precision": precision_score(y_true, pred, zero_division=0) if pred.sum() else np.nan,
        "f2": fbeta_score(y_true, pred, beta=2, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "cost": int(COST_FP * fp + COST_FN * fn),
    }


def evaluate(name, model):
    """Purpose: run the full protocol for one model and save per-fold results.
    Inputs: a name and an unfitted scikit-learn-style model with predict_proba.
    Outputs: DataFrame with one row per fold (also written to results/<name>_folds.csv)."""
    X, y, fold = load_training_data()
    rows = []
    for k in range(5):
        tr, te = fold != k, fold == k
        X_tr, y_tr = X[tr], y[tr]

        t0 = time.perf_counter()
        threshold = best_threshold(y_tr.to_numpy(), inner_oof_proba(model, X_tr, y_tr))
        t1 = time.perf_counter()
        fitted = clone(model).fit(X_tr, y_tr)
        t2 = time.perf_counter()
        proba = fitted.predict_proba(X[te])[:, 1]
        t3 = time.perf_counter()

        row = {"model": name, "fold": k, **fold_metrics(y[te].to_numpy(), proba, threshold)}
        row.update(threshold_search_s=t1 - t0, fit_s=t2 - t1, predict_s=t3 - t2)
        rows.append(row)
        print(f"  fold {k}: threshold {threshold:.2f}  recall {row['recall']:.3f}  "
              f"cost {row['cost']:,}  fit {row['fit_s']:.1f}s")

    result = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    result.to_csv(RESULTS / f"{name}_folds.csv", index=False)
    return result


def summarise(result):
    """Purpose: one summary line per model: mean and SD of fold metrics, summed counts and cost."""
    means = result[["recall", "precision", "f2", "roc_auc", "pr_auc"]].agg(["mean", "std"])
    totals = result[["tn", "fp", "fn", "tp", "cost", "fit_s", "predict_s",
                     "threshold_search_s"]].sum()
    return means, totals


def hardware():
    """Purpose: record where timings were measured, since they depend on the machine."""
    return f"{platform.platform()} | {platform.processor() or platform.machine()} | " \
           f"Python {platform.python_version()}"

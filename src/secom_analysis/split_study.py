"""Compare train/test split and validation schemes on the SECOM training split (D-028).

Question: how much does the *evaluation number* depend on how the data is split?

Every scheme scores the same fixed pipeline, so differences come from the split alone:

    FoldPreparation (zero-variance drop, >60% missing drop, 3σ capping; fitted per fold)
    -> median imputation -> standardisation -> classifier

Only ``data/processed/train_raw.csv`` (1,253 wafers, 83 defective) is used. The sealed
test set is never read. Inside this study the training split plays the role of "all data".

Schemes:

* ``holdout_stratified``   - 80/20 stratified hold-out, one per seed (the project's method)
* ``holdout_random``       - 80/20 hold-out without stratification, one per seed
* ``cv5_stratified``       - stratified 5-fold CV, one estimate (mean of folds) per repeat
* ``time_holdout``         - train on the earliest 80% by timestamp, test on the latest 20%
* ``time_forward``         - forward chaining (expanding window, 5 folds)
* ``leave_one_month_out``  - each calendar month is the test set once
* ``leaky_preparation``    - stratified 5-fold CV, but preparation, imputation and scaling
                             fitted on all rows first (the papers' order, D-022)
* ``leaky_oversampling``   - minority duplicated ~24x *before* a stratified 80/20 split
                             (the papers' "boosting", D-022)

Headline metric: PR-AUC (average precision); chance level = defect rate (~0.066).
ROC-AUC is reported as well.

Run with::

    python -m secom_analysis.split_study               # logistic regression + random forest
    python -m secom_analysis.split_study --quick       # fewer seeds, for a smoke test
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import (
    StratifiedKFold,
    TimeSeriesSplit,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import load_labels
from .preparation import (
    cap_outliers,
    load_split,
    missing_percentage,
    three_sigma_bounds,
    zero_variance_features,
)

ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = ROOT / "data" / "processed" / "train_raw.csv"
RAW_DIR = ROOT / "data" / "raw"
OUTPUT_DIR = ROOT / "reports" / "split_study"
MISSING_THRESHOLD_PCT = 60.0
OVERSAMPLE_FACTOR = 24  # D-022: the papers duplicate the fail class about 24x

BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#7b7a75", "#ecebe7"

Splits = Iterator[tuple[np.ndarray, np.ndarray]]


class FoldPreparation(BaseEstimator, TransformerMixin):
    """Notebook-01 preparation (D-016, D-017, D-019), refitted on whatever rows it sees."""

    def __init__(self, missing_threshold_pct: float = MISSING_THRESHOLD_PCT):
        self.missing_threshold_pct = missing_threshold_pct

    def fit(self, features: pd.DataFrame, target: pd.Series | None = None):
        _, zero = zero_variance_features(features)
        missing = missing_percentage(features)
        dropped = set(zero) | set(missing.index[missing.gt(self.missing_threshold_pct)])
        self.columns_ = [column for column in features.columns if column not in dropped]
        self.bounds_ = three_sigma_bounds(features[self.columns_])
        return self

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        return cap_outliers(features[self.columns_], self.bounds_)


def models(random_state: int = 42) -> dict[str, BaseEstimator]:
    return {
        "logreg": LogisticRegression(
            C=0.01, class_weight="balanced", max_iter=5000, random_state=random_state
        ),
        "forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            n_jobs=4,  # n_jobs=-1 ran out of memory on a 32 GB machine
            random_state=random_state,
        ),
    }


def pipeline(model: BaseEstimator) -> Pipeline:
    return Pipeline(
        [
            ("prepare", FoldPreparation()),
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", clone(model)),
        ]
    )


def score(model: BaseEstimator, x_train, y_train, x_test, y_test) -> dict[str, float]:
    fitted = clone(model).fit(x_train, y_train)
    probability = fitted.predict_proba(x_test)[:, 1]
    return {
        "pr_auc": average_precision_score(y_test, probability),
        "roc_auc": roc_auc_score(y_test, probability),
        "test_rows": len(y_test),
        "test_defective": int(np.sum(y_test)),
        "test_defect_rate": float(np.mean(y_test)),
    }


# --------------------------------------------------------------------------------------
# Split generators: each yields (estimate_id, fold_id, train_index, test_index)
# --------------------------------------------------------------------------------------


def holdouts(target: pd.Series, seeds: range, *, stratify: bool):
    rows = np.arange(len(target))
    for seed in seeds:
        train, test = train_test_split(
            rows, test_size=0.2, random_state=seed, stratify=target if stratify else None
        )
        yield seed, 0, train, test


def repeated_cv(target: pd.Series, repeats: range, folds: int = 5):
    for repeat in repeats:
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=repeat)
        for fold, (train, test) in enumerate(splitter.split(np.zeros(len(target)), target)):
            yield repeat, fold, train, test


def time_holdout(order: np.ndarray):
    cut = int(round(len(order) * 0.8))
    yield 0, 0, order[:cut], order[cut:]


def time_forward(order: np.ndarray, folds: int = 5):
    for fold, (train, test) in enumerate(TimeSeriesSplit(n_splits=folds).split(order)):
        yield 0, fold, order[train], order[test]


def leave_one_month_out(month: pd.Series):
    for fold, value in enumerate(sorted(month.unique())):
        test = np.flatnonzero(month.eq(value).to_numpy())
        train = np.flatnonzero(month.ne(value).to_numpy())
        yield 0, fold, train, test


# --------------------------------------------------------------------------------------
# Study
# --------------------------------------------------------------------------------------


def load_study_data() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Training features, target and timestamp (aligned by ``source_line``)."""
    features, target = load_split(TRAIN_PATH)
    labels = load_labels(RAW_DIR)
    timestamp = labels["timestamp"].loc[features.index].rename("timestamp")
    if timestamp.isna().any():
        raise ValueError("Every training wafer needs a parsed timestamp for the time schemes")
    return features, target, timestamp


def run(seeds: int = 100, repeats: int = 20, model_names: tuple[str, ...] = ("logreg", "forest")):
    features, target, timestamp = load_study_data()
    order = np.argsort(timestamp.to_numpy(), kind="stable")
    month = timestamp.dt.to_period("M").astype(str).reset_index(drop=True)

    # The "leaky" variant: preparation fitted on every row before any split (D-022).
    leaky_matrix = Pipeline(
        [
            ("prepare", FoldPreparation()),
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    ).fit_transform(features)

    rng = np.random.default_rng(42)
    positives = np.flatnonzero(target.to_numpy() == 1)
    duplicated = np.concatenate([np.arange(len(target))] + [positives] * (OVERSAMPLE_FACTOR - 1))
    rng.shuffle(duplicated)
    oversampled_target = target.iloc[duplicated].reset_index(drop=True)

    x_raw = features.reset_index(drop=True)
    y = target.reset_index(drop=True)
    schemes: list[tuple[str, Callable[[], Iterator], str]] = [
        ("holdout_stratified", lambda: holdouts(y, range(seeds), stratify=True), "honest"),
        ("holdout_random", lambda: holdouts(y, range(seeds), stratify=False), "honest"),
        ("cv5_stratified", lambda: repeated_cv(y, range(repeats)), "honest"),
        ("time_holdout", lambda: time_holdout(order), "time"),
        ("time_forward", lambda: time_forward(order), "time"),
        ("leave_one_month_out", lambda: leave_one_month_out(month), "time"),
        ("leaky_preparation", lambda: repeated_cv(y, range(repeats)), "leaky"),
        (
            "leaky_oversampling",
            lambda: holdouts(oversampled_target, range(seeds), stratify=True),
            "leaky",
        ),
    ]

    records = []
    for model_name in model_names:
        model = models()[model_name]
        for scheme, splits, family in schemes:
            started = time.perf_counter()
            for estimate, fold, train, test in splits():
                if scheme == "leaky_preparation":
                    xs, ys, estimator = leaky_matrix, y.to_numpy(), clone(model)
                    x_train, x_test = xs[train], xs[test]
                elif scheme == "leaky_oversampling":
                    rows = duplicated
                    x_train, x_test = x_raw.iloc[rows[train]], x_raw.iloc[rows[test]]
                    ys, estimator = oversampled_target.to_numpy(), pipeline(model)
                else:
                    x_train, x_test = x_raw.iloc[train], x_raw.iloc[test]
                    ys, estimator = y.to_numpy(), pipeline(model)
                result = score(estimator, x_train, ys[train], x_test, ys[test])
                if scheme == "leaky_oversampling":
                    # Copies of the same wafer on both sides of the split.
                    shared = np.intersect1d(duplicated[train], duplicated[test])
                    result["test_wafers_also_in_train"] = len(shared)
                if family == "time":
                    stamps = timestamp.iloc[test]
                    result["test_period"] = f"{stamps.min():%Y-%m-%d} – {stamps.max():%Y-%m-%d}"
                records.append(
                    {
                        "model": model_name,
                        "scheme": scheme,
                        "family": family,
                        "estimate": estimate,
                        "fold": fold,
                        **result,
                    }
                )
            print(f"{model_name:7s} {scheme:22s} {time.perf_counter() - started:6.1f}s", flush=True)
    return pd.DataFrame.from_records(records)


def summarise(scores: pd.DataFrame) -> pd.DataFrame:
    """One PR-AUC/ROC-AUC estimate per (scheme, estimate) = mean over its folds, then spread."""
    per_estimate = (
        scores.groupby(["model", "scheme", "family", "estimate"], sort=False)
        .agg(
            pr_auc=("pr_auc", "mean"),
            roc_auc=("roc_auc", "mean"),
            test_defective=("test_defective", "sum"),
        )
        .reset_index()
    )
    summary = (
        per_estimate.groupby(["model", "scheme", "family"], sort=False)
        .agg(
            estimates=("pr_auc", "size"),
            pr_auc_mean=("pr_auc", "mean"),
            pr_auc_sd=("pr_auc", "std"),
            pr_auc_p05=("pr_auc", lambda s: s.quantile(0.05)),
            pr_auc_p95=("pr_auc", lambda s: s.quantile(0.95)),
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_sd=("roc_auc", "std"),
            defective_in_test_min=("test_defective", "min"),
            defective_in_test_max=("test_defective", "max"),
        )
        .reset_index()
    )
    reference = summary.query("scheme == 'cv5_stratified'").set_index("model")["pr_auc_mean"]
    summary["pr_auc_vs_cv5"] = summary["pr_auc_mean"] - summary["model"].map(reference)
    # Fold-level spread for the single-run schemes (time-based ones have one estimate).
    fold_sd = scores.groupby(["model", "scheme"], sort=False)["pr_auc"].std()
    summary["pr_auc_fold_sd"] = [
        fold_sd.get((m, s)) for m, s in zip(summary["model"], summary["scheme"])
    ]
    return summary, per_estimate


LABELS = {
    "holdout_stratified": "Stratified 80/20 hold-out\n(current method; one seed = one dot)",
    "holdout_random": "Random 80/20 hold-out\n(no stratification)",
    "cv5_stratified": "Stratified 5-fold CV\n(one repeat = one dot)",
    "time_holdout": "Time-ordered hold-out\n(earliest 80% → latest 20%)",
    "time_forward": "Forward chaining\n(5 expanding folds, mean)",
    "leave_one_month_out": "Leave one month out\n(mean of months)",
    "leaky_preparation": "CV, preparation fitted\nbefore splitting (leaky)",
    "leaky_oversampling": "Fail class duplicated 24×\nbefore splitting (leaky)",
}


def save_figure(per_estimate: pd.DataFrame, scores: pd.DataFrame, path: Path) -> None:
    model_list = list(per_estimate["model"].unique())
    order = list(LABELS)
    figure, axes = plt.subplots(
        1, len(model_list), figsize=(7.2 * len(model_list), 6.6), sharey=True
    )
    axes = np.atleast_1d(axes)
    rng = np.random.default_rng(0)
    for axis, model_name in zip(axes, model_list):
        data = per_estimate[per_estimate["model"] == model_name]
        reference = data.loc[data["scheme"] == "cv5_stratified", "pr_auc"].mean()
        for position, scheme in enumerate(order):
            values = data.loc[data["scheme"] == scheme, "pr_auc"].to_numpy()
            y_positions = (
                position + rng.uniform(-0.18, 0.18, len(values)) if len(values) > 1 else [position]
            )
            color = ORANGE if scheme.startswith("leaky") else BLUE
            axis.scatter(
                values,
                y_positions,
                s=14 if len(values) > 1 else 60,
                color=color,
                alpha=0.55 if len(values) > 1 else 1,
                linewidths=0,
            )
            if scheme in {"time_forward", "leave_one_month_out"}:
                folds = scores[(scores["model"] == model_name) & (scores["scheme"] == scheme)][
                    "pr_auc"
                ]
                axis.scatter(folds, [position] * len(folds), s=22, marker="|", color=MUTED)
            if len(values) > 1:
                axis.plot(
                    [np.quantile(values, 0.05), np.quantile(values, 0.95)],
                    [position + 0.3] * 2,
                    color=INK,
                    lw=1.5,
                )
            axis.text(
                1.0,
                position,
                f"{values.mean():.3f}",
                va="center",
                ha="right",
                color=INK,
                fontsize=9,
                transform=axis.get_yaxis_transform(),
            )
        prevalence = scores["test_defect_rate"].median()
        axis.axvline(reference, color=INK, ls="--", lw=1)
        axis.axvline(prevalence, color=MUTED, ls=":", lw=1)
        axis.text(reference, -0.85, " CV reference", color=INK, fontsize=8)
        axis.text(prevalence, -0.85, " chance", color=MUTED, fontsize=8)
        axis.set_yticks(range(len(order)), [LABELS[s] for s in order], fontsize=9)
        axis.set_ylim(len(order) - 0.5, -1.1)
        axis.set_xlim(0, 1.08)
        axis.set_xlabel("PR-AUC (average precision)")
        axis.set_title(
            {
                "logreg": "Logistic regression (L2, C=0.01)",
                "forest": "Random forest (300 trees)",
            }.get(model_name, model_name),
            loc="left",
            fontsize=11,
        )
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="x", color=GRID)
        axis.set_axisbelow(True)
    figure.suptitle(
        "Same pipeline, different split: PR-AUC estimates on the training split "
        "(bar = 5–95% range; grey ticks = single folds; right column = mean)",
        fontsize=11,
        x=0.01,
        ha="left",
    )
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def monthly_table(target: pd.Series, timestamp: pd.Series) -> pd.DataFrame:
    month = timestamp.dt.to_period("M").astype(str)
    table = target.groupby(month.to_numpy()).agg(wafers="size", defective="sum")
    table["defect_rate_pct"] = (table["defective"] / table["wafers"] * 100).round(1)
    return table


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="10 seeds / 3 repeats")
    parser.add_argument("--models", nargs="+", default=["logreg", "forest"])
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)
    seeds, repeats = (10, 3) if args.quick else (100, 20)

    args.output.mkdir(parents=True, exist_ok=True)
    scores = run(seeds=seeds, repeats=repeats, model_names=tuple(args.models))
    summary, per_estimate = summarise(scores)
    scores.to_csv(args.output / "split_scores.csv", index=False)
    summary.to_csv(args.output / "split_summary.csv", index=False, float_format="%.4f")
    save_figure(per_estimate, scores, args.output / "split_comparison.png")
    _, target, timestamp = load_study_data()
    monthly_table(target, timestamp).to_csv(args.output / "monthly_defect_rate.csv")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(summary.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

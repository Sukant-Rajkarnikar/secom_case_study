"""Downstream comparison of the imputation strategies with leakage-free cross-validation.

Protocol (D-026):

* Training split only (``data/processed/train_prepared.csv``); the sealed test set is not read.
* **Repeated stratified 5-fold CV** (3 repeats = 15 folds). In every fold the imputer is fitted
  on the training folds only and then applied to the validation fold, so no statistic of the
  validation rows leaks into the fill values. The scaler and the model are fitted the same way.
* Two models, both with class weighting for the 1:14 imbalance:
  logistic regression (L2, standardised, sensitive to the fill values) and a random forest
  (the week-3 baseline).
* Metrics on the validation fold: **PR-AUC (primary**; no-skill = the defect rate),
  ROC-AUC, recall and precision at a 10% false-positive rate, MCC at that same cut.
* Every strategy sees exactly the same folds, so differences are **paired**; they are tested
  against ``median`` with the Nadeau-Bengio corrected resampled t-test (folds overlap, so the
  naive t-test is over-confident).
* Secondary **time-ordered check**: fit on the July-September wafers of the training split,
  validate on its October wafers (the defect rate drifts over time, D-026).

Note: zero-variance / >60%-missing removal and the 3σ bounds were fitted on the whole training
split (notebook 01). They are label-free and identical for every strategy, so they do not bias
the comparison between strategies.

Run with::

    python -m secom_analysis.impute_cv            # ~15 min; writes reports/imputation_cv/
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score, roc_curve
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import load_labels
from .impute import RANDOM_STATE, STRATEGIES, STRATEGY_BY_NAME, load_prepared

N_SPLITS, N_REPEATS = 5, 3
FPR_TARGET = 0.10
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#7b7a75", "#ecebe7"


def build_models() -> dict[str, BaseEstimator]:
    return {
        "logreg": make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.05, class_weight="balanced", max_iter=5000),
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            n_jobs=2,
            random_state=RANDOM_STATE,
        ),
    }


MODEL_LABELS = {"logreg": "Logistic regression", "random_forest": "Random forest"}


def _imputer(name: str) -> BaseEstimator:
    imputer = STRATEGY_BY_NAME[name].build()
    params = imputer.get_params(deep=True)
    if "imputer__estimator__n_jobs" in params:  # keep parallel folds from oversubscribing
        imputer.set_params(imputer__estimator__n_jobs=2)
    return imputer


def _as_array(out: object) -> np.ndarray:
    return np.asarray(out, dtype=float)


def score(y_true: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    """Threshold-free scores plus recall / precision / MCC at a fixed false-positive rate."""
    fpr, tpr, thresholds = roc_curve(y_true, proba)
    index = np.searchsorted(fpr, FPR_TARGET, side="right") - 1
    cut = thresholds[max(index, 0)]
    predicted = proba >= cut
    tp = int((predicted & (y_true == 1)).sum())
    return {
        "pr_auc": average_precision_score(y_true, proba),
        "roc_auc": roc_auc_score(y_true, proba),
        "recall_at_fpr10": tpr[max(index, 0)],
        "precision_at_fpr10": tp / max(int(predicted.sum()), 1),
        "mcc_at_fpr10": matthews_corrcoef(y_true, predicted),
    }


def evaluate_split(
    name: str,
    features: pd.DataFrame,
    target: pd.Series,
    train_index: np.ndarray,
    valid_index: np.ndarray,
) -> list[dict[str, object]]:
    """Fit imputer + both models on the training rows; score on the validation rows."""
    start = time.perf_counter()
    imputer = _imputer(name).fit(features.iloc[train_index])
    x_train = _as_array(imputer.transform(features.iloc[train_index]))
    x_valid = _as_array(imputer.transform(features.iloc[valid_index]))
    impute_seconds = time.perf_counter() - start
    y_train = target.iloc[train_index].to_numpy()
    y_valid = target.iloc[valid_index].to_numpy()
    rows = []
    for model_name, model in build_models().items():
        model.fit(x_train, y_train)
        proba = model.predict_proba(x_valid)[:, 1]
        rows.append(
            {
                "strategy": name,
                "model": model_name,
                "impute_seconds": impute_seconds,
                "valid_defects": int(y_valid.sum()),
                **score(y_valid, proba),
            }
        )
    return rows


def _fold_task(name, features, target, fold, train_index, valid_index):
    rows = evaluate_split(name, features, target, train_index, valid_index)
    for row in rows:
        row["fold"] = fold
    return rows


def cross_validate(
    features: pd.DataFrame, target: pd.Series, names: list[str], n_jobs: int = 6
) -> pd.DataFrame:
    splitter = RepeatedStratifiedKFold(
        n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE
    )
    folds = list(splitter.split(features, target))
    # slowest strategy first so the parallel pool stays busy
    order = sorted(names, key=lambda n: n != "mice_extratrees")
    tasks = [
        delayed(_fold_task)(name, features, target, fold, tr, va)
        for name in order
        for fold, (tr, va) in enumerate(folds)
    ]
    results = Parallel(n_jobs=n_jobs, verbose=5)(tasks)
    return pd.DataFrame([row for rows in results for row in rows])


def corrected_ttest(differences: np.ndarray, n_train: int, n_test: int) -> tuple[float, float]:
    """Nadeau & Bengio (2003) corrected resampled t-test; returns (t, two-sided p)."""
    k = len(differences)
    mean = differences.mean()
    variance = differences.var(ddof=1)
    if variance == 0:
        return (0.0, 1.0) if mean == 0 else (np.inf * np.sign(mean), 0.0)
    t = mean / np.sqrt((1 / k + n_test / n_train) * variance)
    return float(t), float(2 * stats.t.sf(abs(t), df=k - 1))


def summarise(folds: pd.DataFrame, n_rows: int, baseline: str = "median") -> pd.DataFrame:
    n_test = n_rows // N_SPLITS
    n_train = n_rows - n_test
    metrics = ["pr_auc", "roc_auc", "recall_at_fpr10", "precision_at_fpr10", "mcc_at_fpr10"]
    records = []
    for (model, name), group in folds.groupby(["model", "strategy"]):
        group = group.sort_values("fold")
        base = folds[(folds.model == model) & (folds.strategy == baseline)].sort_values("fold")
        record = {"model": model, "strategy": name}
        for metric in metrics:
            record[f"{metric}_mean"] = group[metric].mean()
            record[f"{metric}_std"] = group[metric].std()
        diff = group["pr_auc"].to_numpy() - base["pr_auc"].to_numpy()
        t, p = corrected_ttest(diff, n_train, n_test) if name != baseline else (0.0, 1.0)
        record.update(
            {
                "pr_auc_diff_vs_median": diff.mean(),
                "pr_auc_wins_vs_median": int((diff > 0).sum()),
                "corrected_t": t,
                "corrected_p": p,
                "impute_seconds": group["impute_seconds"].mean(),
            }
        )
        records.append(record)
    return pd.DataFrame(records).set_index(["model", "strategy"])


def temporal_check(
    features: pd.DataFrame,
    target: pd.Series,
    ids: pd.Series,
    labels: pd.DataFrame,
    names: list[str],
) -> pd.DataFrame:
    """Fit on Jul-Sep wafers of the training split, validate on its October wafers."""
    stamps = labels["timestamp"].to_numpy()[ids.to_numpy() - 1]
    month = pd.DatetimeIndex(stamps).month
    train_index = np.flatnonzero(month < 10)
    valid_index = np.flatnonzero(month == 10)
    rows = Parallel(n_jobs=len(names))(
        delayed(evaluate_split)(name, features, target, train_index, valid_index) for name in names
    )
    frame = pd.DataFrame([row for group in rows for row in group])
    frame["train_rows"] = len(train_index)
    frame["train_defect_rate"] = target.iloc[train_index].mean()
    frame["valid_rows"] = len(valid_index)
    frame["valid_defect_rate"] = target.iloc[valid_index].mean()
    return frame.set_index(["model", "strategy"])


# ---------------------------------------------------------------- figure


def save_figure(folds: pd.DataFrame, summary: pd.DataFrame, prevalence: float, path: Path) -> None:
    names = [s.name for s in STRATEGIES if s.name in set(folds.strategy)]
    labels = [STRATEGY_BY_NAME[n].label for n in names]
    figure, axes = plt.subplots(1, 2, figsize=(16, 4.8), sharey=True)
    for axis, model in zip(axes, ["logreg", "random_forest"]):
        for i, name in enumerate(names):
            values = folds[(folds.model == model) & (folds.strategy == name)]["pr_auc"]
            jitter = np.random.default_rng(i).uniform(-0.12, 0.12, len(values))
            axis.scatter(values, i + jitter, s=14, color="#9ec5f4", lw=0, zorder=2)
            row = summary.loc[(model, name)]
            mean, std = row["pr_auc_mean"], row["pr_auc_std"]
            axis.errorbar(mean, i, xerr=std, fmt="o", color=BLUE, ms=8, capsize=4, lw=2, zorder=3)
            p = row["corrected_p"]
            note = (
                "baseline"
                if name == "median"
                else f"Δ {row['pr_auc_diff_vs_median']:+.3f}, p={p:.2f}"
            )
            axis.text(
                1.02,
                i,
                f"{mean:.3f}  ({note})",
                transform=axis.get_yaxis_transform(),
                ha="left",
                va="center",
                fontsize=8,
                color=INK,
            )
        axis.axvline(prevalence, color=MUTED, ls="--", lw=1)
        median_mean = summary.loc[(model, "median"), "pr_auc_mean"]
        axis.axvline(median_mean, color=ORANGE, ls=":", lw=1.5, zorder=1)
        axis.text(
            prevalence,
            len(names) - 0.45,
            f" no skill ({prevalence:.3f})",
            fontsize=8,
            color=MUTED,
            va="bottom",
        )
        axis.set_yticks(range(len(names)), labels)
        axis.set_ylim(-0.6, len(names) - 0.2)
        axis.invert_yaxis()
        axis.set_xlabel("PR-AUC on validation folds (higher is better)")
        axis.set_title(MODEL_LABELS[model], loc="left", fontsize=11)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="x", color=GRID)
        axis.set_axisbelow(True)
    figure.suptitle(
        f"Imputation strategies, downstream: {N_REPEATS}×{N_SPLITS}-fold stratified CV on the "
        "training split. Dots = folds; marker = mean ± sd; orange line = median "
        "imputation; Δ and p vs median (corrected t-test). No difference is significant.",
        fontsize=10.5,
        x=0.01,
        ha="left",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(path, dpi=150)
    plt.close(figure)


# ---------------------------------------------------------------- report


def write_report(summary: pd.DataFrame, temporal: pd.DataFrame, meta: dict, path: Path) -> None:
    def table(frame: pd.DataFrame, columns: list[str], fmt: str = ".3f") -> str:
        head = "| model | strategy | " + " | ".join(columns) + " |"
        lines = [head, "|---" * (len(columns) + 2) + "|"]
        for (model, name), row in frame[columns].iterrows():
            cells = [
                (str(int(v)) if float(v).is_integer() and abs(v) > 1 else format(v, fmt))
                if isinstance(v, float)
                else str(v)
                for v in row.to_list()
            ]
            lines.append(f"| {MODEL_LABELS[model]} | {name} | " + " | ".join(cells) + " |")
        return "\n".join(lines)

    cv_cols = [
        "pr_auc_mean",
        "pr_auc_std",
        "roc_auc_mean",
        "recall_at_fpr10_mean",
        "precision_at_fpr10_mean",
        "mcc_at_fpr10_mean",
        "pr_auc_diff_vs_median",
        "pr_auc_wins_vs_median",
        "corrected_p",
    ]
    t_cols = ["pr_auc", "roc_auc", "recall_at_fpr10", "precision_at_fpr10"]
    t0 = temporal.iloc[0]
    lines = [
        "# Imputation strategies: downstream cross-validated comparison",
        "",
        (
            f"Training split only: {meta['rows']:,} wafers ({meta['defects']} defective, "
            f"{meta['prevalence']:.2%}). {N_REPEATS}×{N_SPLITS}-fold repeated stratified CV "
            f"({meta['folds']} folds, about {meta['valid_defects_per_fold']:.0f} defective wafers "
            "per validation fold). The imputer, scaler and model are fitted inside every fold. "
            f"Run time {meta['seconds'] / 60:.1f} min."
        ),
        "",
        "## Cross-validation (mean over folds)",
        "",
        table(summary, cv_cols),
        "",
        f"- `pr_auc`: primary metric. No-skill level = defect rate ({meta['prevalence']:.3f}).",
        "- `*_at_fpr10`: the cut-off is set so that 10% of good wafers are flagged.",
        (
            "- `pr_auc_diff_vs_median`, `pr_auc_wins_vs_median` (out of "
            f"{meta['folds']} folds) and `corrected_p` (Nadeau-Bengio corrected t-test): "
            "paired comparison with median imputation."
        ),
        "",
        "## Time-ordered check (secondary)",
        "",
        (
            f"Fit on Jul-Sep wafers of the training split ({int(t0['train_rows'])} wafers, "
            f"{t0['train_defect_rate']:.1%} defective), validate on October "
            f"({int(t0['valid_rows'])} wafers, {t0['valid_defect_rate']:.1%} defective; "
            f"{int(t0['valid_defects'])} defects, so single-wafer noise is large)."
        ),
        "",
        table(temporal, t_cols),
        "",
        "Figure: `cv_pr_auc.png`. Fold-level scores: `cv_folds.csv`.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Cross-validated imputation comparison.")
    parser.add_argument("--input", type=Path, default=Path("data/processed/train_prepared.csv"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--report-out", type=Path, default=Path("reports/imputation_cv"))
    parser.add_argument("--strategies", nargs="*", default=[s.name for s in STRATEGIES])
    parser.add_argument("--n-jobs", type=int, default=6)
    args = parser.parse_args(argv)
    args.report_out.mkdir(parents=True, exist_ok=True)

    features, target, ids = load_prepared(args.input)
    start = time.perf_counter()
    folds = cross_validate(features, target, args.strategies, n_jobs=args.n_jobs)
    folds.to_csv(args.report_out / "cv_folds.csv", index=False)
    summary = summarise(folds, len(features))
    summary.to_csv(args.report_out / "cv_summary.csv")
    print("temporal check ...", flush=True)
    temporal = temporal_check(features, target, ids, load_labels(args.data_dir), args.strategies)
    temporal.to_csv(args.report_out / "temporal_check.csv")
    meta = {
        "rows": len(features),
        "defects": int(target.sum()),
        "prevalence": float(target.mean()),
        "folds": N_SPLITS * N_REPEATS,
        "valid_defects_per_fold": float(folds.groupby("fold")["valid_defects"].first().mean()),
        "seconds": time.perf_counter() - start,
        "random_state": RANDOM_STATE,
    }
    (args.report_out / "cv_run.json").write_text(json.dumps(meta, indent=2) + "\n")
    save_figure(folds, summary, meta["prevalence"], args.report_out / "cv_pr_auc.png")
    write_report(summary, temporal, meta, args.report_out / "cv_report.md")
    print(summary.round(3).to_string())
    print(temporal.round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

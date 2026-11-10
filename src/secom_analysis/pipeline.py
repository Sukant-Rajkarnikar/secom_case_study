"""Weeks 4-5 steps: switchable outlier step, normality, min-max scaling, class balancing, the full
``imblearn`` pipeline, and the first confusion matrices (manual §5, §8, §9, §10, §13).

Protocol (D-035):

* Training split only. Input is ``data/processed/train_reduced.csv`` (450 features after the
  zero-variance and 60%-missing drops, **before** 3σ capping), so the outlier step is fitted
  inside the pipeline and can be switched in the grid search (build plan D3). The sealed test set
  is never read; only its SHA-256 is checked.
* Pipeline order (manual §17.1; the conflict with §10.3 is resolved in D-035)::

      3σ outliers → k-NN imputer (k=5, scaled) → Boruta → min-max → balancer → random forest

  Samplers run only during ``fit`` (``imblearn.pipeline.Pipeline``), so synthetic rows never
  reach validation folds.
* Balancers compared under repeated stratified 5-fold CV (3 repeats) with Boruta refitted in every
  fold: none, class weights, random over-sampling with shrinkage (the Python stand-in for ROSE,
  which is R-only), SMOTE, ADASYN.
* Errors at the default threshold 0.5 with cost = 100·FP + 400·FN (manual §14). Choosing a
  threshold is phase K and is done on cross-validated predictions later.

Run with::

    python -m secom_analysis.pipeline      # ~8 min; writes reports/pipeline/
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    fbeta_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import MinMaxScaler, PowerTransformer

from .eda import _md_table
from .impute import RANDOM_STATE, ScaledImputer
from .preparation import (
    cap_outliers,
    load_split,
    remove_outliers,
    three_sigma_bounds,
    verify_sealed,
)
from .selection import BorutaSelector

COST_FP, COST_FN = 100, 400  # Type I (false alarm) and Type II (missed defect), manual §14
THRESHOLD = 0.5
N_SPLITS, N_REPEATS = 5, 3
BALANCERS = ["none", "class_weight", "rose", "smote", "adasyn"]
BALANCER_LABELS = {
    "none": "None",
    "class_weight": "Class weights",
    "rose": "ROSE-style (smoothed bootstrap)",
    "smote": "SMOTE",
    "adasyn": "ADASYN",
}
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#7b7a75", "#ecebe7"


# ---------------------------------------------------------------- building blocks


class OutlierTreatment(BaseEstimator, TransformerMixin):
    """3σ outlier step with switchable treatment (build plan D3, manual §5).

    ``method``: "cap" (clip to mean ± k·SD, the D-019 choice), "remove" (set to NaN so the imputer
    fills it), or "none". Bounds are learned in ``fit`` (training rows only) and reused unchanged.
    Input and output: DataFrame with the same columns.
    """

    def __init__(self, method: str = "cap", k: float = 3.0):
        self.method = method
        self.k = k

    def fit(self, X: pd.DataFrame, y=None) -> OutlierTreatment:
        self._check(X)
        self.bounds_ = three_sigma_bounds(X, k=self.k)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._check(X)
        if self.method == "cap":
            return cap_outliers(X, self.bounds_)
        if self.method == "remove":
            return remove_outliers(X, self.bounds_)
        return X.copy()

    def _check(self, X) -> None:
        if self.method not in {"cap", "remove", "none"}:
            raise ValueError(f"unknown outlier method: {self.method}")
        if not isinstance(X, pd.DataFrame):
            raise TypeError("OutlierTreatment needs a DataFrame (bounds are matched by column)")


class FrameImputer(BaseEstimator, TransformerMixin):
    """k-NN imputer on standardised features (D-032) that keeps column names.

    Input: DataFrame with NaN. Output: complete DataFrame, same index and columns, so the
    selected feature names stay visible after Boruta and min-max scaling.
    """

    def __init__(self, n_neighbors: int = 5):
        self.n_neighbors = n_neighbors

    def fit(self, X: pd.DataFrame, y=None) -> FrameImputer:
        self.imputer_ = ScaledImputer(
            KNNImputer(n_neighbors=self.n_neighbors, weights="distance")
        ).fit(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(self.imputer_.transform(X), index=X.index, columns=X.columns)


def make_balancer(name: str, seed: int = RANDOM_STATE):
    """Sampler for a balancing option; ``None`` for "none" and "class_weight" (no rows added)."""
    if name in {"none", "class_weight"}:
        return None
    if name == "rose":  # random over-sampling + Gaussian jitter = smoothed bootstrap (ROSE idea)
        return RandomOverSampler(shrinkage=1.0, random_state=seed)
    if name == "smote":
        return SMOTE(k_neighbors=5, random_state=seed)
    if name == "adasyn":
        return ADASYN(n_neighbors=5, random_state=seed)
    raise ValueError(f"unknown balancer: {name}")


def make_model(
    balancer: str = "smote", seed: int = RANDOM_STATE, n_estimators: int = 500
) -> RandomForestClassifier:
    """Random forest (500 trees); class-weighted only for the "class_weight" option."""
    return RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced" if balancer == "class_weight" else None,
        n_jobs=-1,
        random_state=seed,
    )


def build_pipeline(
    outlier: str = "cap",
    sigma: float = 3.0,
    n_neighbors: int = 5,
    selector: str = "boruta",
    balancer: str = "smote",
    seed: int = RANDOM_STATE,
    n_estimators: int = 500,
) -> Pipeline:
    """Full leakage-safe pipeline; every step is refitted on the rows passed to ``fit``.

    Inputs: one option per step (all switchable for the grid search): outlier treatment and its
    σ multiplier, k-NN neighbours, selector ("boruta" or "all"), balancer, forest size.
    Output: an unfitted ``imblearn`` Pipeline that takes the reduced (uncapped, NaN-containing)
    feature frame.
    """
    if selector not in {"boruta", "all"}:
        raise ValueError(f"unknown selector: {selector}")
    return Pipeline(
        [
            ("outliers", OutlierTreatment(method=outlier, k=sigma)),
            ("imputer", FrameImputer(n_neighbors=n_neighbors)),
            ("selector", BorutaSelector(seed=seed) if selector == "boruta" else "passthrough"),
            ("scaler", MinMaxScaler().set_output(transform="pandas")),
            ("balancer", make_balancer(balancer, seed) or "passthrough"),
            ("model", make_model(balancer, seed, n_estimators)),
        ]
    )


def error_metrics(y_true, proba, threshold: float = THRESHOLD) -> dict[str, float]:
    """Confusion counts and the manual's error metrics at a fixed threshold, plus cost.

    Positive class = defective wafer (1). Cost = 100·FP (Type I) + 400·FN (Type II).
    """
    y_true = np.asarray(y_true)
    pred = (np.asarray(proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "TP": int(tp),
        "FN": int(fn),
        "FP": int(fp),
        "TN": int(tn),
        "accuracy": (tp + tn) / len(y_true),
        "sensitivity": tp / max(tp + fn, 1),
        "specificity": tn / max(tn + fp, 1),
        "precision": tp / max(tp + fp, 1),
        "f1": fbeta_score(y_true, pred, beta=1, zero_division=0),
        "f2": fbeta_score(y_true, pred, beta=2, zero_division=0),
        "mcc": matthews_corrcoef(y_true, pred),
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "cost": int(COST_FP * fp + COST_FN * fn),
    }


# ---------------------------------------------------------------- analyses


def normality_table(features: pd.DataFrame) -> pd.DataFrame:
    """Per feature: skewness, excess kurtosis, Shapiro-Wilk p, before and after Yeo-Johnson.

    Input: complete (imputed) training features. Yeo-Johnson is fitted on the same rows; it is
    a diagnostic here, not part of the pipeline (random forests do not need normal inputs).
    """
    transformed = PowerTransformer(method="yeo-johnson", standardize=False).fit_transform(features)
    rows = []
    for position, column in enumerate(features.columns):
        raw = features[column].to_numpy()
        yj = transformed[:, position]
        rows.append(
            {
                "feature": column,
                "skew": stats.skew(raw),
                "kurtosis": stats.kurtosis(raw),
                "shapiro_p": stats.shapiro(raw).pvalue,
                "skew_yeo_johnson": stats.skew(yj),
                "shapiro_p_yeo_johnson": stats.shapiro(yj).pvalue,
            }
        )
    return pd.DataFrame(rows).set_index("feature")


def resampling_counts(features: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
    """Class counts before and after each sampler on the given training rows (manual §9.4)."""
    rows = []
    for name in BALANCERS:
        sampler = make_balancer(name)
        y_out = target if sampler is None else sampler.fit_resample(features, target)[1]
        counts = pd.Series(y_out).value_counts()
        rows.append(
            {
                "balancer": name,
                "good": int(counts.get(0, 0)),
                "defective": int(counts.get(1, 0)),
                "rows": len(y_out),
                "defect_share": counts.get(1, 0) / len(y_out),
            }
        )
    return pd.DataFrame(rows)


def pipeline_checks(features: pd.DataFrame, target: pd.Series, test_path: Path) -> dict:
    """Manual §10.5 checks on one stratified fold: step shapes, row counts, reproducibility.

    Output: a JSON-ready dict. The fold is the first split of the CV used in :func:`compare_balancers`.
    """
    folds = RepeatedStratifiedKFold(
        n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE
    )
    train_index, valid_index = next(folds.split(features, target))
    x_train, y_train = features.iloc[train_index], target.iloc[train_index]
    x_valid = features.iloc[valid_index]

    pipeline = build_pipeline().fit(x_train, y_train)
    shapes = {"input": list(x_train.shape)}
    data, labels = x_train, y_train
    for name, step in pipeline.steps[:-1]:
        if step == "passthrough":
            continue
        if hasattr(step, "fit_resample"):
            data, labels = clone(step).fit_resample(data, labels)
        else:
            data = step.transform(data)
        shapes[name] = list(data.shape)
    first = pipeline.predict_proba(x_valid)[:, 1]
    second = build_pipeline().fit(x_train, y_train).predict_proba(x_valid)[:, 1]
    selector = pipeline.named_steps["selector"]
    scaled = pipeline[:4].transform(x_train)
    return {
        "train_rows": len(train_index),
        "validation_rows": len(valid_index),
        "rows_shared_between_train_and_validation": len(
            set(x_train.index) & set(x_valid.index)
        ),
        "rows_after_balancer_in_fit": shapes.get("balancer", shapes["scaler"])[0],
        "predictions_returned_for_validation": len(first),
        "step_shapes_in_fit": shapes,
        "selected_features": [str(c) for c in selector.columns_[selector.support_]],
        "identical_predictions_on_refit": bool(np.array_equal(first, second)),
        "minmax_range_on_train": [float(scaled.min().min()), float(scaled.max().max())],
        "test_set_sealed": verify_sealed(test_path),
    }


def compare_balancers(features: pd.DataFrame, target: pd.Series, log=print) -> pd.DataFrame:
    """Repeated stratified 5-fold CV of every balancer with Boruta refitted in each fold.

    The expensive front (outliers → imputer → Boruta) is fitted once per fold and shared by the
    balancers; scaler, sampler and forest are fitted per balancer on the fold's training rows.
    Output: one row per (fold, balancer) with the metrics at threshold 0.5, plus out-of-fold
    probabilities in the ``proba`` column (as a list) for the pooled confusion matrices.
    """
    folds = RepeatedStratifiedKFold(
        n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE
    )
    rows = []
    for fold, (train_index, valid_index) in enumerate(folds.split(features, target)):
        start = time.time()
        x_train, y_train = features.iloc[train_index], target.iloc[train_index]
        x_valid, y_valid = features.iloc[valid_index], target.iloc[valid_index]
        front = build_pipeline()[:3].fit(x_train, y_train)
        f_train, f_valid = front.transform(x_train), front.transform(x_valid)
        for name in BALANCERS:
            back = build_pipeline(balancer=name)[3:].fit(f_train, y_train)
            proba = back.predict_proba(f_valid)[:, 1]
            rows.append(
                {
                    "fold": fold,
                    "repeat": fold // N_SPLITS,
                    "balancer": name,
                    "n_features": f_train.shape[1],
                    **error_metrics(y_valid, proba),
                    "index": list(valid_index),
                    "proba": list(proba),
                }
            )
        log(f"fold {fold + 1:2d}/{N_SPLITS * N_REPEATS}: {f_train.shape[1]} features, "
            f"{time.time() - start:.0f} s")
    return pd.DataFrame(rows)


def summarise(folds: pd.DataFrame) -> pd.DataFrame:
    """Mean and SD over folds per balancer, in the BALANCERS order."""
    metrics = ["sensitivity", "specificity", "precision", "f2", "mcc", "roc_auc", "pr_auc", "cost"]
    grouped = folds.groupby("balancer")[metrics]
    table = grouped.mean().add_suffix("_mean").join(grouped.std().add_suffix("_sd"))
    return table.loc[BALANCERS]


def pooled_confusion(folds: pd.DataFrame, target: pd.Series, repeat: int = 0) -> pd.DataFrame:
    """Out-of-fold predictions of one CV repeat (every training row predicted once) → metrics."""
    rows = []
    for name in BALANCERS:
        part = folds[(folds["balancer"] == name) & (folds["repeat"] == repeat)]
        index = np.concatenate(part["index"].to_list())
        proba = np.concatenate(part["proba"].to_list())
        rows.append({"balancer": name, **error_metrics(target.iloc[index], proba)})
    return pd.DataFrame(rows).set_index("balancer")


# ---------------------------------------------------------------- outputs


def _style(axis: plt.Axes) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color(MUTED)
    axis.tick_params(colors=INK, labelsize=9)


def save_normality_figure(all_features: pd.DataFrame, selected: pd.DataFrame, path: Path) -> None:
    """Left: |skew| of all 450 features. Right: skew of the selected features before/after YJ."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.2))
    skew = all_features["skew"].abs().clip(upper=30)
    left.hist(skew, bins=40, color=BLUE)
    left.axvline(1, color=ORANGE, linestyle="--", linewidth=1.2)
    left.set_title("|Skewness| of the 450 imputed training features (clipped at 30)", fontsize=10)
    left.set_xlabel("|skew|  (dashed: 1 = strongly skewed)")
    left.set_ylabel("features")
    positions = np.arange(len(selected))
    right.barh(positions + 0.2, selected["skew"], height=0.4, color=BLUE, label="raw")
    right.barh(positions - 0.2, selected["skew_yeo_johnson"], height=0.4, color=ORANGE,
               label="after Yeo-Johnson")
    right.set_yticks(positions, selected.index)
    right.axvline(0, color=MUTED, linewidth=0.8)
    right.set_title("Skewness of the Boruta features", fontsize=10)
    right.legend(frameon=False, fontsize=9)
    for axis in (left, right):
        _style(axis)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_confusion_figure(pooled: pd.DataFrame, path: Path) -> None:
    """One 2×2 confusion matrix per balancer (out-of-fold, CV repeat 1, threshold 0.5)."""
    fig, axes = plt.subplots(1, len(BALANCERS), figsize=(3.1 * len(BALANCERS), 3.4))
    for axis, name in zip(axes, BALANCERS):
        row = pooled.loc[name]
        matrix = np.array([[row["TN"], row["FP"]], [row["FN"], row["TP"]]])
        vmax = np.log1p(matrix.sum())
        axis.imshow(np.log1p(matrix), cmap="Blues", vmin=0, vmax=vmax)
        for (i, j), value in np.ndenumerate(matrix):
            axis.text(j, i, f"{value}", ha="center", va="center", fontsize=12,
                      color="white" if np.log1p(value) > 0.6 * vmax else INK)
        axis.set_xticks([0, 1], ["pred. good", "pred. defect"], fontsize=8)
        axis.set_yticks([0, 1], ["good", "defect"], fontsize=8)
        axis.set_title(f"{BALANCER_LABELS[name]}\ncost {int(row['cost']):,}", fontsize=9)
    fig.suptitle("Out-of-fold confusion matrices, training split, threshold 0.5", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_report(results: dict, path: Path) -> None:
    """Markdown report with the tables, SIP rows and checks."""

    def table(frame: pd.DataFrame) -> str:
        return _md_table(frame, floatfmt=".3f")

    normality = results["normality"]
    summary = results["summary"]
    pooled = results["pooled"]
    lines = [
        "# Pipeline, normality, scaling, balancing and first error metrics",
        "",
        (f"Training split only ({results['n_rows']} rows, {results['n_defects']} defective). "
        f"Generated by `python -m secom_analysis.pipeline` in {results['minutes']:.1f} min."),
        "",
        "Pipeline: `3σ cap → k-NN imputer (k=5, scaled) → Boruta → min-max → balancer → RF (500)`.",
        "",
        "## Normality (manual §8)",
        "",
        (f"- All 450 imputed features: {results['n_not_normal']} reject normality "
        f"(Shapiro-Wilk p < 0.05); {results['n_skewed']} have |skew| > 1."),
        ("- Random forests split on thresholds, so they do not need normal or scaled inputs. "
        "Yeo-Johnson is **not** added to the RF pipeline; it stays an option for logistic "
        "regression / SVM."),
        "",
        table(normality),
        "",
        "## Min-max scaling (manual §8)",
        "",
        (f"Fitted on training rows after selection; range on train = {results['checks']['minmax_range_on_train']}. "
        "Column names are kept (pandas output)."),
        "",
        "## Class balance (manual §9)",
        "",
        (f"Majority-class baseline accuracy: **{results['baseline_accuracy']:.3f}** "
        "(always predict good, recall 0)."),
        "",
        table(results["resampling"]),
        "",
        "## Balancer comparison (3×5-fold CV, Boruta refitted in each fold, threshold 0.5)",
        "",
        table(summary),
        "",
        "## Initial confusion matrices (out-of-fold, CV repeat 1)",
        "",
        table(pooled[["TP", "FN", "FP", "TN", "accuracy", "sensitivity", "specificity",
                      "precision", "f1", "f2", "cost"]]),
        "",
        "![confusion](confusion_matrices.png)",
        "",
        "## Pipeline checks (manual §10.5)",
        "",
        "```json",
        json.dumps(results["checks"], indent=2),
        "```",
    ]
    path.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/pipeline"))
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()

    features, target = load_split(args.processed_dir / "train_reduced.csv")
    selected = json.loads((args.processed_dir / "selected_features.json").read_text())[
        "selected_features"
    ]
    print(f"train_reduced: {features.shape}, {int(target.sum())} defective")

    # Normality on the capped + imputed training features (diagnostic, fitted on all train rows)
    complete = FrameImputer().fit_transform(OutlierTreatment().fit_transform(features))
    normality_all = normality_table(complete)
    normality_all.to_csv(args.output_dir / "normality_all_features.csv")
    normality_sel = normality_all.loc[selected]
    save_normality_figure(normality_all, normality_sel, args.output_dir / "normality.png")

    # Class counts before/after each sampler, on the min-max-scaled selected features
    scaled = MinMaxScaler().set_output(transform="pandas").fit_transform(complete[selected])
    resampling = resampling_counts(scaled, target)
    resampling.to_csv(args.output_dir / "resampling_counts.csv", index=False)

    checks = pipeline_checks(features, target, args.processed_dir / "test_holdout.csv")
    print(json.dumps(checks, indent=2))

    folds = compare_balancers(features, target)
    folds.drop(columns=["index", "proba"]).to_csv(args.output_dir / "cv_folds.csv", index=False)
    summary = summarise(folds)
    summary.to_csv(args.output_dir / "cv_summary.csv")
    pooled = pooled_confusion(folds, target)
    pooled.to_csv(args.output_dir / "confusion_oof.csv")
    save_confusion_figure(pooled, args.output_dir / "confusion_matrices.png")
    (args.output_dir / "pipeline_checks.json").write_text(json.dumps(checks, indent=2))

    results = {
        "n_rows": len(target),
        "n_defects": int(target.sum()),
        "minutes": (time.time() - start) / 60,
        "normality": normality_sel,
        "n_not_normal": int((normality_all["shapiro_p"] < 0.05).sum()),
        "n_skewed": int((normality_all["skew"].abs() > 1).sum()),
        "baseline_accuracy": 1 - target.mean(),
        "resampling": resampling,
        "summary": summary,
        "pooled": pooled,
        "checks": checks,
    }
    write_report(results, args.output_dir / "report.md")
    print(summary.round(3).to_string())
    print(pooled.round(3).to_string())
    print(f"done in {results['minutes']:.1f} min → {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

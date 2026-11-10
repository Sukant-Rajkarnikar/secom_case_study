"""Week 3-4 steps: duplicate check, first model, and Boruta feature selection (manual §4.3, §6, §7).

Protocol (D-032, D-033):

* Training split only (``data/processed/train_prepared.csv``: 450 features after zero-variance
  removal, the 60% missing threshold and 3σ capping). The sealed test set is never read.
* Imputation: k-NN, k = 5, on standardised features (``ScaledImputer``), the manual's method.
* **First model** (manual week 3): a default random forest. Accuracy on the rows it was trained on
  is compared with stratified cross-validation to show how optimistic training accuracy is.
* **Boruta** (manual week 4): BorutaPy around a depth-5, class-weighted random forest. A single
  run depends on the forest's seed, so Boruta runs under ``N_SEEDS`` seeds and a feature is kept
  when it is confirmed **or tentative** in at least ``KEEP_FRACTION`` of the runs.
* Cross-checks: ANOVA F-test (filter) and random-forest impurity importance (embedded), top-N
  overlap with the Boruta set.
* **§7.6 check**: repeated stratified 5-fold CV (3 repeats), same folds for every variant:
  all 450 features / the fixed Boruta set (the manual's version, slightly optimistic because the
  set was chosen on all training rows) / Boruta refitted inside every fold (leakage-free).

Run with::

    python -m secom_analysis.selection      # ~15 min; writes reports/feature_selection/
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from boruta import BorutaPy
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline

from .impute import RANDOM_STATE, STRATEGY_BY_NAME
from .impute_cv import score
from .preparation import load_split

N_SEEDS = 10
KEEP_FRACTION = 0.5
N_SPLITS, N_REPEATS = 5, 3
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#7b7a75", "#ecebe7"


# ---------------------------------------------------------------- building blocks


def duplicate_features(features: pd.DataFrame) -> pd.DataFrame:
    """Find columns that are exact copies of an earlier column.

    Input: feature frame (NaN allowed; two columns count as equal only when their missing
    cells coincide as well). Output: one row per duplicate with the column it copies.
    """
    duplicated = features.T.duplicated(keep="first")
    rows = []
    for column in features.columns[duplicated.to_numpy()]:
        original = next(c for c in features.columns if features[c].equals(features[column]))
        rows.append({"duplicate": column, "copy_of": original})
    return pd.DataFrame(rows, columns=["duplicate", "copy_of"])


def knn_imputer() -> BaseEstimator:
    """The manual's imputer: k-NN with k = 5 on standardised features (D-025, D-032)."""
    return STRATEGY_BY_NAME["knn"].build()


def boruta_forest(seed: int) -> RandomForestClassifier:
    """Random forest used inside Boruta: shallow trees and class weights for the 1:14 imbalance."""
    return RandomForestClassifier(
        max_depth=5, class_weight="balanced", n_jobs=-1, random_state=seed
    )


class BorutaSelector(BaseEstimator, TransformerMixin):
    """scikit-learn wrapper around BorutaPy that keeps confirmed and (optionally) tentative features.

    Inputs: ``fit(X, y)`` with a complete (imputed) numeric matrix. Output of ``transform``: the
    selected columns, as a DataFrame when the input was one. Usable inside a Pipeline, so it can
    be refitted in every cross-validation fold.
    """

    def __init__(self, seed: int = RANDOM_STATE, max_iter: int = 100, keep_tentative: bool = True):
        self.seed = seed
        self.max_iter = max_iter
        self.keep_tentative = keep_tentative

    def fit(self, X, y) -> BorutaSelector:
        values = np.asarray(X, dtype=float)
        self.columns_ = (
            np.asarray(X.columns) if hasattr(X, "columns") else np.arange(values.shape[1])
        )
        self.boruta_ = BorutaPy(
            boruta_forest(self.seed),
            n_estimators="auto",
            max_iter=self.max_iter,
            random_state=self.seed,
        ).fit(values, np.asarray(y))
        support = self.boruta_.support_.copy()
        if self.keep_tentative:
            support |= self.boruta_.support_weak_
        if not support.any():  # never hand an empty matrix to the model
            support[np.argmin(self.boruta_.ranking_)] = True
        self.support_ = support
        return self

    def transform(self, X):
        if hasattr(X, "iloc"):
            return X.iloc[:, self.support_]
        return np.asarray(X)[:, self.support_]


def boruta_status(features: pd.DataFrame, target: pd.Series, seed: int) -> pd.Series:
    """One Boruta run. Output: 'confirmed' / 'tentative' / 'rejected' per feature."""
    selector = BorutaSelector(seed=seed).fit(features, target)
    status = np.where(
        selector.boruta_.support_,
        "confirmed",
        np.where(selector.boruta_.support_weak_, "tentative", "rejected"),
    )
    return pd.Series(status, index=features.columns, name=f"seed_{seed}")


def stability_selection(runs: pd.DataFrame, keep_fraction: float = KEEP_FRACTION) -> pd.DataFrame:
    """Summarise several Boruta runs (columns = runs, values = status).

    Output: per feature, how often it was confirmed, tentative, or either, and whether it is kept
    (selected, confirmed or tentative, in at least ``keep_fraction`` of the runs).
    """
    n = runs.shape[1]
    table = pd.DataFrame(
        {
            "confirmed": runs.eq("confirmed").sum(axis=1),
            "tentative": runs.eq("tentative").sum(axis=1),
        }
    )
    table["selected"] = table["confirmed"] + table["tentative"]
    table["frequency"] = table["selected"] / n
    table["kept"] = table["frequency"] >= keep_fraction
    return table.sort_values(["selected", "confirmed"], ascending=False)


# ---------------------------------------------------------------- analyses


def first_model(features: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
    """Manual week 3: default random forest on the k-NN-imputed training split.

    Output: accuracy / recall / confusion counts measured on the training rows themselves
    (resubstitution) and under stratified 5-fold CV with the imputer refitted per fold.
    """
    model = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=RANDOM_STATE)
    pipeline = make_pipeline(knn_imputer(), model)
    rows = []
    fitted = clone(pipeline).fit(features, target)
    pred = fitted.predict(features)
    rows.append(_first_model_row("training rows (seen)", target, pred, None))
    folds = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    pred = np.zeros(len(target), dtype=int)
    proba = np.zeros(len(target))
    for train_index, valid_index in folds.split(features, target):
        fold = clone(pipeline).fit(features.iloc[train_index], target.iloc[train_index])
        pred[valid_index] = fold.predict(features.iloc[valid_index])
        proba[valid_index] = fold.predict_proba(features.iloc[valid_index])[:, 1]
    rows.append(_first_model_row("5-fold CV (unseen)", target, pred, proba))
    return pd.DataFrame(rows)


def _first_model_row(name: str, target: pd.Series, pred: np.ndarray, proba) -> dict:
    tn, fp, fn, tp = confusion_matrix(target, pred, labels=[0, 1]).ravel()
    row = {
        "evaluated_on": name,
        "accuracy": accuracy_score(target, pred),
        "recall": recall_score(target, pred, zero_division=0),
        "TP": tp,
        "FN": fn,
        "FP": fp,
        "TN": tn,
    }
    if proba is not None:
        metrics = score(target.to_numpy(), proba)
        row.update(pr_auc=metrics["pr_auc"], roc_auc=metrics["roc_auc"])
    return row


def cross_checks(features: pd.DataFrame, target: pd.Series, selected: list[str]) -> pd.DataFrame:
    """Filter (ANOVA F) and embedded (RF impurity) rankings; overlap of their top-N with Boruta."""
    f_scores, p_values = f_classif(features, target)
    forest = RandomForestClassifier(
        n_estimators=500, class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE
    ).fit(features, target)
    ranks = pd.DataFrame(
        {
            "anova_f": f_scores,
            "anova_p": p_values,
            "rf_importance": forest.feature_importances_,
        },
        index=features.columns,
    )
    ranks["anova_rank"] = ranks["anova_f"].rank(ascending=False, method="first").astype(int)
    ranks["rf_rank"] = ranks["rf_importance"].rank(ascending=False, method="first").astype(int)
    ranks["boruta_kept"] = ranks.index.isin(selected)
    return ranks.sort_values("rf_rank")


def compare_feature_sets(
    features: pd.DataFrame, target: pd.Series, selected: list[str]
) -> pd.DataFrame:
    """§7.6: the same model and folds on three feature sets. Output: one row per fold and variant.

    The model is a class-weighted random forest (400 trees). The imputer is refitted in every fold.
    """
    model = RandomForestClassifier(
        n_estimators=400,
        class_weight="balanced_subsample",
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    variants = {
        "all_450": make_pipeline(knn_imputer(), clone(model)),
        f"boruta_fixed_{len(selected)}": make_pipeline(
            knn_imputer(),
            ColumnSubset([features.columns.get_loc(c) for c in selected]),
            clone(model),
        ),
        "boruta_in_fold": make_pipeline(knn_imputer(), BorutaSelector(), clone(model)),
    }
    folds = RepeatedStratifiedKFold(
        n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE
    )
    rows = []
    for fold, (train_index, valid_index) in enumerate(folds.split(features, target)):
        for name, pipeline in variants.items():
            start = time.perf_counter()
            fitted = clone(pipeline).fit(features.iloc[train_index], target.iloc[train_index])
            proba = fitted.predict_proba(features.iloc[valid_index])[:, 1]
            n_features = (
                fitted[1].support_.sum()
                if name == "boruta_in_fold"
                else len(selected)
                if name.startswith("boruta_fixed")
                else features.shape[1]
            )
            rows.append(
                {
                    "variant": name,
                    "fold": fold,
                    "n_features": int(n_features),
                    "seconds": time.perf_counter() - start,
                    **score(target.iloc[valid_index].to_numpy(), proba),
                }
            )
        print(f"fold {fold + 1}/{N_SPLITS * N_REPEATS} done", flush=True)
    return pd.DataFrame(rows)


class ColumnSubset(BaseEstimator, TransformerMixin):
    """Keep fixed column positions (the Boruta set chosen on all training rows).

    Positions, not names, because the imputer before it returns a plain array.
    """

    def __init__(self, positions: list[int]):
        self.positions = positions

    def fit(self, X, y=None) -> ColumnSubset:
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.positions]


# ---------------------------------------------------------------- outputs


def save_stability_figure(stability: pd.DataFrame, n_runs: int, path: Path) -> None:
    """Horizontal bars: share of Boruta runs that selected each feature (features picked at least once)."""
    shown = stability[stability["selected"] > 0].iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(shown) + 1.2))
    colors = [BLUE if kept else MUTED for kept in shown["kept"]]
    ax.barh(shown.index, shown["frequency"], color=colors, height=0.6)
    ax.axvline(KEEP_FRACTION, color=INK, lw=1, ls="--")
    ax.text(
        KEEP_FRACTION,
        len(shown) - 0.3,
        f" keep ≥ {KEEP_FRACTION:.0%}",
        color=INK,
        fontsize=8,
        va="bottom",
    )
    ax.set_xlim(0, 1.05)
    ax.set_xlabel(f"Share of {n_runs} Boruta runs selecting the feature (confirmed or tentative)")
    ax.set_title(
        f"Boruta stability: {int(stability['kept'].sum())} features kept (blue)",
        loc="left",
        fontsize=11,
        color=INK,
    )
    ax.grid(axis="x", color=GRID)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_cv_figure(folds: pd.DataFrame, prevalence: float, path: Path) -> None:
    """Dot plot: PR-AUC per fold for each feature set, with the mean marked."""
    order = list(dict.fromkeys(folds["variant"]))
    fig, ax = plt.subplots(figsize=(7, 3.2))
    rng = np.random.default_rng(0)
    for i, name in enumerate(order):
        values = folds.loc[folds["variant"] == name, "pr_auc"]
        ax.scatter(
            values,
            i + rng.uniform(-0.12, 0.12, len(values)),
            s=24,
            color=BLUE,
            alpha=0.55,
            edgecolor="white",
            linewidth=0.5,
        )
        ax.plot([values.mean()] * 2, [i - 0.28, i + 0.28], color=INK, lw=2)
        ax.text(
            values.mean(),
            i + 0.32,
            f"mean {values.mean():.3f}",
            fontsize=8,
            color=INK,
            ha="center",
            va="bottom",
        )
    ax.axvline(prevalence, color=ORANGE, lw=1, ls="--")
    ax.text(prevalence, -0.6, " no skill", color=ORANGE, fontsize=8, va="center")
    ax.set_yticks(range(len(order)), order)
    ax.set_ylim(-0.8, len(order) - 0.2)
    ax.set_xlabel("PR-AUC per validation fold (15 folds, training split only)")
    ax.set_title("Feature sets compared (§7.6)", loc="left", fontsize=11, color=INK)
    ax.grid(axis="x", color=GRID)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _md(frame: pd.DataFrame, fmt: str = ".3f") -> str:
    def cell(value):
        return format(value, fmt) if isinstance(value, (float, np.floating)) else str(value)

    head = "| " + " | ".join(map(str, frame.columns)) + " |"
    rule = "|" + "---|" * len(frame.columns)
    body = ["| " + " | ".join(cell(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([head, rule, *body])


def write_report(
    meta: dict,
    duplicates: dict,
    first: pd.DataFrame,
    stability: pd.DataFrame,
    checks: pd.DataFrame,
    summary: pd.DataFrame,
    path: Path,
) -> None:
    """Markdown summary of every step, with the numbers the notebook and diary quote."""
    kept = stability[stability["kept"]]
    n = len(kept)
    overlap_f = int((checks["anova_rank"] <= n)[checks["boruta_kept"]].sum())
    overlap_rf = int((checks["rf_rank"] <= n)[checks["boruta_kept"]].sum())
    lines = [
        "# Feature selection report (training split only)",
        "",
        (
            f"Input: `{meta['input']}`: {meta['rows']} rows × {meta['features']} features, "
            f"{meta['defective']} defective ({meta['prevalence']:.2%}). Imputer: k-NN, k = 5, scaled."
        ),
        "",
        "## 1. Duplicate columns (§4.3)",
        "",
        *[f"- {name}: **{count}** duplicate columns" for name, count in duplicates.items()],
        "",
        "## 2. First model: default random forest (week 3)",
        "",
        _md(first),
        "",
        (
            "Training-row accuracy is measured on wafers the forest has already seen and overstates "
            "real performance; the cross-validated row is the honest estimate."
        ),
        "",
        f"## 3. Boruta stability selection ({meta['n_seeds']} seeds, keep ≥ {KEEP_FRACTION:.0%})",
        "",
        f"**{n} features kept**: {', '.join(kept.index)}",
        "",
        _md(
            stability[stability["selected"] > 0].reset_index(names="feature")[
                ["feature", "confirmed", "tentative", "frequency", "kept"]
            ],
            ".2f",
        ),
        "",
        "## 4. Cross-check with filter and embedded rankings",
        "",
        f"- ANOVA F-test top {n}: {overlap_f} of {n} Boruta features also in it",
        f"- Random-forest importance top {n}: {overlap_rf} of {n} Boruta features also in it",
        "",
        "## 5. Feature sets compared with cross-validation (§7.6)",
        "",
        (
            f"Repeated stratified 5-fold CV × {N_REPEATS}; class-weighted random forest; "
            f"no-skill PR-AUC = {meta['prevalence']:.3f}."
        ),
        "",
        _md(summary.reset_index()),
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Duplicate check, first model, Boruta selection.")
    parser.add_argument("--input", type=Path, default=Path("data/processed/train_prepared.csv"))
    parser.add_argument("--reduced", type=Path, default=Path("data/processed/train_reduced.csv"))
    parser.add_argument("--report-out", type=Path, default=Path("reports/feature_selection"))
    parser.add_argument(
        "--state-out", type=Path, default=Path("data/processed/selected_features.json")
    )
    parser.add_argument("--n-seeds", type=int, default=N_SEEDS)
    parser.add_argument("--skip-cv", action="store_true", help="skip the §7.6 comparison")
    args = parser.parse_args(argv)
    out = args.report_out
    out.mkdir(parents=True, exist_ok=True)

    features, target = load_split(args.input)
    reduced, _ = load_split(args.reduced)
    raw_train, _ = load_split(Path("data/processed/train_raw.csv"))
    duplicates = {
        "train split, 590 raw columns": len(duplicate_features(raw_train)),
        "train split, 474 varying columns (after zero-variance removal)": len(
            duplicate_features(raw_train[[c for c in raw_train.columns if raw_train[c].var() > 0]])
        ),
        "train split, 450 columns (after the 60% threshold)": len(duplicate_features(reduced)),
    }
    print(duplicates, flush=True)

    first = first_model(features, target)
    print(first.to_string(index=False), flush=True)

    imputed = pd.DataFrame(
        knn_imputer().fit(features).transform(features),
        columns=features.columns,
        index=features.index,
    )
    runs = pd.concat([boruta_status(imputed, target, seed) for seed in range(args.n_seeds)], axis=1)
    stability = stability_selection(runs)
    selected = stability.index[stability["kept"]].tolist()
    print("kept:", len(selected), selected, flush=True)
    checks = cross_checks(imputed, target, selected)

    prevalence = float(target.mean())
    summary = pd.DataFrame()
    if not args.skip_cv:
        folds = compare_feature_sets(features, target, selected)
        folds.to_csv(out / "cv_folds.csv", index=False)
        summary = folds.groupby("variant", sort=False).agg(
            n_features=("n_features", "mean"),
            pr_auc=("pr_auc", "mean"),
            pr_auc_sd=("pr_auc", "std"),
            roc_auc=("roc_auc", "mean"),
            recall_at_fpr10=("recall_at_fpr10", "mean"),
            seconds=("seconds", "mean"),
        )
        summary.to_csv(out / "cv_summary.csv")
        save_cv_figure(folds, prevalence, out / "cv_feature_sets.png")

    pd.DataFrame(duplicates.items(), columns=["check", "duplicates"]).to_csv(
        out / "duplicates.csv", index=False
    )
    first.to_csv(out / "first_model.csv", index=False)
    runs.to_csv(out / "boruta_runs.csv", index_label="feature")
    stability.to_csv(out / "boruta_stability.csv", index_label="feature")
    checks.to_csv(out / "ranking_cross_check.csv", index_label="feature")
    save_stability_figure(stability, args.n_seeds, out / "boruta_stability.png")
    meta = {
        "input": str(args.input),
        "rows": len(target),
        "features": features.shape[1],
        "defective": int(target.sum()),
        "prevalence": prevalence,
        "n_seeds": args.n_seeds,
    }
    write_report(meta, duplicates, first, stability, checks, summary, out / "report.md")
    args.state_out.write_text(
        json.dumps(
            {
                "method": "Boruta stability selection",
                "imputer": "k-NN, k=5, standardised (ScaledImputer)",
                "boruta_forest": "RandomForest max_depth=5, class_weight=balanced; BorutaPy max_iter=100",
                "n_seeds": args.n_seeds,
                "keep_fraction": KEEP_FRACTION,
                "fitted_on": str(args.input),
                "selected_features": selected,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Week 4 (T15 / Manual §7): Sophisticated Feature Selection with the Boruta Method.

Takes the Week 2-3 prepared training split (1,253 wafers × 450 features, 3σ-capped) and:
1. Imputes missing values on X_train with Week 3's scaled k-NN imputer (k=5).
2. Runs Boruta (wrapper method using shuffled shadow features around a shallow balanced RF)
   to separate Confirmed, Tentative, and Rejected features.
3. Cross-checks Boruta's selected features against a Filter method (ANOVA F-test) and an
   Embedded method (Random Forest impurity importance).
4. Evaluates All 450 features vs. Boruta-selected features using Stratified 5-Fold CV
   (with k-NN imputation fitted inside every fold to prevent leakage).
5. Saves selected features to ``data/processed/selected_features.json`` and plots/tables
   to ``reports/week4/``.

Run with::

    python -m secom_analysis.week4.feature_selection
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from boruta import BorutaPy
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.metrics import average_precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline

from ..week1.data import ensure_uci_data, load_secom, stratified_split
from ..week2.preparation import (
    cap_outliers,
    load_split,
    missing_percentage,
    save_split,
    three_sigma_bounds,
    verify_sealed,
    zero_variance_features,
)
from ..week3.impute import RANDOM_STATE, knn_imputer

BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#7b7a75", "#ecebe7"


class BorutaSelector(BaseEstimator, TransformerMixin):
    """scikit-learn compatible wrapper around BorutaPy.

    At each iteration, Boruta creates shuffled copies ('shadow features') of all real
    features, trains a shallow class-balanced Random Forest on [real | shadow], and
    tests whether each real feature's importance significantly exceeds max(shadow).

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    max_iter : int
        Maximum Boruta iterations (default 100).
    keep_tentative : bool
        If True (manual recommendation), keep both Confirmed and Tentative features.
    """

    def __init__(
        self,
        seed: int = RANDOM_STATE,
        max_iter: int = 100,
        perc: int = 99,
        keep_tentative: bool = True,
    ) -> None:
        self.seed = seed
        self.max_iter = max_iter
        self.perc = perc
        self.keep_tentative = keep_tentative

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> BorutaSelector:
        values = np.asarray(X, dtype=float)
        self.columns_ = (
            np.asarray(X.columns) if hasattr(X, "columns") else np.arange(values.shape[1])
        )
        rf = RandomForestClassifier(
            max_depth=5,
            class_weight="balanced",
            n_jobs=-1,
            random_state=self.seed,
        )
        self.boruta_ = BorutaPy(
            estimator=rf,
            n_estimators="auto",
            perc=self.perc,
            max_iter=self.max_iter,
            random_state=self.seed,
            verbose=0,
        )
        self.boruta_.fit(values, np.asarray(y))

        support = self.boruta_.support_.copy()
        if self.keep_tentative:
            support |= self.boruta_.support_weak_
        if not support.any():
            support[int(np.argmin(self.boruta_.ranking_))] = True

        self.support_ = support
        self.confirmed_features_ = [str(c) for c in self.columns_[self.boruta_.support_]]
        self.tentative_features_ = [str(c) for c in self.columns_[self.boruta_.support_weak_]]
        self.selected_features_ = [str(c) for c in self.columns_[self.support_]]
        return self

    def transform(self, X: pd.DataFrame | np.ndarray):
        if hasattr(X, "iloc"):
            return X.iloc[:, self.support_]
        return np.asarray(X)[:, self.support_]


def run_boruta_selection(
    x_train_imputed: pd.DataFrame,
    y_train: pd.Series,
    *,
    seed: int = RANDOM_STATE,
    max_iter: int = 100,
    perc: int = 99,
    keep_tentative: bool = True,
) -> tuple[BorutaSelector, pd.DataFrame]:
    """Fit Boruta on the imputed training matrix and return (selector, summary_table)."""
    selector = BorutaSelector(
        seed=seed, max_iter=max_iter, perc=perc, keep_tentative=keep_tentative
    )
    selector.fit(x_train_imputed, y_train)

    status = np.where(
        selector.boruta_.support_,
        "confirmed",
        np.where(selector.boruta_.support_weak_, "tentative", "rejected"),
    )
    summary = pd.DataFrame(
        {
            "feature": x_train_imputed.columns,
            "status": status,
            "boruta_rank": selector.boruta_.ranking_,
            "kept": selector.support_,
        }
    ).sort_values(["kept", "boruta_rank", "feature"], ascending=[False, True, True])
    return selector, summary.reset_index(drop=True)


def cross_check_rankings(
    x_train_imputed: pd.DataFrame,
    y_train: pd.Series,
    selected_features: list[str],
    *,
    seed: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Compare Boruta against Filter (ANOVA F-test) and Embedded (RF importance) rankings."""
    f_scores, p_values = f_classif(x_train_imputed, y_train)
    rf = RandomForestClassifier(
        n_estimators=500,
        class_weight="balanced",
        n_jobs=-1,
        random_state=seed,
    ).fit(x_train_imputed, y_train)

    ranks = pd.DataFrame(
        {
            "feature": x_train_imputed.columns,
            "anova_f": f_scores,
            "anova_p": p_values,
            "rf_importance": rf.feature_importances_,
            "boruta_kept": x_train_imputed.columns.isin(selected_features),
        }
    )
    ranks["anova_rank"] = ranks["anova_f"].rank(ascending=False, method="first").astype(int)
    ranks["rf_rank"] = ranks["rf_importance"].rank(ascending=False, method="first").astype(int)
    return ranks.sort_values("rf_rank").reset_index(drop=True)


def compare_all_vs_selected_cv(
    x_train_raw: pd.DataFrame,
    y_train: pd.Series,
    selected_features: list[str],
    *,
    n_splits: int = 5,
    n_neighbors: int = 5,
    seed: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Stratified 5-Fold CV comparing All 450 features vs. Boruta-selected features.

    k-NN imputation is refitted inside every fold to prevent leakage.
    """
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    base_rf = RandomForestClassifier(
        n_estimators=400,
        class_weight="balanced_subsample",
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=seed,
    )
    variants = {
        f"All features ({x_train_raw.shape[1]})": list(x_train_raw.columns),
        f"Boruta selected ({len(selected_features)})": selected_features,
    }
    rows = []
    for label, cols in variants.items():
        roc_scores, pr_scores, rec_scores = [], [], []
        subset = x_train_raw[cols]
        for train_idx, valid_idx in folds.split(subset, y_train):
            pipe = make_pipeline(knn_imputer(n_neighbors=n_neighbors), clone(base_rf))
            pipe.fit(subset.iloc[train_idx], y_train.iloc[train_idx])
            proba = pipe.predict_proba(subset.iloc[valid_idx])[:, 1]
            pred = (proba >= 0.5).astype(int)
            y_val = y_train.iloc[valid_idx]
            roc_scores.append(roc_auc_score(y_val, proba))
            pr_scores.append(average_precision_score(y_val, proba))
            rec_scores.append(recall_score(y_val, pred, zero_division=0))
        rows.append(
            {
                "feature_set": label,
                "n_features": len(cols),
                "roc_auc_mean": float(np.mean(roc_scores)),
                "roc_auc_sd": float(np.std(roc_scores, ddof=1)),
                "pr_auc_mean": float(np.mean(pr_scores)),
                "pr_auc_sd": float(np.std(pr_scores, ddof=1)),
                "recall_at_0.5_mean": float(np.mean(rec_scores)),
            }
        )
    return pd.DataFrame(rows)


def save_boruta_plot(ranks: pd.DataFrame, output_path: Path, top_n: int = 25) -> None:
    """Plot top-N features by RF importance, highlighting Boruta-selected sensors in orange."""
    top = ranks.head(top_n).iloc[::-1]
    colors = [ORANGE if kept else BLUE for kept in top["boruta_kept"]]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top["feature"], top["rf_importance"], color=colors, height=0.7)
    ax.set_xlabel("Random Forest Impurity Importance (balanced RF)")
    ax.set_title(
        f"Top {top_n} Sensors: Boruta Selected (orange) vs. Not Selected (blue)",
        loc="left",
        fontsize=11,
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color=GRID)
    ax.set_axisbelow(True)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_perc_comparison_plot(
    comparison: pd.DataFrame,
    output_path: Path,
) -> None:
    """Create a 2-panel presentation visual comparing Boruta perc=100 vs. perc=98 vs. All 450.

    Expected columns in ``comparison``:
    ``label``, ``confirmed``, ``tentative``, ``n_features``,
    ``roc_auc_mean``, ``roc_auc_sd``, ``pr_auc_mean``, ``pr_auc_sd``.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2), dpi=200)
    fig.patch.set_facecolor("white")

    # --- Panel A: Confirmed + Tentative feature counts (Boruta rows only)
    boruta_rows = comparison[comparison["confirmed"].notna()].copy()
    y_pos = np.arange(len(boruta_rows))
    confirmed = boruta_rows["confirmed"].to_numpy(dtype=float)
    tentative = boruta_rows["tentative"].to_numpy(dtype=float)
    totals = confirmed + tentative

    ax1.axvspan(15, 20, color="#e8f1fb", alpha=0.85, zorder=0, label="Manual target (15–20)")
    ax1.barh(y_pos, confirmed, height=0.48, color=BLUE, label="Confirmed", zorder=2)
    ax1.barh(
        y_pos,
        tentative,
        left=confirmed,
        height=0.48,
        color=ORANGE,
        label="Tentative (kept)",
        zorder=2,
    )

    for idx, (c_val, t_val, tot) in enumerate(zip(confirmed, tentative, totals)):
        ax1.text(
            c_val / 2,
            idx,
            f"{int(c_val)}",
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
            fontsize=11,
            zorder=3,
        )
        if t_val > 0:
            ax1.text(
                c_val + t_val / 2,
                idx,
                f"{int(t_val)}",
                ha="center",
                va="center",
                color="white",
                fontweight="bold",
                fontsize=10,
                zorder=3,
            )
        ax1.text(
            tot + 0.8,
            idx,
            f"Total: {int(tot)} / 450",
            ha="left",
            va="center",
            color=INK,
            fontweight="bold",
            fontsize=10.5,
        )

    ax1.set_yticks(y_pos, boruta_rows["label"])
    ax1.set_xlim(0, max(totals) * 1.38)
    ax1.set_xlabel("Selected Sensor Features (from 450)", fontsize=10, color=INK)
    ax1.set_title("A. Boruta Feature Count by Shadow Percentile", loc="left", fontsize=11.5, fontweight="bold", color=INK)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.grid(axis="x", color=GRID, zorder=1)
    ax1.legend(frameon=False, loc="lower right", fontsize=9)

    # --- Panel B: 5-Fold CV ROC-AUC & PR-AUC comparison
    x = np.arange(len(comparison))
    width = 0.34
    roc_bars = ax2.bar(
        x - width / 2,
        comparison["roc_auc_mean"],
        width,
        yerr=comparison["roc_auc_sd"],
        capsize=4,
        color=BLUE,
        ecolor=MUTED,
        label="ROC-AUC (mean ± SD)",
        zorder=2,
    )
    pr_bars = ax2.bar(
        x + width / 2,
        comparison["pr_auc_mean"],
        width,
        yerr=comparison["pr_auc_sd"],
        capsize=4,
        color=ORANGE,
        ecolor=MUTED,
        label="PR-AUC (mean ± SD)",
        zorder=2,
    )

    for bars, sds in ((roc_bars, comparison["roc_auc_sd"]), (pr_bars, comparison["pr_auc_sd"])):
        for bar, sd in zip(bars, sds):
            h = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                h + sd + 0.02,
                f"{h:.3f}",
                ha="center",
                va="bottom",
                fontsize=9.5,
                fontweight="bold",
                color=INK,
            )

    ax2.axhline(0.066, color=MUTED, ls="--", lw=1, label="PR-AUC no-skill (0.066)", zorder=1)
    ax2.set_xticks(x, comparison["label"])
    ax2.set_ylim(0, 1.12)
    ax2.set_ylabel("5-Fold CV Score (X_train)", fontsize=10, color=INK)
    ax2.set_title("B. Downstream 5-Fold CV Performance (Balanced RF)", loc="left", fontsize=11.5, fontweight="bold", color=INK)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.grid(axis="y", color=GRID, zorder=0)
    ax2.legend(frameon=False, loc="upper left", ncol=2, fontsize=8.5)

    fig.suptitle(
        "Boruta Shadow Percentile Sensitivity: perc=100 vs. perc=99 vs. perc=98",
        fontsize=13,
        fontweight="bold",
        color=INK,
        y=0.99,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def ensure_prepared_training_split(
    raw_dir: Path = Path("data/raw"),
    processed_dir: Path = Path("data/processed"),
) -> tuple[pd.DataFrame, pd.Series]:
    """Load data/processed/train_prepared.csv, generating Weeks 1-2 artifacts if missing."""
    prepared_path = processed_dir / "train_prepared.csv"
    test_path = processed_dir / "test_holdout.csv"
    if prepared_path.exists() and test_path.exists():
        assert verify_sealed(test_path), "Sealed test set checksum mismatch!"
        return load_split(prepared_path)

    # Run Weeks 1-2 preparation automatically (downloading UCI data if needed)
    data_dir = ensure_uci_data(raw_dir)
    features, target = load_secom(data_dir)
    x_train, x_test, y_train, y_test = stratified_split(
        features, target, test_size=0.2, random_state=RANDOM_STATE
    )
    processed_dir.mkdir(parents=True, exist_ok=True)
    save_split(x_train, y_train, processed_dir / "train_raw.csv", seal=False)
    save_split(x_test, y_test, test_path, seal=True)

    # Rough reduction on train only: zero-variance (590->474) then >60% missing (474->450)
    _, zero_var = zero_variance_features(x_train)
    x_train = x_train.drop(columns=zero_var)
    miss_pct = missing_percentage(x_train)
    high_miss = miss_pct[miss_pct > 60.0].index.tolist()
    x_train_reduced = x_train.drop(columns=high_miss)
    save_split(x_train_reduced, y_train, processed_dir / "train_reduced.csv", seal=False)

    # 3-sigma capping on train only
    bounds = three_sigma_bounds(x_train_reduced, k=3.0)
    bounds.to_csv(processed_dir / "three_sigma_bounds.csv")
    x_train_capped = cap_outliers(x_train_reduced, bounds)
    save_split(x_train_capped, y_train, prepared_path, seal=False)
    return load_split(prepared_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Week 4 (T15): Boruta feature selection on X_train.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/week4"))
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--perc", type=int, default=99, help="Shadow importance percentile (default 99)")
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    args = parser.parse_args(argv)

    x_train, y_train = ensure_prepared_training_split(args.raw_dir, args.processed_dir)
    print(f"Loaded X_train: {x_train.shape[0]:,} wafers × {x_train.shape[1]} features "
          f"({int(y_train.sum())} defective = {y_train.mean():.2%})")

    # Step 1: Week 3 k-NN imputation (k=5, scaled) on X_train
    imputer = knn_imputer(n_neighbors=5)
    x_imputed = pd.DataFrame(
        imputer.fit_transform(x_train),
        index=x_train.index,
        columns=x_train.columns,
    )
    print(f"After k-NN (k=5) imputation: {int(x_imputed.isna().sum().sum())} NaNs remaining.")

    # Step 2: Boruta feature selection (Confirmed + Tentative)
    selector, summary = run_boruta_selection(
        x_imputed, y_train, seed=args.seed, max_iter=args.max_iter, perc=args.perc, keep_tentative=True
    )
    selected = selector.selected_features_
    print(
        f"Boruta (perc={args.perc}) result: {len(selector.confirmed_features_)} confirmed, "
        f"{len(selector.tentative_features_)} tentative -> {len(selected)} selected "
        f"(from {x_train.shape[1]})"
    )
    print("Selected features:", selected)

    # Step 3: Filter (ANOVA F) & Embedded (RF importance) cross-check
    ranks = cross_check_rankings(x_imputed, y_train, selected, seed=args.seed)
    n_sel = len(selected)
    overlap_rf = int(ranks.head(n_sel)["boruta_kept"].sum())
    overlap_anova = int(ranks.nsmallest(n_sel, "anova_rank")["boruta_kept"].sum())
    print(f"Overlap with top-{n_sel} RF importance: {overlap_rf}/{n_sel} | ANOVA F-test: {overlap_anova}/{n_sel}")

    # Step 4: 5-Fold CV comparison (All 450 vs. Boruta selected)
    cv_table = compare_all_vs_selected_cv(x_train, y_train, selected, seed=args.seed)
    print("\n5-Fold CV Comparison (k-NN refitted inside each fold):")
    print(cv_table.to_string(index=False))

    # Save outputs
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "boruta_summary.csv", index=False)
    ranks.to_csv(args.output_dir / "ranking_cross_check.csv", index=False)
    cv_table.to_csv(args.output_dir / "cv_feature_comparison.csv", index=False)
    save_boruta_plot(ranks, args.output_dir / "boruta_top25_importance.png")

    payload = {
        "seed": args.seed,
        "max_iter": args.max_iter,
        "perc": args.perc,
        "input_features": int(x_train.shape[1]),
        "confirmed_features": selector.confirmed_features_,
        "tentative_features": selector.tentative_features_,
        "selected_features": selected,
    }
    (args.processed_dir / "selected_features.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(f"\nSaved selected features to {args.processed_dir / 'selected_features.json'}")
    print(f"Saved Week 4 T15 reports & plot to {args.output_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

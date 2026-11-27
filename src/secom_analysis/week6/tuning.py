"""Week 6 (Manual §11, §12, §14): Model Comparison, Grid Search & Cost-Optimal Thresholding.

Strictly uses ``X_train`` (``data/processed/train_reduced.csv``, 1,253 wafers, 25 Boruta
``perc=99`` features). Verifies ``data/processed/test_holdout.csv`` SHA-256 seal and
**NEVER reads the test set**.

Stages:
- **Stage A (Model Families, §11):** Dummy (always-pass), Logistic Regression (Yeo-Johnson),
  Random Forest, and HistGradientBoosting on the default preparation (3σ cap, k=5, SMOTE).
- **Stage B (Preparation Grid, §12):** OutlierTreatment {cap, remove, none} × k-NN k {3, 5, 10}
  × Balancer {none, class_weight, smote, rose, adasyn} = 45 trials.
- **Stage C (RF Hyperparameter Grid, §12):** Top 3 Stage-B preparations × max_depth
  {3, 5, 8, 12, None} × min_samples_leaf {1, 3, 10} × max_features {"sqrt", 0.3} = 90 trials.
- **Cost-Optimal Honest Thresholding (§14):** For every trial, picks the probability threshold
  that minimises ``100·FP + 400·FN`` on the other 4 folds and applies it to the held-out fold.
  Freezes the winner into ``data/processed/final_config.json`` for later one-time test evaluation.

Run with::

    python -m secom_analysis.week6.tuning
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from itertools import product
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, PowerTransformer

from ..week2.preparation import load_split, verify_sealed
from ..week3.impute import RANDOM_STATE, knn_imputer
from ..week4.balancing_and_eval import (
    BALANCER_LABELS,
    COST_FN,
    COST_FP,
    compute_error_metrics,
    make_sampler,
)
from ..week4.feature_selection import ensure_prepared_training_split
from ..week5.pipeline_and_error import OutlierTreatment

BLUE, ORANGE, TEAL, PURPLE, INK, MUTED, GRID = (
    "#2a78d6",
    "#eb6834",
    "#1f9d8b",
    "#7b52ab",
    "#0b0b0b",
    "#7b7a75",
    "#ecebe7",
)

THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.01), 2)
OUTLIERS = ["cap", "remove", "none"]
NEIGHBOURS = [3, 5, 10]
BALANCERS = ["none", "class_weight", "smote", "rose", "adasyn"]
RF_DEPTHS = [3, 5, 8, 12, None]
RF_MIN_LEAVES = [1, 3, 10]
RF_MAX_FEATURES = ["sqrt", 0.3]


# ---------------------------------------------------------------- Cost & Threshold helpers


def costs_by_threshold(
    y_true: np.ndarray,
    proba: np.ndarray,
    cost_fp: int = COST_FP,
    cost_fn: int = COST_FN,
) -> np.ndarray:
    """Vectorised asymmetric cost across all candidate probability thresholds."""
    y_b = np.asarray(y_true, dtype=bool)
    flagged = np.asarray(proba, dtype=float)[None, :] >= THRESHOLDS[:, None]
    fp = (flagged & ~y_b).sum(axis=1)
    fn = (~flagged & y_b).sum(axis=1)
    return cost_fp * fp + cost_fn * fn


def best_threshold(
    y_true: np.ndarray,
    proba: np.ndarray,
    cost_fp: int = COST_FP,
    cost_fn: int = COST_FN,
) -> float:
    """Probability threshold minimising total asymmetric cost (ties broken closest to 0.50)."""
    costs = costs_by_threshold(y_true, proba, cost_fp=cost_fp, cost_fn=cost_fn)
    best = THRESHOLDS[costs == costs.min()]
    return float(best[np.argmin(np.abs(best - 0.50))])


def honest_cv_threshold_eval(
    y_true: np.ndarray,
    oof_proba: np.ndarray,
    fold_ids: np.ndarray,
    *,
    cost_fp: int = COST_FP,
    cost_fn: int = COST_FN,
) -> dict[str, object]:
    """Nested out-of-fold threshold evaluation (zero threshold leakage).

    For each fold ``f``, chooses ``threshold_f`` using ONLY predictions outside fold ``f``,
    then evaluates fold ``f`` at ``threshold_f``.
    """
    y_arr = np.asarray(y_true, dtype=int)
    p_arr = np.asarray(oof_proba, dtype=float)
    f_arr = np.asarray(fold_ids, dtype=int)

    preds = np.zeros(len(y_arr), dtype=int)
    fold_thresholds = []
    for f in np.unique(f_arr):
        mask = f_arr == f
        t_star = best_threshold(y_arr[~mask], p_arr[~mask], cost_fp=cost_fp, cost_fn=cost_fn)
        fold_thresholds.append(t_star)
        preds[mask] = (p_arr[mask] >= t_star).astype(int)

    tp = int(((preds == 1) & (y_arr == 1)).sum())
    fn = int(((preds == 0) & (y_arr == 1)).sum())
    fp = int(((preds == 1) & (y_arr == 0)).sum())
    tn = int(((preds == 0) & (y_arr == 0)).sum())
    honest_cost = int(cost_fp * fp + cost_fn * fn)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    accuracy = (tp + tn) / len(y_arr)
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0
    f2 = 5 * precision * sensitivity / (4 * precision + sensitivity) if (4 * precision + sensitivity) > 0 else 0.0

    base = compute_error_metrics(y_arr, p_arr, threshold=0.50)
    return {
        "threshold_median": float(np.median(fold_thresholds)),
        "threshold_mean": float(np.mean(fold_thresholds)),
        "threshold_full_oof": best_threshold(y_arr, p_arr, cost_fp=cost_fp, cost_fn=cost_fn),
        "TP": tp,
        "FN": fn,
        "FP": fp,
        "TN": tn,
        "accuracy": float(accuracy),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "f1": float(f1),
        "f2": float(f2),
        "roc_auc": float(base["roc_auc"]),
        "pr_auc": float(base["pr_auc"]),
        "cost_at_0.50": int(base["cost"]),
        "cost_cv_threshold": honest_cost,
    }


# ---------------------------------------------------------------- Fast cached fold preparation


def precompute_fold_matrices(
    x_sel: pd.DataFrame,
    y_train: pd.Series,
    folds: list[tuple[np.ndarray, np.ndarray]],
    outliers: list[str],
    neighbours: list[int],
) -> dict[tuple[int, str, int], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Fit OutlierTreatment -> Scaled k-NN -> MinMaxScaler strictly per fold and cache."""
    cache: dict[tuple[int, str, int], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for fold_idx, (tr_idx, va_idx) in enumerate(folds):
        x_tr_raw = x_sel.iloc[tr_idx]
        x_va_raw = x_sel.iloc[va_idx]
        y_tr = y_train.iloc[tr_idx].to_numpy(dtype=int)
        y_va = y_train.iloc[va_idx].to_numpy(dtype=int)
        for outlier in outliers:
            out_tx = OutlierTreatment(method=outlier, sigma=3.0).fit(x_tr_raw)
            x_tr_o = out_tx.transform(x_tr_raw)
            x_va_o = out_tx.transform(x_va_raw)
            for k in neighbours:
                imp = knn_imputer(n_neighbors=k).fit(x_tr_o)
                x_tr_i = imp.transform(x_tr_o)
                x_va_i = imp.transform(x_va_o)
                scaler = MinMaxScaler().fit(x_tr_i)
                cache[(fold_idx, outlier, k)] = (
                    scaler.transform(x_tr_i),
                    y_tr,
                    scaler.transform(x_va_i),
                    y_va,
                )
    return cache


def make_model(family: str, balancer: str, params: dict | None = None, seed: int = RANDOM_STATE):
    """Instantiate candidate classifier with appropriate class weighting."""
    params = dict(params or {})
    if family == "dummy":
        return DummyClassifier(strategy="most_frequent")
    if family == "logreg":
        cw = "balanced" if balancer == "class_weight" else None
        return LogisticRegression(max_iter=5000, class_weight=cw, random_state=seed, **params)
    if family == "random_forest":
        cw = "balanced_subsample" if balancer == "class_weight" else None
        defaults = {"n_estimators": 300, "n_jobs": -1, "random_state": seed}
        defaults.update(params)
        return RandomForestClassifier(class_weight=cw, **defaults)
    if family == "gradient_boosting":
        cw = "balanced" if balancer == "class_weight" else None
        defaults = {"max_depth": 4, "learning_rate": 0.05, "max_iter": 150, "random_state": seed}
        defaults.update(params)
        return HistGradientBoostingClassifier(class_weight=cw, **defaults)
    raise ValueError(f"Unknown model family: {family}")


def run_cv_trial(
    cache: dict,
    folds: list[tuple[np.ndarray, np.ndarray]],
    n_samples: int,
    *,
    outlier: str,
    n_neighbors: int,
    balancer: str,
    family: str,
    use_yeo_johnson: bool = False,
    model_params: dict | None = None,
    seed: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray]:
    """Run one 5-fold CV trial using cached fold matrices; return (oof_proba, fold_ids)."""
    oof = np.zeros(n_samples, dtype=float)
    fold_ids = np.zeros(n_samples, dtype=int)
    for fold_idx, (_, va_idx) in enumerate(folds):
        x_tr, y_tr, x_va, _ = cache[(fold_idx, outlier, n_neighbors)]
        if use_yeo_johnson:
            pt = PowerTransformer(method="yeo-johnson").fit(x_tr)
            x_tr = pt.transform(x_tr)
            x_va = pt.transform(x_va)
        sampler = make_sampler(balancer, seed=seed)
        if sampler != "passthrough":
            x_tr, y_tr = sampler.fit_resample(x_tr, y_tr)
        clf = make_model(family, balancer, params=model_params, seed=seed)
        clf.fit(x_tr, y_tr)
        proba = clf.predict_proba(x_va)[:, 1] if hasattr(clf, "predict_proba") else clf.predict(x_va).astype(float)
        oof[va_idx] = proba
        fold_ids[va_idx] = fold_idx
    return oof, fold_ids


# ---------------------------------------------------------------- Presentation Visuals (No Week Numbers)


def save_model_and_grid_plot(
    stage_a_df: pd.DataFrame,
    top_grid_df: pd.DataFrame,
    always_pass_cost: int,
    output_path: Path,
) -> None:
    """2-panel presentation chart: Model Family Comparison + Top-10 Grid Search Trials."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 5.4), dpi=200)
    fig.patch.set_facecolor("white")

    # Panel A: Stage A Model Families (Cost at CV Threshold & ROC-AUC)
    sa = stage_a_df.iloc[::-1].reset_index(drop=True)
    y1 = np.arange(len(sa))
    colors1 = [ORANGE if row["cost_cv_threshold"] == sa["cost_cv_threshold"].min() else BLUE for _, row in sa.iterrows()]
    ax1.barh(y1, sa["cost_cv_threshold"], height=0.52, color=colors1)
    ax1.axvline(always_pass_cost, color=MUTED, ls="--", lw=1.2, label=f"Always-pass baseline ({always_pass_cost:,})")
    for idx, row in sa.iterrows():
        ax1.text(
            row["cost_cv_threshold"] + 450,
            idx,
            f"{int(row['cost_cv_threshold']):,}  (AUC {row['roc_auc']:.3f})",
            va="center",
            fontsize=9,
            fontweight="bold",
            color=INK,
        )
    ax1.set_yticks(y1, sa["label"])
    ax1.set_xlim(0, always_pass_cost * 1.48)
    ax1.set_xlabel("5-Fold CV Total Cost (100·FP + 400·FN, lower is better)")
    ax1.set_title("A. Model Family Comparison (Default Prep + SMOTE)", loc="left", fontsize=11, fontweight="bold")
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.grid(axis="x", color=GRID)
    ax1.legend(frameon=False, loc="lower right", fontsize=8.5)

    # Panel B: Top 10 Grid Search Trials by Honest CV Cost
    top = top_grid_df.head(10).iloc[::-1].reset_index(drop=True)
    y2 = np.arange(len(top))
    colors2 = [ORANGE if idx == len(top) - 1 else TEAL for idx in range(len(top))]
    ax2.barh(y2, top["cost_cv_threshold"], height=0.62, color=colors2)
    ax2.axvline(always_pass_cost, color=MUTED, ls="--", lw=1.2, label=f"Always-pass ({always_pass_cost:,})")
    for idx, row in top.iterrows():
        ax2.text(
            row["cost_cv_threshold"] + 350,
            idx,
            f"{int(row['cost_cv_threshold']):,} (Sens {row['sensitivity']:.0%})",
            va="center",
            fontsize=8.3,
            fontweight="bold" if idx == len(top) - 1 else "normal",
            color=INK,
        )
    ax2.set_yticks(y2, top["short_label"], fontsize=8.2)
    ax2.set_xlim(0, always_pass_cost * 1.32)
    ax2.set_xlabel("Honest 5-Fold CV Total Cost (100·FP + 400·FN)")
    ax2.set_title("B. Top 10 Grid Search Configurations (25 Boruta Features)", loc="left", fontsize=11, fontweight="bold")
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.grid(axis="x", color=GRID)
    ax2.legend(frameon=False, loc="lower right", fontsize=8.5)

    fig.suptitle(
        "Model Family Benchmarking & Systematic Pipeline Grid Search (5-Fold CV on X_train)",
        fontsize=13,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_cost_threshold_plot(
    y_true: np.ndarray,
    winner_proba: np.ndarray,
    winner_row: pd.Series,
    always_pass_cost: int,
    output_path: Path,
) -> None:
    """2-panel presentation chart: Asymmetric Cost vs. Threshold Curve + Winning CV Confusion Matrix."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.8, 5.0), dpi=200)
    fig.patch.set_facecolor("white")

    # Panel A: Cost curve across thresholds
    costs = costs_by_threshold(y_true, winner_proba)
    t_opt = float(winner_row["threshold_median"])
    c_opt = int(winner_row["cost_cv_threshold"])
    c_half = int(winner_row["cost_at_0.50"])

    ax1.plot(THRESHOLDS, costs, color=BLUE, lw=2.3, label="Total Cost (100·FP + 400·FN)")
    ax1.axhline(always_pass_cost, color=MUTED, ls="--", lw=1.2, label=f"Always-pass baseline ({always_pass_cost:,})")
    ax1.axvline(t_opt, color=ORANGE, ls=":", lw=1.8, label=f"Optimal CV threshold (p* = {t_opt:.2f})")
    ax1.scatter([t_opt], [c_opt], color=ORANGE, s=75, zorder=5)
    ax1.annotate(
        f"Optimal: {c_opt:,}\n(p* = {t_opt:.2f}, -{always_pass_cost - c_opt:,} vs baseline)",
        xy=(t_opt, c_opt),
        xytext=(min(t_opt + 0.14, 0.55), c_opt - 4500 if c_opt > 24000 else c_opt + 5000),
        arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.4),
        fontsize=9,
        fontweight="bold",
        color=ORANGE,
    )
    ax1.scatter([0.50], [c_half], color=PURPLE, s=55, zorder=5, label=f"Default 0.50 ({c_half:,})")
    ax1.set_xlabel("Decision Probability Threshold P(Defective)")
    ax1.set_ylabel("Out-of-Fold Total Cost (100·FP + 400·FN)")
    ax1.set_title("A. Cost-Sensitive Threshold Optimisation (5-Fold CV)", loc="left", fontsize=11, fontweight="bold")
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.grid(color=GRID)
    ax1.legend(frameon=False, loc="upper right", fontsize=8.5)

    # Panel B: Winning Confusion Matrix at Honest CV Threshold
    cm = np.array([[winner_row["TN"], winner_row["FP"]], [winner_row["FN"], winner_row["TP"]]])
    labels = [
        [f"TN = {int(cm[0, 0])}\n(Correct Pass)", f"FP = {int(cm[0, 1])}\n(Type I × 100)"],
        [f"FN = {int(cm[1, 0])}\n(Type II × 400)", f"TP = {int(cm[1, 1])}\n(Caught Fail)"],
    ]
    ax2.imshow(np.log1p(cm), cmap="Blues")
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 3 else INK
            ax2.text(j, i, labels[i][j], ha="center", va="center", fontsize=11, fontweight="bold", color=color)
    ax2.set_xticks([0, 1], ["Pred: Pass (0)", "Pred: Fail (1)"])
    ax2.set_yticks([0, 1], ["Actual: Pass (0)", "Actual: Fail (1)"])
    ax2.set_title(
        f"B. Winning Pipeline Confusion Matrix (p* = {t_opt:.2f})\n"
        f"Sens={winner_row['sensitivity']:.1%}, Spec={winner_row['specificity']:.1%}, Cost={c_opt:,}",
        loc="left",
        fontsize=10.5,
        fontweight="bold",
    )

    fig.suptitle(
        f"Cost-Optimal Decision Threshold & Winning Configuration: {winner_row['short_label']}",
        fontsize=12.5,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- Main CLI


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Week 6: Model comparison, grid search, and cost threshold.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/week6"))
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    args = parser.parse_args(argv)

    ensure_prepared_training_split(args.raw_dir, args.processed_dir)
    test_path = args.processed_dir / "test_holdout.csv"
    assert verify_sealed(test_path), "Sealed test set checksum mismatch!"

    x_train_reduced, y_train = load_split(args.processed_dir / "train_reduced.csv")
    selected_meta = json.loads((args.processed_dir / "selected_features.json").read_text(encoding="utf-8"))
    selected_cols = selected_meta["selected_features"]
    x_sel = x_train_reduced[selected_cols]
    y_arr = y_train.to_numpy(dtype=int)
    always_pass_cost = int(COST_FN * y_arr.sum())

    print(f"X_train: {len(x_sel):,} wafers × {len(selected_cols)} Boruta (perc={selected_meta.get('perc', 99)}) features")
    print(f"Always-pass baseline cost (100·FP + 400·FN): {always_pass_cost:,} | Test seal: INTACT (NOT READ)")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    folds = list(skf.split(x_sel, y_train))

    print("Precomputing fold preprocessing matrices (3 outliers × 3 k-NN values × 5 folds)...")
    cache = precompute_fold_matrices(x_sel, y_train, folds, OUTLIERS, NEIGHBOURS)

    # --- Stage A: Candidate Model Families (Manual §11)
    stage_a_specs = [
        ("dummy", "Dummy (Always-Pass)", False, "none"),
        ("logreg", "Logistic Regression (YJ + SMOTE)", True, "smote"),
        ("random_forest", "Random Forest (SMOTE)", False, "smote"),
        ("gradient_boosting", "HistGradientBoosting (SMOTE)", False, "smote"),
    ]
    stage_a_rows = []
    for fam, label, use_yj, bal in stage_a_specs:
        oof, fids = run_cv_trial(
            cache, folds, len(x_sel),
            outlier="cap", n_neighbors=5, balancer=bal, family=fam,
            use_yeo_johnson=use_yj, seed=args.seed,
        )
        ev = honest_cv_threshold_eval(y_arr, oof, fids)
        stage_a_rows.append({"family": fam, "label": label, "yeo_johnson": use_yj, "balancer": bal, **ev})
    stage_a_df = pd.DataFrame(stage_a_rows).sort_values("cost_cv_threshold").reset_index(drop=True)
    print("\nStage A — Candidate Model Families (5-Fold CV):")
    print(stage_a_df[["label", "cost_cv_threshold", "cost_at_0.50", "threshold_median", "sensitivity", "roc_auc", "pr_auc"]].to_string(index=False))

    # --- Stage B: Preparation Grid (3 outliers × 3 k × 5 balancers = 45 trials)
    print("\nStage B — Running 45 preparation trials (Outlier × k-NN k × Balancer)...")
    trials = []
    trial_probas: dict[str, np.ndarray] = {}
    for idx, (outlier, k, bal) in enumerate(product(OUTLIERS, NEIGHBOURS, BALANCERS), 1):
        tid = f"B{idx:02d}"
        oof, fids = run_cv_trial(
            cache, folds, len(x_sel),
            outlier=outlier, n_neighbors=k, balancer=bal, family="random_forest", seed=args.seed,
        )
        ev = honest_cv_threshold_eval(y_arr, oof, fids)
        trials.append({
            "trial_id": tid,
            "stage": "B_prep",
            "outlier": outlier,
            "n_neighbors": k,
            "balancer": bal,
            "max_depth": "None",
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "short_label": f"{outlier} | k={k} | {bal} | d=None,l=1",
            **ev,
        })
        trial_probas[tid] = oof

    stage_b_df = pd.DataFrame(trials).sort_values(["cost_cv_threshold", "roc_auc"], ascending=[True, False])
    top3_preps = stage_b_df.head(3)[["outlier", "n_neighbors", "balancer"]].to_dict("records")
    print("Top 3 Stage-B Preparations advancing to Stage C:")
    print(stage_b_df.head(5)[["trial_id", "outlier", "n_neighbors", "balancer", "cost_cv_threshold", "sensitivity", "roc_auc"]].to_string(index=False))

    # --- Stage C: RF Hyperparameter Grid on Top 3 Preparations (3 × 5 × 3 × 2 = 90 trials)
    print("\nStage C — Running 90 Random Forest hyperparameter trials on top 3 preparations...")
    c_idx = 0
    for prep in top3_preps:
        for depth, leaf, mfeat in product(RF_DEPTHS, RF_MIN_LEAVES, RF_MAX_FEATURES):
            c_idx += 1
            tid = f"C{c_idx:02d}"
            rf_params = {"max_depth": depth, "min_samples_leaf": leaf, "max_features": mfeat}
            oof, fids = run_cv_trial(
                cache, folds, len(x_sel),
                outlier=prep["outlier"], n_neighbors=int(prep["n_neighbors"]),
                balancer=prep["balancer"], family="random_forest",
                model_params=rf_params, seed=args.seed,
            )
            ev = honest_cv_threshold_eval(y_arr, oof, fids)
            d_str = "None" if depth is None else str(depth)
            trials.append({
                "trial_id": tid,
                "stage": "C_rf_grid",
                "outlier": prep["outlier"],
                "n_neighbors": int(prep["n_neighbors"]),
                "balancer": prep["balancer"],
                "max_depth": d_str,
                "min_samples_leaf": leaf,
                "max_features": str(mfeat),
                "short_label": f"{prep['outlier']} | k={prep['n_neighbors']} | {prep['balancer']} | d={d_str},l={leaf},{mfeat}",
                **ev,
            })
            trial_probas[tid] = oof

    all_trials_df = pd.DataFrame(trials).sort_values(
        ["cost_cv_threshold", "pr_auc", "roc_auc"], ascending=[True, False, False]
    ).reset_index(drop=True)

    print("\nTop 10 Overall Grid Search Trials (Ranked by Honest CV Cost 100·FP + 400·FN):")
    show_cols = [
        "trial_id", "outlier", "n_neighbors", "balancer", "max_depth", "min_samples_leaf",
        "max_features", "threshold_median", "TP", "FN", "FP", "sensitivity", "roc_auc", "pr_auc", "cost_cv_threshold",
    ]
    print(all_trials_df.head(10)[show_cols].to_string(index=False))

    # Winning trial + sensitivity check (€200 FP / €5,000 FN from Manual §14.2)
    winner = all_trials_df.iloc[0]
    winner_proba = trial_probas[winner["trial_id"]]
    # Recompute fold_ids for sensitivity check
    fold_ids = np.zeros(len(y_arr), dtype=int)
    for f_idx, (_, va_idx) in enumerate(folds):
        fold_ids[va_idx] = f_idx
    alt_cost_eval = honest_cv_threshold_eval(y_arr, winner_proba, fold_ids, cost_fp=200, cost_fn=5000)
    always_pass_alt = int(5000 * y_arr.sum())
    print(
        f"\nSensitivity Check (Manual §14.2 €200 FP / €5,000 FN): "
        f"Always-pass = €{always_pass_alt:,} vs. Winner = €{alt_cost_eval['cost_cv_threshold']:,} "
        f"(p* = {alt_cost_eval['threshold_median']:.2f}, Sensitivity = {alt_cost_eval['sensitivity']:.1%})"
    )

    # Save reports, visuals, and frozen config (WITHOUT touching test_holdout.csv!)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stage_a_df.to_csv(args.output_dir / "w6_stage_a_models.csv", index=False)
    all_trials_df.to_csv(args.output_dir / "w6_trials.csv", index=False)
    save_model_and_grid_plot(
        stage_a_df, all_trials_df, always_pass_cost, args.output_dir / "w6_model_and_grid_comparison.png"
    )
    save_cost_threshold_plot(
        y_arr, winner_proba, winner, always_pass_cost, args.output_dir / "w6_cost_threshold_curve.png"
    )

    final_config = {
        "frozen_at": datetime.now().isoformat(timespec="seconds"),
        "trial_id": str(winner["trial_id"]),
        "test_set_evaluated": False,
        "selected_features": selected_cols,
        "boruta_perc": selected_meta.get("perc", 99),
        "pipeline": {
            "outlier": str(winner["outlier"]),
            "sigma": 3.0,
            "n_neighbors": int(winner["n_neighbors"]),
            "balancer": str(winner["balancer"]),
            "normality": "none",
            "model_family": "random_forest",
            "rf_params": {
                "n_estimators": 500,
                "max_depth": None if winner["max_depth"] == "None" else int(winner["max_depth"]),
                "min_samples_leaf": int(winner["min_samples_leaf"]),
                "max_features": "sqrt" if winner["max_features"] == "sqrt" else float(winner["max_features"]),
            },
            "random_state": args.seed,
        },
        "threshold": float(winner["threshold_median"]),
        "cv_estimate": {
            "cost_cv_threshold": int(winner["cost_cv_threshold"]),
            "always_pass_cost": always_pass_cost,
            "cost_savings_vs_baseline": int(always_pass_cost - winner["cost_cv_threshold"]),
            "TP": int(winner["TP"]),
            "FN": int(winner["FN"]),
            "FP": int(winner["FP"]),
            "TN": int(winner["TN"]),
            "sensitivity": float(winner["sensitivity"]),
            "specificity": float(winner["specificity"]),
            "precision": float(winner["precision"]),
            "roc_auc": float(winner["roc_auc"]),
            "pr_auc": float(winner["pr_auc"]),
            "alt_cost_200_5000": int(alt_cost_eval["cost_cv_threshold"]),
            "alt_threshold_200_5000": float(alt_cost_eval["threshold_median"]),
        },
    }
    (args.processed_dir / "final_config.json").write_text(json.dumps(final_config, indent=2), encoding="utf-8")
    print(f"\nFrozen winning config saved to {args.processed_dir / 'final_config.json'}")
    print(f"Saved Week 6 tables & visuals to {args.output_dir}/ (test_holdout.csv remains SEALED)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

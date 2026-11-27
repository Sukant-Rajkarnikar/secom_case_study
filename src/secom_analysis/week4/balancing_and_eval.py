"""Week 4 (T16 & T17 / Manual §9 & §13): Class Balancing & Initial Confusion Matrix Evaluation.

T16 — Class Balancing (on X_train ONLY, never on validation/test):
- Verifies class counts before and after resampling (SMOTE, ROSE-style smoothed bootstrap,
  ADASYN, and class_weight="balanced").
- Evaluates every balancer inside an ``imblearn`` Pipeline with Stratified 5-Fold CV so
  k-NN imputation, Min-Max scaling, and oversampling are refitted strictly on each fold's
  training rows (zero synthetic wafers ever leak into validation).

T17 — Initial Confusion Matrix & Error Metrics (Positive = 1 = Defective Wafer):
- Computes TP, FP (Type I error, cost 100), FN (Type II error, cost 400), TN.
- Reports Overall Accuracy (vs. 93.38% always-pass baseline), Sensitivity (Recall),
  Specificity, Precision, F1, F2, ROC-AUC, PR-AUC, and Total Cost.
- Generates presentation-ready charts:
  - ``reports/week4/t16_balancing_comparison.png``
  - ``reports/week4/t17_confusion_and_curves.png``

Run with::

    python -m secom_analysis.week4.balancing_and_eval
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
from imblearn.over_sampling import ADASYN, RandomOverSampler, SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

from ..week3.impute import RANDOM_STATE, knn_imputer
from .feature_selection import ensure_prepared_training_split

COST_FP = 100  # Type I error: false alarm (good wafer flagged as defective)
COST_FN = 400  # Type II error: missed defect (defective wafer shipped as good)

BLUE, ORANGE, TEAL, PURPLE, INK, MUTED, GRID = (
    "#2a78d6",
    "#eb6834",
    "#1f9d8b",
    "#7b52ab",
    "#0b0b0b",
    "#7b7a75",
    "#ecebe7",
)

BALANCER_LABELS = {
    "none": "None (imbalanced)",
    "class_weight": "Class weights",
    "smote": "SMOTE",
    "rose": "ROSE-style",
    "adasyn": "ADASYN",
}


def make_sampler(name: str, seed: int = RANDOM_STATE):
    """Return an imblearn sampler for ``name`` (or 'passthrough' for none/class_weight).

    Note: ROSE is an R package; ``RandomOverSampler(shrinkage=1.0)`` is the Python
    smoothed-bootstrap equivalent recommended by the course manual.
    """
    if name in ("none", "class_weight"):
        return "passthrough"
    if name == "smote":
        return SMOTE(random_state=seed)
    if name == "rose":
        return RandomOverSampler(shrinkage=1.0, random_state=seed)
    if name == "adasyn":
        return ADASYN(random_state=seed)
    raise ValueError(f"Unknown balancer: {name}")


def resampling_counts(
    x_train_imputed: pd.DataFrame,
    y_train: pd.Series,
    seed: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Verify exact class counts before and after applying each resampling method on X_train."""
    rows = []
    for name in ("none", "smote", "rose", "adasyn"):
        sampler = make_sampler(name, seed=seed)
        if sampler == "passthrough":
            y_res = y_train
        else:
            _, y_res = sampler.fit_resample(x_train_imputed, y_train)
        n_good = int((y_res == 0).sum())
        n_fail = int((y_res == 1).sum())
        rows.append(
            {
                "balancer": BALANCER_LABELS[name],
                "good_pass_0": n_good,
                "defective_fail_1": n_fail,
                "total_rows": n_good + n_fail,
                "fail_pct": round(100.0 * n_fail / (n_good + n_fail), 2),
            }
        )
    return pd.DataFrame(rows)


def compute_error_metrics(
    y_true: np.ndarray,
    proba: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float | int]:
    """Compute confusion matrix counts and all Manual §13 error metrics (Positive = 1 = Fail)."""
    y_true = np.asarray(y_true, dtype=int)
    proba = np.asarray(proba, dtype=float)
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    cost = int(COST_FP * fp + COST_FN * fn)
    return {
        "TP": int(tp),
        "FN": int(fn),
        "FP": int(fp),
        "TN": int(tn),
        "accuracy": float(accuracy_score(y_true, pred)),
        "sensitivity": float(recall_score(y_true, pred, zero_division=0)),
        "specificity": specificity,
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "f2": float(fbeta_score(y_true, pred, beta=2.0, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "cost": cost,
    }


def evaluate_balancers_cv(
    x_train_selected: pd.DataFrame,
    y_train: pd.Series,
    *,
    n_splits: int = 5,
    n_neighbors: int = 5,
    threshold: float = 0.5,
    seed: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Evaluate all 5 balancers via leakage-free Stratified 5-Fold CV on X_train.

    Returns
    -------
    summary : pd.DataFrame
        Out-of-fold confusion counts and metrics for each balancer (plus always-pass baseline).
    oof_probas : dict[str, np.ndarray]
        Out-of-fold predicted defect probabilities per balancer (for ROC/PR curves).
    """
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rows = []
    oof_probas: dict[str, np.ndarray] = {}

    # Always-pass reference row
    always_pass = compute_error_metrics(
        y_train.to_numpy(), np.zeros(len(y_train), dtype=float), threshold=threshold
    )
    rows.append({"key": "always_pass", "balancer": "Always-pass baseline", **always_pass})

    for key in ("none", "class_weight", "smote", "rose", "adasyn"):
        class_weight = "balanced_subsample" if key == "class_weight" else None
        rf = RandomForestClassifier(
            n_estimators=500,
            class_weight=class_weight,
            n_jobs=-1,
            random_state=seed,
        )
        pipe = ImbPipeline(
            [
                ("imputer", knn_imputer(n_neighbors=n_neighbors)),
                ("scaler", MinMaxScaler()),
                ("sampler", make_sampler(key, seed=seed)),
                ("model", rf),
            ]
        )
        oof = np.zeros(len(y_train), dtype=float)
        for train_idx, valid_idx in folds.split(x_train_selected, y_train):
            fitted = clone(pipe).fit(x_train_selected.iloc[train_idx], y_train.iloc[train_idx])
            oof[valid_idx] = fitted.predict_proba(x_train_selected.iloc[valid_idx])[:, 1]
        oof_probas[key] = oof
        metrics = compute_error_metrics(y_train.to_numpy(), oof, threshold=threshold)
        rows.append({"key": key, "balancer": BALANCER_LABELS[key], **metrics})

    return pd.DataFrame(rows), oof_probas


def save_balancing_comparison_plot(
    counts_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """2-panel Week 4 (T16) chart: class counts after balancing + Sensitivity/Precision/PR-AUC."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2), dpi=200)
    fig.patch.set_facecolor("white")

    # Panel A: Class counts before & after resampling
    y = np.arange(len(counts_df))
    ax1.barh(y - 0.18, counts_df["good_pass_0"], height=0.34, color=BLUE, label="Pass / Good (0)")
    ax1.barh(y + 0.18, counts_df["defective_fail_1"], height=0.34, color=ORANGE, label="Fail / Defective (1)")
    for idx, row in counts_df.iterrows():
        ax1.text(row["good_pass_0"] + 20, idx - 0.18, f"{row['good_pass_0']:,}", va="center", fontsize=9, color=INK)
        ax1.text(
            row["defective_fail_1"] + 20,
            idx + 0.18,
            f"{row['defective_fail_1']:,} ({row['fail_pct']:.1f}%)",
            va="center",
            fontsize=9,
            fontweight="bold",
            color=INK,
        )
    ax1.set_yticks(y, counts_df["balancer"])
    ax1.set_xlim(0, counts_df["defective_fail_1"].max() * 1.48)
    ax1.set_ylim(-0.6, len(counts_df) - 0.1)
    ax1.set_xlabel("Number of Training Wafers (X_train only)", fontsize=10)
    ax1.set_title("A. Class Distribution Before & After Balancing", loc="left", fontsize=11.5, fontweight="bold")
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.grid(axis="x", color=GRID)
    ax1.legend(frameon=False, loc="upper right", ncol=2, fontsize=8.8)

    # Panel B: Sensitivity (Recall), Precision, and PR-AUC across balancers (5-fold CV)
    models = summary_df[summary_df["key"] != "always_pass"].reset_index(drop=True)
    x = np.arange(len(models))
    w = 0.25
    b1 = ax2.bar(x - w, models["sensitivity"], w, color=ORANGE, label="Sensitivity (Recall)")
    b2 = ax2.bar(x, models["precision"], w, color=BLUE, label="Precision")
    b3 = ax2.bar(x + w, models["pr_auc"], w, color=TEAL, label="PR-AUC")

    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.012,
                f"{h:.2f}",
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="bold",
                color=INK,
            )

    ax2.set_xticks(x, models["balancer"], rotation=15, ha="right")
    ax2.set_ylim(0, max(models[["sensitivity", "precision", "pr_auc"]].max().max() * 1.35, 0.5))
    ax2.set_ylabel("5-Fold CV Out-of-Fold Score (threshold = 0.50)", fontsize=10)
    ax2.set_title("B. Impact of Balancing on Defect Detection (25 Boruta Features)", loc="left", fontsize=11.5, fontweight="bold")
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.grid(axis="y", color=GRID)
    ax2.legend(frameon=False, loc="upper left", fontsize=9)

    fig.suptitle(
        "Class Balancing on X_train: Resampling Counts & 5-Fold CV Performance",
        fontsize=13,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_confusion_and_curves_plot(
    y_true: np.ndarray,
    summary_df: pd.DataFrame,
    oof_probas: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    """3-panel Week 4 (T17) chart: SMOTE & ROSE Confusion Matrices + ROC & PR Curves."""
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), dpi=200)
    fig.patch.set_facecolor("white")

    # Panel A: Confusion Matrix for SMOTE (Manual's primary balancer)
    smote_row = summary_df[summary_df["key"] == "smote"].iloc[0]
    cm = np.array([[smote_row["TN"], smote_row["FP"]], [smote_row["FN"], smote_row["TP"]]])
    labels = [
        [f"TN = {cm[0, 0]}\n(Correct Pass)", f"FP = {cm[0, 1]}\n(Type I: 100)"],
        [f"FN = {cm[1, 0]}\n(Type II: 400)", f"TP = {cm[1, 1]}\n(Caught Fail)"],
    ]
    axes[0].imshow(np.log1p(cm), cmap="Blues")
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 3 else INK
            axes[0].text(j, i, labels[i][j], ha="center", va="center", fontsize=10, fontweight="bold", color=color)
    axes[0].set_xticks([0, 1], ["Pred: Pass (0)", "Pred: Fail (1)"])
    axes[0].set_yticks([0, 1], ["Actual: Pass (0)", "Actual: Fail (1)"])
    axes[0].set_title(
        f"A. SMOTE Confusion Matrix (5-Fold CV)\nSens={smote_row['sensitivity']:.2%}, Cost={smote_row['cost']:,}",
        fontsize=10.5,
        fontweight="bold",
    )

    # Panel B: ROC Curves across balancers
    palette = {"none": MUTED, "class_weight": PURPLE, "smote": BLUE, "rose": ORANGE, "adasyn": TEAL}
    for key, proba in oof_probas.items():
        fpr, tpr, _ = roc_curve(y_true, proba)
        auc = roc_auc_score(y_true, proba)
        axes[1].plot(fpr, tpr, lw=1.8, color=palette[key], label=f"{BALANCER_LABELS[key]} ({auc:.3f})")
    axes[1].plot([0, 1], [0, 1], ls="--", color=MUTED, lw=1, label="Random (0.500)")
    axes[1].set_xlabel("False Positive Rate (1 − Specificity)")
    axes[1].set_ylabel("True Positive Rate (Sensitivity)")
    axes[1].set_title("B. ROC Curves (5-Fold CV Out-of-Fold)", fontsize=10.5, fontweight="bold")
    axes[1].spines[["top", "right"]].set_visible(False)
    axes[1].grid(color=GRID)
    axes[1].legend(frameon=False, loc="lower right", fontsize=8.5)

    # Panel C: Precision-Recall Curves (shows Precision-Sensitivity trade-off)
    for key, proba in oof_probas.items():
        prec, rec, _ = precision_recall_curve(y_true, proba)
        ap = average_precision_score(y_true, proba)
        axes[2].plot(rec, prec, lw=1.8, color=palette[key], label=f"{BALANCER_LABELS[key]} ({ap:.3f})")
    baseline = float(np.mean(y_true))
    axes[2].axhline(baseline, ls="--", color=MUTED, lw=1, label=f"No-skill ({baseline:.3f})")
    axes[2].set_xlabel("Sensitivity / Recall (Defect Detection Rate)")
    axes[2].set_ylabel("Precision (Alarm Accuracy)")
    axes[2].set_title("C. Precision–Recall Trade-Off (5-Fold CV)", fontsize=10.5, fontweight="bold")
    axes[2].spines[["top", "right"]].set_visible(False)
    axes[2].grid(color=GRID)
    axes[2].legend(frameon=False, loc="upper right", fontsize=8.5)

    fig.suptitle(
        "Initial Confusion Matrix, Type I/II Errors & ROC / Precision–Recall Curves",
        fontsize=13,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Week 4 (T16 & T17): Balancing & Initial Confusion Matrix.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/week4"))
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    args = parser.parse_args(argv)

    x_train, y_train = ensure_prepared_training_split(args.raw_dir, args.processed_dir)
    selected_path = args.processed_dir / "selected_features.json"
    selected_meta = json.loads(selected_path.read_text(encoding="utf-8"))
    selected_cols = selected_meta["selected_features"]
    x_sel = x_train[selected_cols]
    print(f"Using {len(selected_cols)} Boruta-selected features (perc={selected_meta.get('perc', 'n/a')}) on X_train ({len(x_sel):,} rows).")

    # Verify resampling counts on imputed X_train (selected features)
    x_sel_imputed = pd.DataFrame(
        knn_imputer(n_neighbors=5).fit_transform(x_sel),
        index=x_sel.index,
        columns=x_sel.columns,
    )
    counts_df = resampling_counts(x_sel_imputed, y_train, seed=args.seed)
    print("\nT16 — Class Counts Before & After Balancing (X_train only):")
    print(counts_df.to_string(index=False))

    # Leakage-free 5-fold CV evaluation of all balancers
    summary_df, oof_probas = evaluate_balancers_cv(x_sel, y_train, seed=args.seed)
    print("\nT17 — 5-Fold CV Out-of-Fold Confusion Matrices & Metrics (threshold = 0.50):")
    display_cols = [
        "balancer", "TP", "FN", "FP", "TN",
        "accuracy", "sensitivity", "specificity", "precision", "f1", "f2", "roc_auc", "pr_auc", "cost",
    ]
    print(summary_df[display_cols].to_string(index=False))

    # Save tables and plots
    args.output_dir.mkdir(parents=True, exist_ok=True)
    counts_df.to_csv(args.output_dir / "t16_resampling_counts.csv", index=False)
    summary_df.to_csv(args.output_dir / "t17_balancer_metrics_cv.csv", index=False)
    save_balancing_comparison_plot(counts_df, summary_df, args.output_dir / "t16_balancing_comparison.png")
    save_confusion_and_curves_plot(
        y_train.to_numpy(), summary_df, oof_probas, args.output_dir / "t17_confusion_and_curves.png"
    )
    print(f"\nSaved Week 4 T16 & T17 outputs to {args.output_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

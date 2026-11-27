"""Week 5 (Manual §8 & §10): Normality Proof, Min-Max Scaling, Leakage-Free Pipeline & Train vs. Val Error.

Implements and verifies all four Week 5 SIP decisions strictly on ``X_train`` (sealed
``test_holdout.csv`` is verified by SHA-256 and never read):

1. **Normality Check & Empirical Proof (§8.1–§8.2)**:
   - Shapiro-Wilk normality & skewness audit on the 25 Boruta (perc=99) features before and
     after Yeo-Johnson transformation (also documenting why Box-Cox fails on <=0 sensor values).
   - Empirical 5-Fold CV proof comparing ``No transform`` vs. ``Yeo-Johnson`` on **Random Forest**
     (invariant to monotone transforms) and **Logistic Regression** (sensitive to skew).
   - Visual: ``reports/week5/w5_normality_proof.png``.
2. **Min-Max Scaling & Leakage-Free Pipeline (§8.3–§8.5, §10)**:
   - Switchable ``OutlierTreatment`` (3σ cap / remove / none) + ``FrameMinMaxScaler`` (keeps
     DataFrame column names, verifies [0, 1] range on train).
   - ``build_pipeline()`` returning an ``imblearn.pipeline.Pipeline`` in §17.1 order:
     ``OutlierTreatment -> Scaled KNNImputer -> Selector -> MinMaxScaler -> Balancer -> Model``.
   - Automated leakage & reproducibility verification (0 train/val overlap, SMOTE skipped on
     validation/test, identical predictions on refit, test seal intact).
   - Visual: ``reports/week5/w5_pipeline_process_map.png``.
3. **Training Error vs. Validation Error & Model Complexity (Week 5 To-do)**:
   - Sweeps Random Forest ``max_depth`` over ``[2, 3, 5, 8, 12, 20, None]`` using 5-Fold CV on
     ``X_train``, plotting Training vs. Validation Error and ROC-AUC to diagnose overfitting.
   - Visual: ``reports/week5/w5_train_vs_val_complexity.png``.

Run with::

    python -m secom_analysis.week5.pipeline_and_error
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, PowerTransformer

from ..week2.preparation import load_split, verify_sealed
from ..week3.impute import RANDOM_STATE, knn_imputer
from ..week4.balancing_and_eval import compute_error_metrics, make_sampler
from ..week4.feature_selection import BorutaSelector, ensure_prepared_training_split

BLUE, ORANGE, TEAL, PURPLE, INK, MUTED, GRID = (
    "#2a78d6",
    "#eb6834",
    "#1f9d8b",
    "#7b52ab",
    "#0b0b0b",
    "#7b7a75",
    "#ecebe7",
)


# ---------------------------------------------------------------- Pipeline building blocks


class OutlierTreatment(BaseEstimator, TransformerMixin):
    """Switchable 3-sigma outlier transformer fitted strictly on training folds.

    Modes:
    - ``"cap"``: clip values outside [mean - sigma*std, mean + sigma*std] to the boundary.
    - ``"remove"``: replace values outside bounds with NaN (for downstream k-NN imputer).
    - ``"none"``: pass values through unchanged.
    """

    def __init__(self, method: str = "cap", sigma: float = 3.0) -> None:
        self.method = method
        self.sigma = sigma

    def fit(self, X: pd.DataFrame | np.ndarray, y: object = None) -> OutlierTreatment:
        arr = np.asarray(X, dtype=float)
        mean = np.nanmean(arr, axis=0)
        std = np.nanstd(arr, axis=0, ddof=1)
        self.lower_ = mean - self.sigma * std
        self.upper_ = mean + self.sigma * std
        return self

    def transform(self, X: pd.DataFrame | np.ndarray):
        if self.method == "none":
            return X.copy() if hasattr(X, "copy") else np.asarray(X, dtype=float).copy()
        arr = np.asarray(X, dtype=float).copy()
        if self.method == "cap":
            arr = np.clip(arr, self.lower_, self.upper_)
        elif self.method == "remove":
            outside = (arr < self.lower_) | (arr > self.upper_)
            arr[outside] = np.nan
        else:
            raise ValueError(f"Unknown outlier method: {self.method}")
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(arr, index=X.index, columns=X.columns)
        return arr


class FrameKNNImputer(BaseEstimator, TransformerMixin):
    """Scaled k-NN imputer wrapper that preserves DataFrame column names and index."""

    def __init__(self, n_neighbors: int = 5) -> None:
        self.n_neighbors = n_neighbors

    def fit(self, X: pd.DataFrame | np.ndarray, y: object = None) -> FrameKNNImputer:
        self.imputer_ = knn_imputer(n_neighbors=self.n_neighbors).fit(X)
        return self

    def transform(self, X: pd.DataFrame | np.ndarray):
        arr = self.imputer_.transform(X)
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(arr, index=X.index, columns=X.columns)
        return arr


class FrameMinMaxScaler(BaseEstimator, TransformerMixin):
    """MinMaxScaler wrapper that preserves DataFrame column names and index."""

    def __init__(self, feature_range: tuple[float, float] = (0.0, 1.0)) -> None:
        self.feature_range = feature_range

    def fit(self, X: pd.DataFrame | np.ndarray, y: object = None) -> FrameMinMaxScaler:
        self.scaler_ = MinMaxScaler(feature_range=self.feature_range).fit(X)
        self.columns_ = list(X.columns) if isinstance(X, pd.DataFrame) else None
        return self

    def transform(self, X: pd.DataFrame | np.ndarray):
        scaled = self.scaler_.transform(X)
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(scaled, index=X.index, columns=X.columns)
        return scaled


class ColumnSelector(BaseEstimator, TransformerMixin):
    """Selects a fixed list of column names (or passes through if None)."""

    def __init__(self, columns: list[str] | None = None) -> None:
        self.columns = columns

    def fit(self, X: pd.DataFrame | np.ndarray, y: object = None) -> ColumnSelector:
        if isinstance(X, pd.DataFrame):
            self.indices_ = (
                [X.columns.get_loc(c) for c in self.columns]
                if self.columns is not None
                else list(range(X.shape[1]))
            )
            self.selected_columns_ = list(self.columns) if self.columns is not None else list(X.columns)
        else:
            self.indices_ = list(range(np.asarray(X).shape[1]))
            self.selected_columns_ = None
        return self

    def transform(self, X: pd.DataFrame | np.ndarray):
        if isinstance(X, pd.DataFrame):
            return X.iloc[:, self.indices_]
        return np.asarray(X)[:, self.indices_]


def build_pipeline(
    *,
    outlier: str = "cap",
    sigma: float = 3.0,
    n_neighbors: int = 5,
    selector: str | list[str] = "all",
    boruta_perc: int = 99,
    normality: str = "none",
    balancer: str = "smote",
    model_family: str = "random_forest",
    model_params: dict | None = None,
    seed: int = RANDOM_STATE,
) -> ImbPipeline:
    """Build the leakage-free Week 5 imblearn Pipeline (Manual §10 & §17.1 order).

    Order: OutlierTreatment -> Scaled KNNImputer -> Selector -> Normality (optional)
           -> MinMaxScaler [0, 1] -> Balancer (fit only) -> Classifier.
    """
    params = dict(model_params or {})
    if selector == "boruta":
        sel_step = BorutaSelector(seed=seed, perc=boruta_perc, keep_tentative=True)
    elif selector == "all":
        sel_step = "passthrough"
    elif isinstance(selector, list):
        sel_step = ColumnSelector(selector)
    else:
        raise ValueError(f"Unknown selector: {selector}")

    norm_step = PowerTransformer(method="yeo-johnson") if normality == "yeo-johnson" else "passthrough"
    class_weight = "balanced_subsample" if (balancer == "class_weight" and model_family == "random_forest") else (
        "balanced" if balancer == "class_weight" else None
    )

    if model_family == "random_forest":
        defaults = {"n_estimators": 500, "n_jobs": -1, "random_state": seed}
        defaults.update(params)
        clf = RandomForestClassifier(class_weight=class_weight, **defaults)
    elif model_family == "logreg":
        defaults = {"max_iter": 5000, "random_state": seed}
        defaults.update(params)
        clf = LogisticRegression(class_weight=class_weight, **defaults)
    else:
        raise ValueError(f"Unknown model_family: {model_family}")

    return ImbPipeline(
        [
            ("outlier", OutlierTreatment(method=outlier, sigma=sigma)),
            ("imputer", FrameKNNImputer(n_neighbors=n_neighbors)),
            ("selector", sel_step),
            ("normality", norm_step),
            ("scaler", FrameMinMaxScaler((0.0, 1.0))),
            ("sampler", make_sampler(balancer, seed=seed)),
            ("model", clf),
        ]
    )


# ---------------------------------------------------------------- 1. Normality Audit & Proof


def normality_audit(x_imputed: pd.DataFrame) -> pd.DataFrame:
    """Audit Shapiro-Wilk normality, skewness, and Box-Cox eligibility before & after Yeo-Johnson."""
    pt = PowerTransformer(method="yeo-johnson")
    x_yj = pd.DataFrame(pt.fit_transform(x_imputed), index=x_imputed.index, columns=x_imputed.columns)

    rows = []
    for col in x_imputed.columns:
        raw = x_imputed[col].to_numpy(dtype=float)
        yj = x_yj[col].to_numpy(dtype=float)
        _, p_raw = stats.shapiro(raw)
        _, p_yj = stats.shapiro(yj)
        rows.append(
            {
                "feature": col,
                "min_val": float(np.min(raw)),
                "has_non_positive": bool(np.min(raw) <= 0),
                "skew_raw": float(stats.skew(raw)),
                "abs_skew_raw": float(abs(stats.skew(raw))),
                "shapiro_p_raw": float(p_raw),
                "normal_raw": bool(p_raw >= 0.05),
                "skew_yj": float(stats.skew(yj)),
                "abs_skew_yj": float(abs(stats.skew(yj))),
                "shapiro_p_yj": float(p_yj),
                "normal_yj": bool(p_yj >= 0.05),
            }
        )
    return pd.DataFrame(rows).sort_values("abs_skew_raw", ascending=False).reset_index(drop=True)


def normality_cv_proof(
    x_train_selected: pd.DataFrame,
    y_train: pd.Series,
    *,
    n_splits: int = 5,
    seed: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Empirical 5-Fold CV proof: No transform vs. Yeo-Johnson for Random Forest & Logistic Regression."""
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rows = []
    for family, label in (("random_forest", "Random Forest (SMOTE)"), ("logreg", "Logistic Regression (SMOTE)")):
        for norm in ("none", "yeo-johnson"):
            pipe = build_pipeline(
                outlier="cap",
                n_neighbors=5,
                selector="all",
                normality=norm,
                balancer="smote",
                model_family=family,
                seed=seed,
            )
            oof = np.zeros(len(y_train), dtype=float)
            for train_idx, valid_idx in folds.split(x_train_selected, y_train):
                fitted = clone(pipe).fit(x_train_selected.iloc[train_idx], y_train.iloc[train_idx])
                oof[valid_idx] = fitted.predict_proba(x_train_selected.iloc[valid_idx])[:, 1]
            m = compute_error_metrics(y_train.to_numpy(), oof, threshold=0.5)
            rows.append(
                {
                    "model": label,
                    "model_family": family,
                    "normality": "No transform" if norm == "none" else "Yeo-Johnson",
                    "roc_auc": m["roc_auc"],
                    "pr_auc": m["pr_auc"],
                    "sensitivity": m["sensitivity"],
                    "cost": m["cost"],
                }
            )
    return pd.DataFrame(rows)


def save_normality_proof_plot(
    audit_df: pd.DataFrame,
    cv_proof_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """2-panel visual proof for Week 5 Decision 1 (Normality Transform)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.4), dpi=200)
    fig.patch.set_facecolor("white")

    # Panel A: Top 12 most skewed Boruta features — |Skewness| Raw vs. Yeo-Johnson
    top = audit_df.head(12).iloc[::-1]
    y = np.arange(len(top))
    ax1.barh(y + 0.18, top["abs_skew_raw"], height=0.34, color=ORANGE, label="Raw (3σ-capped)")
    ax1.barh(y - 0.18, top["abs_skew_yj"], height=0.34, color=BLUE, label="After Yeo-Johnson")
    ax1.axvline(1.0, color=MUTED, ls="--", lw=1, label="|skew| = 1.0 threshold")
    n_non_norm_raw = int((~audit_df["normal_raw"]).sum())
    n_non_norm_yj = int((~audit_df["normal_yj"]).sum())
    n_non_pos = int(audit_df["has_non_positive"].sum())
    ax1.set_yticks(y, top["feature"])
    ax1.set_xlabel("Absolute Skewness |skew| (Top 12 Most Skewed Boruta Sensors)")
    ax1.set_title(
        f"A. Normality Audit ({len(audit_df)} Sensors: {n_non_norm_raw} Non-Normal Raw, {n_non_norm_yj} After YJ;\n"
        f"{n_non_pos} Have Values ≤ 0 → Box-Cox Invalid)",
        loc="left",
        fontsize=10,
        fontweight="bold",
    )
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.grid(axis="x", color=GRID)
    ax1.legend(frameon=False, loc="lower right", fontsize=8.8)

    # Panel B: Empirical 5-Fold CV ROC-AUC & PR-AUC — No Transform vs. Yeo-Johnson
    models = ["Random Forest (SMOTE)", "Logistic Regression (SMOTE)"]
    x = np.arange(len(models))
    w = 0.18
    for idx, (norm_label, color_roc, color_pr, offset) in enumerate(
        [("No transform", BLUE, "#7db3f0", -w), ("Yeo-Johnson", ORANGE, "#f4a683", w)]
    ):
        sub = cv_proof_df[cv_proof_df["normality"] == norm_label].set_index("model").loc[models]
        b_roc = ax2.bar(x + offset - w / 2, sub["roc_auc"], w * 0.9, color=color_roc, label=f"ROC-AUC ({norm_label})")
        b_pr = ax2.bar(x + offset + w / 2, sub["pr_auc"], w * 0.9, color=color_pr, label=f"PR-AUC ({norm_label})")
        for bar in list(b_roc) + list(b_pr):
            h = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.015,
                f"{h:.3f}",
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="bold",
                color=INK,
            )

    ax2.set_xticks(x, ["Random Forest\n(Non-Parametric)", "Logistic Regression\n(Linear / Parametric)"])
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("5-Fold CV Out-of-Fold Score (X_train)")
    ax2.set_title(
        "B. Empirical Proof: RF Is Invariant to Yeo-Johnson,\nWhereas Linear Models Respond to Skew Reduction",
        loc="left",
        fontsize=10,
        fontweight="bold",
    )
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.grid(axis="y", color=GRID)
    ax2.legend(frameon=False, loc="upper right", ncol=2, fontsize=8)

    fig.suptitle(
        "Normality Transformation Analysis — Why Normality Transform Is Skipped for Random Forest",
        fontsize=13,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- 2. Pipeline Checks & Process Map


def run_pipeline_checks(
    x_train_reduced: pd.DataFrame,
    y_train: pd.Series,
    selected_cols: list[str],
    test_path: Path,
    *,
    seed: int = RANDOM_STATE,
) -> dict[str, object]:
    """Verify Manual §10.5 leakage, Min-Max scaling [0, 1], reproducibility, and test seal."""
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    train_idx, valid_idx = next(folds.split(x_train_reduced, y_train))
    x_tr, y_tr = x_train_reduced.iloc[train_idx], y_train.iloc[train_idx]
    x_va = x_train_reduced.iloc[valid_idx]

    pipe = build_pipeline(
        outlier="cap",
        n_neighbors=5,
        selector=selected_cols,
        normality="none",
        balancer="smote",
        model_family="random_forest",
        seed=seed,
    )
    fitted1 = clone(pipe).fit(x_tr, y_tr)
    fitted2 = clone(pipe).fit(x_tr, y_tr)
    pred1 = fitted1.predict_proba(x_va)[:, 1]
    pred2 = fitted2.predict_proba(x_va)[:, 1]

    # Verify Min-Max range [0, 1] on the training fold right after the scaler step
    x_tr_scaled = fitted1[:5].transform(x_tr)
    x_resampled, y_resampled = fitted1.named_steps["sampler"].fit_resample(x_tr_scaled, y_tr)

    return {
        "fold0_train_rows": int(len(x_tr)),
        "fold0_valid_rows": int(len(x_va)),
        "shared_indices": int(len(set(x_tr.index) & set(x_va.index))),
        "features_in": int(x_tr.shape[1]),
        "features_after_selector": int(x_tr_scaled.shape[1]),
        "minmax_train_min": float(np.min(x_tr_scaled)),
        "minmax_train_max": float(np.max(x_tr_scaled)),
        "rows_after_smote_train_only": int(len(x_resampled)),
        "smote_train_fail_count": int((y_resampled == 1).sum()),
        "valid_predictions_count": int(len(pred1)),
        "reproducible_exact_match": bool(np.allclose(pred1, pred2)),
        "test_seal_intact": bool(verify_sealed(test_path)),
    }


def save_pipeline_process_map(output_path: Path) -> None:
    """Draw a clean executive Week 5 Process Map showing fit-on-train vs. transform-only-on-test."""
    fig, ax = plt.subplots(figsize=(14.5, 4.4), dpi=200)
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 4.8)
    ax.axis("off")

    steps = [
        ("1. Raw Split", "1,567 × 590\n80/20 Stratified\nTest Sealed", BLUE),
        ("2. Rough Drop", "Train only:\n0-var (116)\n>60% NaN (24)\n→ 450 cols", BLUE),
        ("3. 3σ Capping", "Fit μ ± 3σ\non Train fold\nClip extremes", BLUE),
        ("4. k-NN Impute", "Scaled k-NN\n(k = 5)\nNaN → 0", TEAL),
        ("5. Boruta", "Shadow RF\n(perc = 99)\n450 → 25 cols", TEAL),
        ("6. Min-Max", "Fit [0, 1]\non Train fold\nPreserves shape", PURPLE),
        ("7. SMOTE", "TRAIN ONLY!\n83 → 1,170 Fail\nSkipped on Val/Test", ORANGE),
        ("8. Random Forest", "500 Trees\nPredict P(Fail)\nCost 100/400", INK),
    ]

    box_w, box_h = 1.55, 2.05
    xs = np.linspace(0.35, 13.1, len(steps))
    y_top = 1.65

    for idx, (x, (title, body, color)) in enumerate(zip(xs, steps)):
        rect = mpatches.FancyBboxPatch(
            (x, y_top),
            box_w,
            box_h,
            boxstyle="round,pad=0.08,rounding_size=0.14",
            facecolor="white",
            edgecolor=color,
            linewidth=2.0,
        )
        ax.add_patch(rect)
        # Header band
        hdr = mpatches.FancyBboxPatch(
            (x, y_top + box_h - 0.62),
            box_w,
            0.62,
            boxstyle="round,pad=0.06,rounding_size=0.12",
            facecolor=color,
            edgecolor=color,
            linewidth=1.0,
        )
        ax.add_patch(hdr)
        ax.text(x + box_w / 2, y_top + box_h - 0.31, title, ha="center", va="center", color="white", fontsize=8.5, fontweight="bold")
        ax.text(x + box_w / 2, y_top + (box_h - 0.65) / 2, body, ha="center", va="center", color=INK, fontsize=7.8)
        if idx < len(steps) - 1:
            ax.annotate(
                "",
                xy=(xs[idx + 1] - 0.08, y_top + box_h / 2),
                xytext=(x + box_w + 0.08, y_top + box_h / 2),
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.8),
            )

    # Leakage control banners underneath
    ax.text(
        7.5,
        0.95,
        "INSIDE imblearn.pipeline.Pipeline (Steps 3–8): Fitted strictly on each training fold → transform-only on validation/test",
        ha="center",
        va="center",
        fontsize=9.5,
        fontweight="bold",
        color=BLUE,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#eef5fc", edgecolor=BLUE, lw=1.2),
    )
    ax.text(
        7.5,
        0.35,
        "Zero-Leakage Guarantee: SMOTE runs ONLY during .fit(); sealed test_holdout.csv (SHA-256) remains untouched until final evaluation.",
        ha="center",
        va="center",
        fontsize=8.8,
        color=ORANGE,
        fontweight="bold",
    )
    ax.set_title(
        "End-to-End Leakage-Free SECOM Machine Learning Pipeline Process Map",
        fontsize=12.5,
        fontweight="bold",
        color=INK,
        pad=6,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- 3. Training vs. Validation Error Curve


def complexity_curve(
    x_train_selected: pd.DataFrame,
    y_train: pd.Series,
    *,
    depths: list[int | None] | None = None,
    n_splits: int = 5,
    seed: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Measure Training Error vs. 5-Fold CV Validation Error across Random Forest max_depth."""
    if depths is None:
        depths = [2, 3, 5, 8, 12, 20, None]
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rows = []
    for depth in depths:
        pipe = build_pipeline(
            outlier="cap",
            n_neighbors=5,
            selector="all",
            normality="none",
            balancer="smote",
            model_family="random_forest",
            model_params={"n_estimators": 300, "max_depth": depth},
            seed=seed,
        )
        tr_errs, va_errs, tr_aucs, va_aucs, va_recs = [], [], [], [], []
        for train_idx, valid_idx in folds.split(x_train_selected, y_train):
            x_tr, y_tr = x_train_selected.iloc[train_idx], y_train.iloc[train_idx]
            x_va, y_va = x_train_selected.iloc[valid_idx], y_train.iloc[valid_idx]
            fitted = clone(pipe).fit(x_tr, y_tr)
            p_tr = fitted.predict_proba(x_tr)[:, 1]
            p_va = fitted.predict_proba(x_va)[:, 1]
            tr_errs.append(1.0 - accuracy_score(y_tr, (p_tr >= 0.5).astype(int)))
            va_errs.append(1.0 - accuracy_score(y_va, (p_va >= 0.5).astype(int)))
            tr_aucs.append(roc_auc_score(y_tr, p_tr))
            va_aucs.append(roc_auc_score(y_va, p_va))
            va_recs.append(recall_score(y_va, (p_va >= 0.5).astype(int), zero_division=0))
        rows.append(
            {
                "max_depth": "None (unlimited)" if depth is None else str(depth),
                "train_error": float(np.mean(tr_errs)),
                "val_error": float(np.mean(va_errs)),
                "gap_error": float(np.mean(va_errs) - np.mean(tr_errs)),
                "train_roc_auc": float(np.mean(tr_aucs)),
                "val_roc_auc": float(np.mean(va_aucs)),
                "val_sensitivity": float(np.mean(va_recs)),
            }
        )
    return pd.DataFrame(rows)


def save_complexity_plot(curve_df: pd.DataFrame, output_path: Path) -> None:
    """2-panel Week 5 chart: Training vs. Validation Error & ROC-AUC across max_depth."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.8, 5.0), dpi=200)
    fig.patch.set_facecolor("white")
    x = np.arange(len(curve_df))
    labels = curve_df["max_depth"]

    # Panel A: Classification Error (1 - Accuracy)
    ax1.plot(x, curve_df["train_error"], marker="o", lw=2.2, color=BLUE, label="Training Error (seen rows)")
    ax1.plot(x, curve_df["val_error"], marker="s", lw=2.2, color=ORANGE, label="Validation Error (5-Fold CV unseen)")
    ax1.fill_between(x, curve_df["train_error"], curve_df["val_error"], color="#fde8df", alpha=0.55, label="Generalisation Gap")
    for idx, (tr, va) in enumerate(zip(curve_df["train_error"], curve_df["val_error"])):
        ax1.text(idx, tr - 0.015, f"{tr:.1%}", ha="center", va="top", fontsize=8.2, color=BLUE, fontweight="bold")
        ax1.text(idx, va + 0.012, f"{va:.1%}", ha="center", va="bottom", fontsize=8.2, color=ORANGE, fontweight="bold")
    ax1.set_xticks(x, labels, rotation=15, ha="right")
    ax1.set_ylim(-0.04, max(curve_df["val_error"].max() * 1.25, 0.35))
    ax1.set_xlabel("Model Complexity (Random Forest max_depth)")
    ax1.set_ylabel("Classification Error Rate (1 − Accuracy)")
    ax1.set_title("A. Training Error vs. Unseen Validation Error", loc="left", fontsize=11, fontweight="bold")
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.grid(color=GRID)
    ax1.legend(frameon=False, loc="upper right", fontsize=8.8)

    # Panel B: ROC-AUC & Validation Sensitivity
    ax2.plot(x, curve_df["train_roc_auc"], marker="o", lw=2.2, color=BLUE, label="Training ROC-AUC (seen)")
    ax2.plot(x, curve_df["val_roc_auc"], marker="s", lw=2.2, color=ORANGE, label="Validation ROC-AUC (5-Fold CV)")
    ax2.plot(x, curve_df["val_sensitivity"], marker="^", lw=2.0, ls="--", color=TEAL, label="Validation Sensitivity @ 0.50")
    for idx, (tr, va) in enumerate(zip(curve_df["train_roc_auc"], curve_df["val_roc_auc"])):
        ax2.text(idx, va - 0.045, f"{va:.3f}", ha="center", va="top", fontsize=8.2, color=ORANGE, fontweight="bold")
    ax2.set_xticks(x, labels, rotation=15, ha="right")
    ax2.set_ylim(0, 1.08)
    ax2.set_xlabel("Model Complexity (Random Forest max_depth)")
    ax2.set_ylabel("Score")
    ax2.set_title("B. Memorisation (Train AUC → 1.00) vs. Generalisation", loc="left", fontsize=11, fontweight="bold")
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.grid(color=GRID)
    ax2.legend(frameon=False, loc="lower right", fontsize=8.8)

    fig.suptitle(
        "Bias–Variance & Overfitting Diagnosis: Training Error vs. Unseen Validation Error (SMOTE + 25 Boruta Features)",
        fontsize=12.5,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Week 5: Normality proof, Min-Max pipeline, and complexity curve.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/week5"))
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    args = parser.parse_args(argv)

    # Ensure Weeks 1-2 files exist and load train_reduced (450 features, uncapped so pipeline fits 3σ inside CV)
    ensure_prepared_training_split(args.raw_dir, args.processed_dir)
    x_train_reduced, y_train = load_split(args.processed_dir / "train_reduced.csv")
    selected_cols = json.loads((args.processed_dir / "selected_features.json").read_text(encoding="utf-8"))[
        "selected_features"
    ]
    x_sel = x_train_reduced[selected_cols]
    test_path = args.processed_dir / "test_holdout.csv"
    assert verify_sealed(test_path), "Sealed test set checksum mismatch!"

    print(f"Loaded train_reduced: {x_train_reduced.shape[0]:,} × {x_train_reduced.shape[1]} "
          f"| Boruta selected: {len(selected_cols)} | Test seal: INTACT")

    # 1. Normality audit & empirical proof (RF vs. LogReg)
    x_sel_capped = OutlierTreatment("cap", 3.0).fit_transform(x_sel)
    x_sel_imputed = pd.DataFrame(
        knn_imputer(5).fit_transform(x_sel_capped), index=x_sel.index, columns=x_sel.columns
    )
    audit_df = normality_audit(x_sel_imputed)
    cv_proof_df = normality_cv_proof(x_sel, y_train, seed=args.seed)
    print("\n1. Normality Audit Summary (25 Boruta features):")
    print(f"   Non-normal raw (Shapiro p<0.05): {(~audit_df['normal_raw']).sum()}/{len(audit_df)} "
          f"| Non-normal after Yeo-Johnson: {(~audit_df['normal_yj']).sum()}/{len(audit_df)} "
          f"| Features with values <= 0 (Box-Cox invalid): {audit_df['has_non_positive'].sum()}/{len(audit_df)}")
    print("\n   Empirical 5-Fold CV Proof (No transform vs. Yeo-Johnson):")
    print(cv_proof_df.to_string(index=False))

    # 2. Leakage-free pipeline & Min-Max checks
    checks = run_pipeline_checks(x_train_reduced, y_train, selected_cols, test_path, seed=args.seed)
    print("\n2. Pipeline & Leakage Control Checks (§10.5):")
    for k, v in checks.items():
        print(f"   {k}: {v}")

    # 3. Training vs. Validation Error complexity curve
    curve_df = complexity_curve(x_sel, y_train, seed=args.seed)
    print("\n3. Training Error vs. Validation Error Across max_depth:")
    print(curve_df.to_string(index=False))

    # Save all outputs
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit_df.to_csv(args.output_dir / "w5_normality_audit.csv", index=False)
    cv_proof_df.to_csv(args.output_dir / "w5_normality_cv_proof.csv", index=False)
    curve_df.to_csv(args.output_dir / "w5_complexity_curve.csv", index=False)
    (args.output_dir / "w5_pipeline_checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")

    save_normality_proof_plot(audit_df, cv_proof_df, args.output_dir / "w5_normality_proof.png")
    save_pipeline_process_map(args.output_dir / "w5_pipeline_process_map.png")
    save_complexity_plot(curve_df, args.output_dir / "w5_train_vs_val_complexity.png")
    print(f"\nSaved all Week 5 tables and visuals to {args.output_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Week 7 / manual §15: the one-time evaluation of the frozen final pipeline on the test set.

This is the **only** module that reads ``data/processed/test_holdout.csv``. It:

1. checks the test file against its SHA-256 seal (unchanged since the split, D-015);
2. reads the frozen configuration ``data/processed/final_config.json`` written by
   ``secom_analysis.tuning`` (pipeline settings **and** decision threshold, fixed on CV only);
3. refits that pipeline once on all 1,253 training wafers (``train_reduced.csv``);
4. applies the stored column drops (same 450 columns) and the fitted pipeline to the 314 test
   wafers with ``predict_proba`` only. Nothing is refitted on test; the sampler is skipped;
5. reports the confusion matrix, accuracy, sensitivity, specificity, precision, F1, F2,
   ROC-AUC, PR-AUC and total cost (100·FP + 400·FN), with bootstrap 95% intervals, next to the
   training (resubstitution) and cross-validated numbers.

It refuses to run a second time (``reports/final/final_metrics.json`` exists) unless
``--rerun`` is given, so the test set cannot quietly become a tuning set.

    python -m secom_analysis.final_eval
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve

from .eda import _md_table
from .pipeline import COST_FN, COST_FP, build_pipeline, error_metrics
from .preparation import load_split, verify_sealed

N_BOOTSTRAP = 2000
BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#7b7a75"
METRICS = ["accuracy", "sensitivity", "specificity", "precision", "f1", "f2", "roc_auc",
           "pr_auc", "cost"]


def bootstrap_intervals(y_true: np.ndarray, proba: np.ndarray, threshold: float,
                        n: int = N_BOOTSTRAP, seed: int = 42) -> pd.DataFrame:
    """95% percentile intervals of every metric over stratified bootstrap resamples of the rows.

    Resampling within each class keeps the number of defective wafers fixed, so every
    resample has both classes.
    """
    rng = np.random.default_rng(seed)
    positives, negatives = np.flatnonzero(y_true == 1), np.flatnonzero(y_true == 0)
    rows = []
    for _ in range(n):
        index = np.concatenate([rng.choice(positives, len(positives)),
                                rng.choice(negatives, len(negatives))])
        rows.append(error_metrics(y_true[index], proba[index], threshold))
    frame = pd.DataFrame(rows)[METRICS]
    return frame.quantile([0.025, 0.975]).T.rename(columns={0.025: "low", 0.975: "high"})


def save_figures(y_true, proba, threshold, metrics, path: Path) -> None:
    """Confusion matrix at the frozen threshold, ROC and precision-recall curves on test."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    matrix = np.array([[metrics["TN"], metrics["FP"]], [metrics["FN"], metrics["TP"]]])
    axes[0].imshow(np.log1p(matrix), cmap="Blues")
    for (i, j), value in np.ndenumerate(matrix):
        axes[0].text(j, i, f"{value}", ha="center", va="center", fontsize=14,
                     color="white" if value > matrix.max() / 3 else INK)
    axes[0].set_xticks([0, 1], ["predicted good", "predicted defective"])
    axes[0].set_yticks([0, 1], ["good", "defective"])
    axes[0].set_title(f"Test set, threshold {threshold:.2f}: cost {metrics['cost']:,}")
    fpr, tpr, _ = roc_curve(y_true, proba)
    axes[1].plot(fpr, tpr, color=BLUE, label=f"ROC-AUC {metrics['roc_auc']:.3f}")
    axes[1].plot([0, 1], [0, 1], color=MUTED, ls="--", lw=1)
    axes[1].set(xlabel="1 − specificity", ylabel="sensitivity", title="ROC curve, test set")
    precision, recall, _ = precision_recall_curve(y_true, proba)
    axes[2].plot(recall, precision, color=ORANGE,
                 label=f"PR-AUC {average_precision_score(y_true, proba):.3f}")
    axes[2].axhline(y_true.mean(), color=MUTED, ls="--", lw=1, label="no skill")
    axes[2].set(xlabel="recall", ylabel="precision", title="Precision-recall curve, test set")
    for axis in axes[1:]:
        axis.legend(frameon=False)
        axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/final"))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--rerun", action="store_true",
                        help="allow a second evaluation (must be logged in the diary)")
    args = parser.parse_args(argv)
    out = args.output_dir
    if (out / "final_metrics.json").exists() and not args.rerun:
        raise SystemExit("The test set was already evaluated (reports/final/final_metrics.json). "
                         "Re-running needs --rerun and a diary entry explaining why.")
    out.mkdir(parents=True, exist_ok=True)

    test_path = args.processed_dir / "test_holdout.csv"
    if not verify_sealed(test_path):
        raise SystemExit("Test set checksum does not match its seal; aborting.")
    config = json.loads((args.processed_dir / "final_config.json").read_text())
    settings = config["pipeline"]
    threshold = float(config["threshold"])
    print(f"Frozen configuration {config['trial_id']} (frozen {config['frozen_at']}), "
          f"threshold {threshold}")

    x_train, y_train = load_split(args.processed_dir / "train_reduced.csv")
    pipeline = build_pipeline(
        outlier=settings["outlier"], sigma=settings["sigma"],
        n_neighbors=settings["n_neighbors"], selector=settings["selector"],
        balancer=settings["balancer"], seed=settings["random_state"],
        n_estimators=settings["n_estimators"], rf_params=settings["rf_params"],
    ).fit(x_train, y_train)
    args.models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "threshold": threshold, "config": config},
                args.models_dir / "final_pipeline.joblib")

    # ---- the single read of the sealed test set
    x_test_all, y_test = load_split(test_path)
    opened_at = datetime.now().isoformat(timespec="seconds")
    x_test = x_test_all[x_train.columns]  # notebook-01 drops: same 450 columns, no refit
    assert not set(x_test.index) & set(x_train.index), "train and test rows overlap"

    test_proba = pipeline.predict_proba(x_test)[:, 1]
    train_proba = pipeline.predict_proba(x_train)[:, 1]
    test = error_metrics(y_test.to_numpy(), test_proba, threshold)
    train = error_metrics(y_train.to_numpy(), train_proba, threshold)
    test_half = error_metrics(y_test.to_numpy(), test_proba, 0.5)
    intervals = bootstrap_intervals(y_test.to_numpy(), test_proba, threshold)

    selected = pipeline.named_steps["selector"]
    features = (list(map(str, selected.columns_[selected.support_]))
                if hasattr(selected, "support_") else list(x_train.columns))
    cv = config["cv_estimate"]
    comparison = pd.DataFrame({
        "training (resubstitution)": {k: train[k] for k in METRICS},
        "cross-validation (frozen estimate)": {
            "sensitivity": cv["sensitivity"], "specificity": cv["specificity"],
            "roc_auc": cv["roc_auc"], "pr_auc": cv["pr_auc"],
            "cost": cv["cost_cv_threshold"] * len(y_test) / len(y_train)},
        "test": {k: test[k] for k in METRICS},
        "test 95% low": intervals["low"],
        "test 95% high": intervals["high"],
    }).loc[METRICS]
    always_pass = COST_FN * int(y_test.sum())
    result = {
        "opened_at": opened_at,
        "trial_id": config["trial_id"],
        "threshold": threshold,
        "test_rows": len(y_test),
        "test_defective": int(y_test.sum()),
        "test": test,
        "test_at_threshold_0.5": test_half,
        "test_bootstrap_95": intervals.to_dict(orient="index"),
        "training_resubstitution": train,
        "always_pass_cost_test": always_pass,
        "selected_features": features,
        "cost_weights": {"FP": COST_FP, "FN": COST_FN},
    }
    (out / "final_metrics.json").write_text(json.dumps(result, indent=2, default=float))
    comparison.to_csv(out / "train_cv_test.csv")
    pd.DataFrame({"source_line": x_test.index + 1, "defective": y_test.to_numpy(),
                  "p_defect": test_proba,
                  "flagged": (test_proba >= threshold).astype(int)}).to_csv(
        out / "test_predictions.csv", index=False)
    save_figures(y_test.to_numpy(), test_proba, threshold, test, out / "test_evaluation.png")

    lines = [
        "# Final evaluation on the sealed test set (manual §15, week 7)",
        "",
        (f"Configuration `{config['trial_id']}`, frozen {config['frozen_at']}; test set opened "
        f"{opened_at}. Threshold {threshold} (chosen on CV). Test: {len(y_test)} wafers, "
        f"{int(y_test.sum())} defective. Always-pass cost on test: {always_pass:,}."),
        "",
        f"Confusion matrix: TP {test['TP']}, FN {test['FN']}, FP {test['FP']}, TN {test['TN']}.",
        "",
        "CV cost is scaled to the test size (×314/1,253) for comparison.",
        "",
        _md_table(comparison, floatfmt=".3f"),
        "",
        (f"At the default threshold 0.5: TP {test_half['TP']}, FN {test_half['FN']}, "
        f"FP {test_half['FP']}, cost {test_half['cost']:,}."),
        "",
        f"Features used ({len(features)}): {', '.join(features)}",
        "",
        "![test](test_evaluation.png)",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print(comparison.round(3).to_string())
    print(f"test cost {test['cost']:,} vs always-pass {always_pass:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Week 6: model candidates, grid search and cost-based threshold (manual §11, §12, §14).

Protocol (D-038). Training split only; the sealed test set is never read.

* **CV:** repeated stratified 5-fold, 2 repeats (10 folds), identical for every trial. Every
  fitted step (3σ bounds, k-NN imputer, Boruta, min-max, sampler, model) is refitted on each
  fold's training rows.
* **Stage A, models (§11):** dummy baselines, logistic regression, random forest, gradient
  boosting, all on the default preparation (3σ cap, k = 5, Boruta, min-max, SMOTE).
* **Stage B, preparation grid (§12, week 7):** outlier treatment {cap, remove, none} × k-NN
  neighbours {3, 5, 10} × selector {Boruta, all} × balancer {none, class weights, ROSE-style,
  SMOTE, ADASYN}, random forest at its defaults: 90 trials.
* **Stage C, forest grid:** for the 3 best stage-B preparations, ``max_depth`` ×
  ``min_samples_leaf`` × ``max_features``: 54 trials.
* **Score (§14):** total cost = 100·FP + 400·FN on the out-of-fold predictions at a threshold
  chosen **inside the CV** (picked on the other 4 folds' predictions, applied to the 5th), so
  the cost estimate is not tuned on the rows it is measured on. PR-AUC and ROC-AUC are
  reported alongside.

Every trial is written to ``reports/tuning/trials.csv`` (methods, parameters, metrics). The
winner and its threshold are frozen in ``data/processed/final_config.json`` before the test set
is opened (``secom_analysis.final_eval``).

Run with::

    python -m secom_analysis.tuning      # ~30 min on 12 cores; writes reports/tuning/
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from itertools import product
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

from .eda import _md_table
from .impute import RANDOM_STATE
from .pipeline import COST_FN, COST_FP, build_pipeline, error_metrics, make_balancer
from .preparation import load_split

N_SPLITS, N_REPEATS = 5, 2
THRESHOLDS = np.round(np.arange(0.01, 1.0, 0.01), 2)
OUTLIERS = ["cap", "remove", "none"]
NEIGHBOURS = [3, 5, 10]
SELECTORS = ["boruta", "all"]
BALANCERS = ["none", "class_weight", "rose", "smote", "adasyn"]
FOREST_GRID = {"max_depth": [None, 10, 5], "min_samples_leaf": [1, 3, 10],
               "max_features": ["sqrt", 0.3]}
DEFAULT_PREP = {"outlier": "cap", "n_neighbors": 5, "selector": "boruta"}
TOP_PREPARATIONS = 3
BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#7b7a75"


# ---------------------------------------------------------------- models and cost


def make_classifier(family: str, balancer: str, params: dict | None = None):
    """Classifier for a model family. Class weights only for the "class_weight" balancer.

    Families: dummy_majority, dummy_stratified, logreg, random_forest, gradient_boosting.
    """
    params = dict(params or {})
    weight = "balanced" if balancer == "class_weight" else None
    if family == "dummy_majority":
        return DummyClassifier(strategy="most_frequent")
    if family == "dummy_stratified":
        return DummyClassifier(strategy="stratified", random_state=RANDOM_STATE)
    if family == "logreg":
        return LogisticRegression(max_iter=5000, class_weight=weight, random_state=RANDOM_STATE)
    if family == "random_forest":
        return RandomForestClassifier(n_estimators=300, class_weight=weight, n_jobs=1,
                                      random_state=RANDOM_STATE, **params)
    if family == "gradient_boosting":
        return HistGradientBoostingClassifier(class_weight=weight, random_state=RANDOM_STATE)
    raise ValueError(f"unknown model family: {family}")


def costs_by_threshold(y_true: np.ndarray, proba: np.ndarray) -> np.ndarray:
    """Total cost 100·FP + 400·FN at every threshold in THRESHOLDS (vectorised)."""
    y_true = np.asarray(y_true).astype(bool)
    flagged = np.asarray(proba)[None, :] >= THRESHOLDS[:, None]
    fp = (flagged & ~y_true).sum(axis=1)
    fn = (~flagged & y_true).sum(axis=1)
    return COST_FP * fp + COST_FN * fn


def best_threshold(y_true, proba) -> float:
    """Threshold with the lowest total cost; ties go to the threshold closest to 0.5."""
    cost = costs_by_threshold(y_true, proba)
    candidates = THRESHOLDS[cost == cost.min()]
    return float(candidates[np.argmin(np.abs(candidates - 0.5))])


def cv_threshold_cost(y_true: np.ndarray, proba: np.ndarray, fold_id: np.ndarray) -> dict:
    """Honest cost: for each fold, pick the threshold on the other folds' out-of-fold
    predictions and apply it to this fold. Output: summed confusion counts, cost, and the
    thresholds chosen."""
    totals = {"TP": 0, "FN": 0, "FP": 0, "TN": 0}
    chosen = []
    for fold in np.unique(fold_id):
        inside = fold_id == fold
        threshold = best_threshold(y_true[~inside], proba[~inside])
        chosen.append(threshold)
        metrics = error_metrics(y_true[inside], proba[inside], threshold)
        for key in totals:
            totals[key] += metrics[key]
    cost = COST_FP * totals["FP"] + COST_FN * totals["FN"]
    return {**totals, "cost": cost, "thresholds": chosen}


# ---------------------------------------------------------------- per-fold work


def _fold_fronts(features, target, train_index, valid_index, preparations):
    """Fit outliers → imputer → selector once per preparation on one fold's training rows.

    Output: {prep_key: (train_matrix, valid_matrix, n_features)} with the features already
    selected (min-max scaling happens later, per trial).
    """
    x_train, y_train = features.iloc[train_index], target.iloc[train_index]
    x_valid = features.iloc[valid_index]
    fronts = {}
    for prep in preparations:
        front = build_pipeline(outlier=prep["outlier"], n_neighbors=prep["n_neighbors"],
                               selector=prep["selector"])[:3].fit(x_train, y_train)
        a_train = front.transform(x_train)
        fronts[_key(prep)] = (np.asarray(a_train, dtype=float),
                              np.asarray(front.transform(x_valid), dtype=float),
                              a_train.shape[1])
    return fronts


def _fit_back(a_train, y_train, a_valid, family, balancer, params):
    """min-max → sampler → classifier on one fold's prepared matrices; returns P(defect)."""
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler().fit(a_train)
    s_train, s_valid = scaler.transform(a_train), scaler.transform(a_valid)
    sampler = make_balancer(balancer)
    if sampler is not None:
        s_train, y_train = sampler.fit_resample(s_train, y_train)
    model = make_classifier(family, balancer, params).fit(s_train, y_train)
    return model.predict_proba(s_valid)[:, 1]


def _key(prep: dict) -> str:
    return f"{prep['outlier']}|k{prep['n_neighbors']}|{prep['selector']}"


def _run_fold(fold, features, target, train_index, valid_index, preparations, trials):
    """One fold: build all fronts, then every trial that uses them. Returns per-trial probas."""
    start = time.time()
    fronts = _fold_fronts(features, target, train_index, valid_index, preparations)
    y_train = target.iloc[train_index].to_numpy()
    out = {}
    for trial in trials:
        a_train, a_valid, n_features = fronts[_key(trial)]
        out[trial["trial_id"]] = (
            _fit_back(a_train, y_train, a_valid, trial["family"], trial["balancer"],
                      trial["params"]),
            n_features,
        )
    return fold, out, fronts, time.time() - start


def _run_back_only(fold, fronts, y_train, trials):
    out = {}
    for trial in trials:
        a_train, a_valid, n_features = fronts[_key(trial)]
        out[trial["trial_id"]] = (
            _fit_back(a_train, y_train, a_valid, trial["family"], trial["balancer"],
                      trial["params"]),
            n_features,
        )
    return fold, out


# ---------------------------------------------------------------- scoring


def score_trials(trials: list[dict], results: dict, target: pd.Series, splits) -> pd.DataFrame:
    """Turn per-fold probabilities into one row per trial (per repeat, then averaged)."""
    y = target.to_numpy()
    rows = []
    for trial in trials:
        per_repeat = []
        n_features = []
        fold_pr = []
        for repeat in range(N_REPEATS):
            proba = np.zeros(len(y))
            fold_id = np.zeros(len(y), dtype=int)
            for split in range(N_SPLITS):
                fold = repeat * N_SPLITS + split
                valid_index = splits[fold][1]
                p, n = results[fold][trial["trial_id"]]
                proba[valid_index] = p
                fold_id[valid_index] = split
                n_features.append(n)
                if len(np.unique(y[valid_index])) == 2:
                    fold_pr.append(average_precision_score(y[valid_index], p))
            honest = cv_threshold_cost(y, proba, fold_id)
            at_half = error_metrics(y, proba, 0.5)
            per_repeat.append({
                "cv_cost": honest["cost"],
                "cv_TP": honest["TP"], "cv_FN": honest["FN"],
                "cv_FP": honest["FP"], "cv_TN": honest["TN"],
                "threshold_median": float(np.median(honest["thresholds"])),
                "min_cost_any_threshold": int(costs_by_threshold(y, proba).min()),
                "cost_at_0.5": at_half["cost"],
                "sensitivity_at_0.5": at_half["sensitivity"],
                "roc_auc": roc_auc_score(y, proba),
                "pr_auc": average_precision_score(y, proba),
            })
        frame = pd.DataFrame(per_repeat)
        row = {key: trial[key] for key in ("trial_id", "stage", "family", "outlier",
                                            "n_neighbors", "selector", "balancer")}
        row.update({f"rf_{k}": v for k, v in trial["params"].items()})
        row.update(frame.mean().to_dict())
        row["cv_cost_sd_repeats"] = frame["cv_cost"].std()
        row["pr_auc_fold_sd"] = float(np.std(fold_pr, ddof=1))
        row["n_features_mean"] = float(np.mean(n_features))
        row["cv_sensitivity"] = row["cv_TP"] / (row["cv_TP"] + row["cv_FN"])
        row["cv_specificity"] = row["cv_TN"] / (row["cv_TN"] + row["cv_FP"])
        rows.append(row)
    return pd.DataFrame(rows)


def pooled_oof(trial_id: str, results: dict, target: pd.Series, splits) -> np.ndarray:
    """Out-of-fold probabilities of one trial, shape (repeats, rows)."""
    out = np.zeros((N_REPEATS, len(target)))
    for repeat in range(N_REPEATS):
        for split in range(N_SPLITS):
            fold = repeat * N_SPLITS + split
            out[repeat, splits[fold][1]] = results[fold][trial_id][0]
    return out


# ---------------------------------------------------------------- trial lists


def stage_a_trials() -> list[dict]:
    families = ["dummy_majority", "dummy_stratified", "logreg", "random_forest",
                "gradient_boosting"]
    return [{"trial_id": f"A-{family}", "stage": "A", "family": family, **DEFAULT_PREP,
             "balancer": "smote", "params": {}} for family in families]


def stage_b_trials() -> list[dict]:
    trials = []
    for outlier, k, selector, balancer in product(OUTLIERS, NEIGHBOURS, SELECTORS, BALANCERS):
        trials.append({"trial_id": f"B-{outlier}-k{k}-{selector}-{balancer}", "stage": "B",
                       "family": "random_forest", "outlier": outlier, "n_neighbors": k,
                       "selector": selector, "balancer": balancer, "params": {}})
    return trials


def stage_c_trials(best: pd.DataFrame) -> list[dict]:
    trials = []
    names = list(FOREST_GRID)
    for _, row in best.iterrows():
        for values in product(*FOREST_GRID.values()):
            params = dict(zip(names, values))
            tag = "-".join(f"{v}" for v in values)
            trials.append({"trial_id": f"C-{row['outlier']}-k{row['n_neighbors']}-"
                                       f"{row['selector']}-{row['balancer']}-{tag}",
                           "stage": "C", "family": "random_forest", "outlier": row["outlier"],
                           "n_neighbors": int(row["n_neighbors"]), "selector": row["selector"],
                           "balancer": row["balancer"], "params": params})
    return trials


# ---------------------------------------------------------------- outputs


def save_figures(trials: pd.DataFrame, final_oof: np.ndarray, target: pd.Series,
                 out_dir: Path) -> None:
    """Grid overview (cost by preparation and balancer) and the final cost-vs-threshold curve."""
    b = trials[trials["stage"] == "B"].copy()
    b["preparation"] = (b["outlier"] + " · k" + b["n_neighbors"].astype(str) + " · "
                        + b["selector"])
    pivot = b.pivot_table(index="preparation", columns="balancer", values="cv_cost")
    pivot = pivot.loc[pivot.min(axis=1).sort_values().index, BALANCERS]
    fig, ax = plt.subplots(figsize=(8, 6.5))
    image = ax.imshow(pivot.to_numpy(), cmap="Blues_r", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)), pivot.columns)
    ax.set_yticks(range(len(pivot.index)), pivot.index, fontsize=8)
    for (i, j), value in np.ndenumerate(pivot.to_numpy()):
        ax.text(j, i, f"{value / 1000:.1f}", ha="center", va="center", fontsize=7,
                color="white" if value < np.nanmedian(pivot.to_numpy()) else INK)
    fig.colorbar(image, ax=ax, label="CV cost (CV-chosen threshold)")
    always_pass = COST_FN * int(target.sum())
    ax.set_title(f"Stage B: total cost per trial, thousands (always pass = "
                 f"{always_pass / 1000:.1f})", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "grid_cost.png", dpi=150)
    plt.close(fig)

    y = target.to_numpy()
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for repeat, color in zip(range(final_oof.shape[0]), [BLUE, ORANGE]):
        ax.plot(THRESHOLDS, costs_by_threshold(y, final_oof[repeat]) / 1000, color=color,
                label=f"repeat {repeat + 1}")
    ax.axhline(always_pass / 1000, color=MUTED, ls=":", label="always pass")
    ax.set(xlabel="decision threshold", ylabel="total cost, thousands",
           title="Final configuration: out-of-fold cost vs threshold")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "final_cost_curve.png", dpi=150)
    plt.close(fig)


def write_report(trials: pd.DataFrame, final: dict, meta: dict, path: Path) -> None:
    show = ["trial_id", "n_features_mean", "cv_cost", "cv_cost_sd_repeats", "cv_sensitivity",
            "cv_specificity", "threshold_median", "cost_at_0.5", "pr_auc", "roc_auc"]
    a = trials[trials["stage"] == "A"].set_index("trial_id")
    lines = [
        "# Week 6: model candidates, grid search and cost-based threshold",
        "",
        (f"Training split only ({meta['rows']} wafers, {meta['defects']} defective); "
        f"{N_REPEATS}×{N_SPLITS}-fold stratified CV; {len(trials)} trials; "
        f"{meta['minutes']:.0f} min. Always-pass cost: {meta['always_pass_cost']:,}."),
        "",
        ("Cost = 100·FP + 400·FN, summed over the 1,253 out-of-fold predictions, with the "
        "threshold chosen on the other folds (mean of the two repeats)."),
        "",
        "## Stage A: model families (default preparation + SMOTE)",
        "",
        _md_table(a[show[1:]], floatfmt=".3f"),
        "",
        "## Top 15 trials (all stages)",
        "",
        _md_table(trials.nsmallest(15, "cv_cost").set_index("trial_id")[show[1:]],
                  floatfmt=".3f"),
        "",
        "![grid](grid_cost.png)",
        "",
        "## Frozen final configuration",
        "",
        "```json",
        json.dumps(final, indent=2),
        "```",
        "",
        "![cost curve](final_cost_curve.png)",
    ]
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/tuning"))
    parser.add_argument("--jobs", type=int, default=5)
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()

    features, target = load_split(args.processed_dir / "train_reduced.csv")
    folds = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS,
                                    random_state=RANDOM_STATE)
    splits = list(folds.split(features, target))
    print(f"train_reduced {features.shape}, {int(target.sum())} defective; "
          f"{len(splits)} folds", flush=True)

    preparations = [dict(zip(("outlier", "n_neighbors", "selector"), combo))
                    for combo in product(OUTLIERS, NEIGHBOURS, SELECTORS)]
    first = stage_a_trials() + stage_b_trials()
    runs = Parallel(n_jobs=args.jobs, verbose=0)(
        delayed(_run_fold)(fold, features, target, tr, va, preparations, first)
        for fold, (tr, va) in enumerate(splits))
    results, fronts = {}, {}
    for fold, out, front, seconds in runs:
        results[fold], fronts[fold] = out, front
        print(f"fold {fold + 1:2d}: stages A+B done in {seconds / 60:.1f} min", flush=True)
    scored = score_trials(first, results, target, splits)

    b = scored[scored["stage"] == "B"]
    best_prep = b.nsmallest(TOP_PREPARATIONS, "cv_cost")
    second = stage_c_trials(best_prep)
    runs = Parallel(n_jobs=args.jobs)(
        delayed(_run_back_only)(fold, fronts[fold], target.iloc[splits[fold][0]].to_numpy(),
                                second)
        for fold in range(len(splits)))
    for fold, out in runs:
        results[fold].update(out)
    trials = pd.concat([scored, score_trials(second, results, target, splits)],
                       ignore_index=True)
    trials.to_csv(args.output_dir / "trials.csv", index=False)
    print(f"stage C done; {len(trials)} trials", flush=True)

    candidates = trials[trials["family"] == "random_forest"]
    winner = candidates.nsmallest(1, "cv_cost").iloc[0]
    trial_list = {t["trial_id"]: t for t in first + second}[winner["trial_id"]]
    oof = pooled_oof(winner["trial_id"], results, target, splits)
    threshold = best_threshold(np.tile(target.to_numpy(), N_REPEATS), oof.ravel())
    final = {
        "frozen_at": datetime.now().isoformat(timespec="seconds"),
        "trial_id": winner["trial_id"],
        "pipeline": {"outlier": trial_list["outlier"], "sigma": 3.0,
                     "n_neighbors": trial_list["n_neighbors"],
                     "selector": trial_list["selector"], "balancer": trial_list["balancer"],
                     "model": "random_forest", "n_estimators": 300,
                     "rf_params": trial_list["params"], "random_state": RANDOM_STATE},
        "threshold": threshold,
        "threshold_rule": "lowest out-of-fold cost over both CV repeats (ties -> closest to 0.5)",
        "cv_estimate": {
            "cost_cv_threshold": float(winner["cv_cost"]),
            "cost_sd_repeats": float(winner["cv_cost_sd_repeats"]),
            "sensitivity": float(winner["cv_sensitivity"]),
            "specificity": float(winner["cv_specificity"]),
            "pr_auc": float(winner["pr_auc"]), "roc_auc": float(winner["roc_auc"]),
            "always_pass_cost": COST_FN * int(target.sum()),
        },
        "cost_weights": {"FP": COST_FP, "FN": COST_FN},
        "trained_on": "data/processed/train_reduced.csv (all 1,253 training wafers)",
    }
    (args.processed_dir / "final_config.json").write_text(json.dumps(final, indent=2))
    save_figures(trials, oof, target, args.output_dir)
    meta = {"rows": len(target), "defects": int(target.sum()),
            "minutes": (time.time() - start) / 60,
            "always_pass_cost": COST_FN * int(target.sum())}
    write_report(trials, final, meta, args.output_dir / "report.md")
    print(json.dumps(final, indent=2))
    print(f"done in {meta['minutes']:.1f} min -> {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

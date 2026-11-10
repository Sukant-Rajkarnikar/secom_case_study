"""Build one imputed training dataset per imputation strategy and compare them.

Input is the prepared training split from notebook 01 (``data/processed/train_prepared.csv``:
450 features after zero-variance and >60% missing removal, 3σ-capped, NaN = missing).
The sealed test set is never read here. Each strategy is fitted on the training split only
and saved with ``joblib`` so the same fitted transform can be applied to the test set at
final evaluation.

Strategies (D-021):

* ``median``          - per-column median (simplest baseline).
* ``median_flags``    - median plus one 0/1 flag per *missingness block* (columns that are
                        always missing together share a flag), so "was missing" stays visible.
* ``knn``             - k-NN (k=5, distance-weighted) on standardised features.
* ``mice_bayesridge`` - MICE / IterativeImputer with BayesianRidge, standardised features.
* ``mice_extratrees`` - MICE with ExtraTrees (MissForest-style, non-linear).

Every strategy is compared without using the label:

* **masked reconstruction** - hide observed cells using the missingness pattern of another
  randomly chosen wafer (keeps the real block structure), impute, and measure the error on
  the hidden cells in standard-deviation units (NRMSE);
* **distribution consistency** - per-column variance ratio and mean shift of the imputed
  column vs the observed values, and the KS statistic between imputed and observed values;
* **correlation distortion** - mean |Δr| between the observed (pairwise-complete)
  correlation matrix and the imputed one.

Run with::

    python -m secom_analysis.impute               # build, compare, load into Postgres
    python -m secom_analysis.impute --no-db       # files and report only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.linear_model import BayesianRidge

ID_COLUMN, TARGET_COLUMN = "source_line", "defective"
RANDOM_STATE = 42
# Distribution metrics are summarised over columns with at least this much missing data:
# a column with 5 of 1,253 values filled looks the same whatever the strategy.
MATERIAL_MISSING_PCT = 5.0


# ---------------------------------------------------------------- imputers


class ScaledImputer(BaseEstimator, TransformerMixin):
    """Standardise (NaN-aware), impute, then map back to the original units.

    Only the cells that were missing are filled; observed values are returned unchanged.
    Filled values are clipped to the observed training range of their column, so a
    regression-based imputer cannot extrapolate past what the sensor ever reported.
    """

    def __init__(self, imputer: BaseEstimator) -> None:
        self.imputer = imputer

    def fit(self, X: pd.DataFrame, y: object = None) -> ScaledImputer:
        values = np.asarray(X, dtype=float)
        self.mean_ = np.nanmean(values, axis=0)
        std = np.nanstd(values, axis=0, ddof=1)
        self.scale_ = np.where(std > 0, std, 1.0)
        self.min_ = np.nanmin(values, axis=0)
        self.max_ = np.nanmax(values, axis=0)
        self.imputer_ = clone(self.imputer).fit((values - self.mean_) / self.scale_)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        values = np.asarray(X, dtype=float)
        missing = np.isnan(values)
        filled = self.imputer_.transform((values - self.mean_) / self.scale_)
        filled = np.clip(filled * self.scale_ + self.mean_, self.min_, self.max_)
        return np.where(missing, filled, values)


class MedianFlagImputer(BaseEstimator, TransformerMixin):
    """Median imputation plus one missing-indicator per block of co-missing columns.

    Columns whose missing-value masks are identical on the training data form one block.
    Blocks missing on fewer than ``min_flag_pct`` % of training rows get no flag (a flag
    that is 1 on two wafers is noise, not signal).
    """

    def __init__(self, min_flag_pct: float = 1.0) -> None:
        self.min_flag_pct = min_flag_pct

    def fit(self, X: pd.DataFrame, y: object = None) -> MedianFlagImputer:
        frame = pd.DataFrame(X)
        self.median_ = SimpleImputer(strategy="median").fit(frame.to_numpy(dtype=float))
        mask = frame.isna()
        blocks: dict[bytes, list[str]] = {}
        for column in frame.columns[mask.any()]:
            blocks.setdefault(mask[column].to_numpy().tobytes(), []).append(str(column))
        kept = [
            members
            for members in blocks.values()
            if mask[members[0]].mean() * 100 >= self.min_flag_pct
        ]
        kept.sort(key=lambda members: (-mask[members[0]].mean(), members[0]))
        self.blocks_ = {f"missing_flag_{i:02d}": members for i, members in enumerate(kept, 1)}
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = pd.DataFrame(X)
        filled = pd.DataFrame(
            self.median_.transform(frame.to_numpy(dtype=float)),
            index=frame.index,
            columns=frame.columns,
        )
        # A wafer is flagged when the block's representative column is missing.
        flags = pd.DataFrame(
            {
                name: frame[members[0]].isna().astype("int8")
                for name, members in self.blocks_.items()
            },
            index=frame.index,
        )
        return pd.concat([filled, flags], axis=1)


@dataclass(frozen=True)
class Strategy:
    name: str
    label: str
    description: str
    params: dict

    def build(self) -> BaseEstimator:
        if self.name == "median":
            return SimpleImputer(strategy="median")
        if self.name == "median_flags":
            return MedianFlagImputer(**self.params)
        if self.name == "knn":
            return ScaledImputer(KNNImputer(**self.params))
        if self.name == "mice_bayesridge":
            return ScaledImputer(_iterative(BayesianRidge(), **self.params))
        if self.name == "mice_extratrees":
            estimator = ExtraTreesRegressor(
                n_estimators=30,
                min_samples_leaf=3,
                max_features=0.5,
                n_jobs=-1,
                random_state=RANDOM_STATE,
            )
            return ScaledImputer(_iterative(estimator, **self.params))
        raise ValueError(self.name)


def _iterative(estimator: BaseEstimator, **params: object) -> IterativeImputer:
    return IterativeImputer(
        estimator=estimator,
        skip_complete=True,
        random_state=RANDOM_STATE,
        tol=1e-3,
        **params,
    )


STRATEGIES = [
    Strategy("median", "Median", "Per-column median of the training split.", {}),
    Strategy(
        "median_flags",
        "Median + missing flags",
        "Median, plus one 0/1 flag per block of columns that are always missing together "
        "(blocks missing on >= 1% of wafers).",
        {"min_flag_pct": 1.0},
    ),
    Strategy(
        "knn",
        "k-NN (k=5)",
        "Distance-weighted mean of the 5 most similar wafers, on standardised features.",
        {"n_neighbors": 5, "weights": "distance"},
    ),
    Strategy(
        "mice_bayesridge",
        "MICE · BayesianRidge",
        "Chained linear regressions (each column from its 40 most correlated columns), "
        "10 rounds, standardised features.",
        {"max_iter": 10, "n_nearest_features": 40},
    ),
    Strategy(
        "mice_extratrees",
        "MICE · ExtraTrees",
        "Chained ExtraTrees regressions (MissForest-style, non-linear), 30 trees, "
        "40 nearest columns, 5 rounds.",
        {"max_iter": 5, "n_nearest_features": 40},
    ),
]
STRATEGY_BY_NAME = {s.name: s for s in STRATEGIES}


# ---------------------------------------------------------------- evaluation


def pattern_mask(features: pd.DataFrame, seed: int = RANDOM_STATE) -> np.ndarray:
    """Boolean mask of observed cells to hide, copied from another wafer's missing pattern.

    For each wafer i a random wafer j != i is drawn; cells missing for j but observed for i
    are hidden. This keeps the real block structure of the missingness (whole sensor groups
    go missing together) instead of hiding random single cells.
    """
    rng = np.random.default_rng(seed)
    missing = features.isna().to_numpy()
    n = len(features)
    donors = (np.arange(n) + rng.integers(1, n, size=n)) % n
    return missing[donors] & ~missing


def masked_reconstruction(
    strategy: Strategy,
    features: pd.DataFrame,
    seed: int = RANDOM_STATE,
    cells_out: Path | None = None,
) -> dict[str, float]:
    """Hide observed cells, impute with a freshly fitted strategy, score the hidden cells.

    With ``cells_out`` the hidden cells are also written as a long table
    (row, column, true value, imputed value, both in original and standardised units).
    """
    hide = pattern_mask(features, seed)
    masked = features.mask(hide)
    imputed = _impute_values(strategy.build().fit(masked), masked)
    truth = features.to_numpy(dtype=float)
    std = np.nanstd(truth, axis=0, ddof=1)
    std = np.where(std > 0, std, 1.0)
    scaled = (imputed - truth) / std
    error = scaled[hide]
    material = (features.isna().mean() * 100 >= MATERIAL_MISSING_PCT).to_numpy()
    material_error = scaled[:, material][hide[:, material]]
    if cells_out is not None:
        rows, cols = np.nonzero(hide)
        mean = np.nanmean(truth, axis=0)
        pd.DataFrame(
            {
                "row": rows,
                "column": features.columns[cols],
                "true": truth[rows, cols],
                "imputed": imputed[rows, cols],
                "true_std_units": (truth[rows, cols] - mean[cols]) / std[cols],
                "imputed_std_units": (imputed[rows, cols] - mean[cols]) / std[cols],
            }
        ).to_csv(cells_out, index=False)
    return {
        "masked_cells": int(hide.sum()),
        "nrmse": float(np.sqrt(np.mean(error**2))),
        "nmae": float(np.mean(np.abs(error))),
        "nrmse_material": float(np.sqrt(np.mean(material_error**2))),
    }


def _impute_values(fitted: BaseEstimator, features: pd.DataFrame) -> np.ndarray:
    """Imputed feature values only (drops any flag columns), as a float array."""
    out = fitted.transform(features)
    if isinstance(out, pd.DataFrame):
        out = out[features.columns].to_numpy(dtype=float)
    return np.asarray(out, dtype=float)[:, : features.shape[1]]


def distribution_consistency(observed: pd.DataFrame, imputed: pd.DataFrame) -> pd.DataFrame:
    """Per-column comparison of the imputed column against its observed values."""
    missing = observed.isna()
    rows = []
    for column in observed.columns[missing.any()]:
        seen = observed[column].dropna()
        filled = imputed.loc[missing[column], column]
        full = imputed[column]
        std = seen.std()
        rows.append(
            {
                "column": column,
                "missing_pct": missing[column].mean() * 100,
                "variance_ratio": full.var() / seen.var() if seen.var() > 0 else np.nan,
                "mean_shift_std": (full.mean() - seen.mean()) / std if std > 0 else np.nan,
                "ks_imputed_vs_observed": stats.ks_2samp(filled, seen).statistic,
            }
        )
    return pd.DataFrame(rows).set_index("column")


def correlation_distortion(observed: pd.DataFrame, imputed: pd.DataFrame) -> float:
    """Mean |r_imputed - r_observed| over column pairs (observed = pairwise complete)."""
    varying = observed.columns[observed.std() > 0]
    before = observed[varying].corr().to_numpy()
    after = imputed[varying].corr().to_numpy()
    upper = np.triu_indices_from(before, k=1)
    diff = np.abs(after[upper] - before[upper])
    return float(np.nanmean(diff))


# ---------------------------------------------------------------- pipeline


def load_prepared(path: Path) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    frame = pd.read_csv(path)
    ids = frame.pop(ID_COLUMN)
    target = frame.pop(TARGET_COLUMN).astype("int8")
    return frame, target, ids


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(
    input_path: Path,
    data_out: Path,
    model_out: Path,
    report_out: Path,
    strategies: list[Strategy] = STRATEGIES,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Fit every strategy, write datasets, fitted imputers and the comparison."""
    features, target, ids = load_prepared(input_path)
    for directory in (data_out, model_out, report_out):
        directory.mkdir(parents=True, exist_ok=True)

    summary, datasets, per_column = [], {}, {}
    for strategy in strategies:
        print(f"[{strategy.name}] fitting ...", flush=True)
        start = time.perf_counter()
        fitted = strategy.build().fit(features)
        out = fitted.transform(features)
        fit_seconds = time.perf_counter() - start
        if not isinstance(out, pd.DataFrame):
            out = pd.DataFrame(out, columns=features.columns, index=features.index)
        if out.isna().any().any():
            raise RuntimeError(f"{strategy.name}: NaN left after imputation")
        observed_cells = features.notna().to_numpy()
        if not np.allclose(
            out[features.columns].to_numpy()[observed_cells], features.to_numpy()[observed_cells]
        ):
            raise RuntimeError(f"{strategy.name}: observed values were changed")

        dataset = pd.concat([ids, out, target], axis=1)
        dataset_path = data_out / f"train_{strategy.name}.csv"
        dataset.to_csv(dataset_path, index=False)
        joblib.dump(fitted, model_out / f"{strategy.name}.joblib")
        datasets[strategy.name] = dataset

        imputed_features = out[features.columns]
        consistency = distribution_consistency(features, imputed_features)
        material = consistency[consistency["missing_pct"] >= MATERIAL_MISSING_PCT]
        per_column[strategy.name] = consistency
        consistency.to_csv(report_out / f"consistency_{strategy.name}.csv")
        print(f"[{strategy.name}] masked reconstruction ...", flush=True)
        recon = masked_reconstruction(
            strategy, features, cells_out=report_out / f"masked_cells_{strategy.name}.csv"
        )
        summary.append(
            {
                "strategy": strategy.name,
                "label": strategy.label,
                "columns": out.shape[1],
                "flag_columns": out.shape[1] - features.shape[1],
                "cells_imputed": int(features.isna().sum().sum()),
                "fit_seconds": round(fit_seconds, 1),
                **recon,
                "material_columns": len(material),
                "median_variance_ratio": float(material["variance_ratio"].median()),
                "min_variance_ratio": float(material["variance_ratio"].min()),
                "median_abs_mean_shift_std": float(material["mean_shift_std"].abs().median()),
                "median_ks": float(material["ks_imputed_vs_observed"].median()),
                "correlation_distortion": correlation_distortion(features, imputed_features),
                "dataset_sha256": _sha256(dataset_path),
            }
        )
        print(f"[{strategy.name}] done: {summary[-1]['nrmse']:.3f} NRMSE", flush=True)

    comparison = pd.DataFrame(summary).set_index("strategy")
    comparison.to_csv(report_out / "comparison.csv")
    meta = {
        "input": str(input_path),
        "input_sha256": _sha256(input_path),
        "rows": len(features),
        "features": features.shape[1],
        "missing_cells": int(features.isna().sum().sum()),
        "random_state": RANDOM_STATE,
        "strategies": {
            s.name: {"description": s.description, "params": s.params} for s in strategies
        },
    }
    (report_out / "imputation_run.json").write_text(json.dumps(meta, indent=2) + "\n")
    save_comparison_figure(comparison, report_out / "imputation_comparison.png")
    save_distribution_figure(features, datasets, report_out / "imputed_distributions.png")
    write_report(comparison, meta, datasets, report_out / "imputation_report.md")
    from .impute_viz import save_all_figures

    save_all_figures(features, datasets, comparison, report_out)
    return comparison, datasets


# ---------------------------------------------------------------- figures and report

BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#7b7a75", "#ecebe7"


def _style(axis: plt.Axes, grid: str = "x") -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis=grid, color=GRID)
    axis.set_axisbelow(True)


def save_comparison_figure(comparison: pd.DataFrame, output_path: Path) -> None:
    panels = [
        (
            "nrmse_material",
            "Masked reconstruction error\n(NRMSE, std units, lower is better;\n1.0 = guessing the mean)",
            1.0,
        ),
        (
            "median_variance_ratio",
            "Variance kept after imputation\n(median imputed / observed ratio;\n1.0 = unchanged)",
            1.0,
        ),
        (
            "median_ks",
            "Filled vs observed distribution\n(median KS statistic,\nlower is better)",
            None,
        ),
        (
            "correlation_distortion",
            "Correlation distortion\n(mean |Δr| over all pairs,\nlower is better)",
            None,
        ),
    ]
    labels = comparison["label"][::-1]
    figure, axes = plt.subplots(1, 4, figsize=(17, 4.6), sharey=True)
    for axis, (column, title, reference) in zip(axes, panels):
        values = comparison[column][::-1]
        best = (values - 1).abs().idxmin() if column == "median_variance_ratio" else values.idxmin()
        colors = [ORANGE if index == best else BLUE for index in values.index]
        bars = axis.barh(labels, values, color=colors, height=0.6)
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_width(),
                bar.get_y() + bar.get_height() / 2,
                f" {value:.4f}" if column == "correlation_distortion" else f" {value:.3f}",
                va="center",
                color=INK,
                fontsize=9,
            )
        if reference is not None:
            axis.axvline(reference, color=INK, ls="--", lw=1)
        axis.set_xlim(0, values.max() * 1.3)
        axis.set_title(title, fontsize=10, loc="left")
        _style(axis)
    figure.suptitle(
        "Imputation strategies on the training split: panels 1-3 use the 28 columns with ≥5% missing (best per panel in orange)",
        fontsize=12,
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def save_distribution_figure(
    features: pd.DataFrame, datasets: dict[str, pd.DataFrame], output_path: Path
) -> None:
    """Observed vs imputed values for the most-missing feature, one panel per strategy."""
    column = features.isna().mean().idxmax()
    observed = features[column].dropna()
    missing = features[column].isna()
    bins = np.histogram_bin_edges(observed, bins=30)
    figure, axes = plt.subplots(1, len(datasets), figsize=(4 * len(datasets), 3.4), sharey=True)
    for axis, (name, dataset) in zip(np.atleast_1d(axes), datasets.items()):
        imputed = dataset.loc[missing.to_numpy(), column]
        axis.hist(observed, bins=bins, color=BLUE, label=f"observed ({len(observed)})")
        axis.hist(imputed, bins=bins, color=ORANGE, alpha=0.85, label=f"imputed ({len(imputed)})")
        axis.set_title(STRATEGY_BY_NAME[name].label, fontsize=10, loc="left")
        axis.set_xlabel(column)
        _style(axis, "y")
    np.atleast_1d(axes)[0].set_ylabel("wafers")
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=8)
    figure.suptitle(
        f"{column} ({missing.mean():.1%} missing): where each strategy puts the filled values",
        fontsize=12,
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def write_report(
    comparison: pd.DataFrame,
    meta: dict,
    datasets: dict[str, pd.DataFrame],
    output_path: Path,
) -> None:
    def fmt(value: object) -> str:
        return f"{value:.3f}" if isinstance(value, float) else str(value)

    columns = [
        "label",
        "columns",
        "fit_seconds",
        "nrmse",
        "nmae",
        "nrmse_material",
        "median_variance_ratio",
        "min_variance_ratio",
        "median_ks",
        "correlation_distortion",
    ]
    table = ["| strategy | " + " | ".join(columns) + " |", "|---" * (len(columns) + 1) + "|"]
    for name, row in comparison[columns].iterrows():
        table.append(f"| {name} | " + " | ".join(fmt(row[c]) for c in columns) + " |")
    strategies = "\n".join(
        f"- **{name}**: {info['description']}" for name, info in meta["strategies"].items()
    )
    lines = [
        "# Imputation candidates: comparison on the training split",
        "",
        (
            f"Input: `{meta['input']}`: {meta['rows']:,} wafers × {meta['features']} features, "
            f"{meta['missing_cells']:,} missing cells. The test set is not used. All statistics "
            "below are label-free; the downstream (model) comparison is a separate step."
        ),
        "",
        "## Strategies",
        "",
        strategies,
        "",
        "## Results",
        "",
        "\n".join(table),
        "",
        "How to read the columns:",
        "",
        (
            "- `nrmse` / `nmae`: error on observed cells hidden with another wafer's real "
            "missingness pattern, in standard-deviation units. 1.0 is about as bad as guessing "
            "the mean. `nrmse_material` restricts this to columns with at least 5% missing."
        ),
        (
            "- `median_variance_ratio` / `min_variance_ratio`: variance of the imputed column "
            "divided by the variance of its observed values, over the columns with at least 5% "
            "missing. Below 1 means imputation shrank the spread."
        ),
        (
            "- `median_ks`: KS statistic between the filled values and the observed values of the "
            "same column (columns with at least 5% missing). Median imputation scores high because "
            "all fills sit on one value."
        ),
        "- `correlation_distortion`: mean absolute change of the pairwise correlations.",
        "",
        (
            "Figures: `imputation_comparison.png`, `imputed_distributions.png`. Per-column details: "
            "`consistency_<strategy>.csv`."
        ),
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build and compare imputed training datasets.")
    parser.add_argument("--input", type=Path, default=Path("data/processed/train_prepared.csv"))
    parser.add_argument("--data-out", type=Path, default=Path("data/processed/imputed"))
    parser.add_argument("--model-out", type=Path, default=Path("models/imputers"))
    parser.add_argument("--report-out", type=Path, default=Path("reports/imputation"))
    parser.add_argument("--strategies", nargs="*", default=[s.name for s in STRATEGIES])
    parser.add_argument("--no-db", action="store_true", help="skip loading into Postgres")
    args = parser.parse_args(argv)

    chosen = [STRATEGY_BY_NAME[name] for name in args.strategies]
    comparison, datasets = run(args.input, args.data_out, args.model_out, args.report_out, chosen)
    print(comparison.drop(columns=["dataset_sha256"]).to_string())
    if not args.no_db:
        from .production import load_production

        features, _, _ = load_prepared(args.input)
        counts = load_production(datasets, comparison, features, args.input)
        print(f"Loaded into schema secom_production: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

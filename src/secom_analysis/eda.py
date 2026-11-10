"""Exploratory analysis helpers used by the first two project weeks."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def descriptive_statistics(features: pd.DataFrame, target: pd.Series) -> dict[str, object]:
    """Return dataset shape, class balance, and per-feature missingness."""
    class_counts = target.value_counts().sort_index()
    return {
        "rows": int(features.shape[0]),
        "features": int(features.shape[1]),
        "missing_cells": int(features.isna().sum().sum()),
        "class_counts": {str(index): int(value) for index, value in class_counts.items()},
        "class_percentages": {
            str(index): float(value / len(target) * 100) for index, value in class_counts.items()
        },
        "missing_percentage": features.isna().mean().mul(100).to_dict(),
        "variance": features.var(axis=0, skipna=True).to_dict(),
    }


def constant_features(features: pd.DataFrame) -> list[str]:
    """Return columns whose observed variance is exactly zero."""
    return features.var(axis=0, skipna=True).eq(0).loc[lambda values: values].index.tolist()


def remove_features_above_missing_threshold(
    features: pd.DataFrame, threshold_percentage: float = 60.0
) -> tuple[pd.DataFrame, list[str]]:
    """Remove features exceeding the missing-value threshold."""
    missing_percentage = features.isna().mean().mul(100)
    removed = missing_percentage[missing_percentage > threshold_percentage].index.tolist()
    return features.drop(columns=removed), removed


def column_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-column descriptive table: dtype, nulls, unique values and basic statistics.

    ``unique`` counts distinct non-null values. ``unique_pct`` is relative to the non-null
    rows, so a column with 2 observed values out of 2 non-null rows is 100% unique.
    Numeric statistics are left empty for non-numeric columns.
    """
    rows = len(frame)
    non_null = frame.notna().sum()
    unique = frame.nunique(dropna=True)
    numeric = frame.select_dtypes("number")
    top_share = frame.apply(
        lambda column: (
            column.value_counts(normalize=True, dropna=True).max() * 100
            if column.notna().any()
            else float("nan")
        )
    )
    table = pd.DataFrame(
        {
            "dtype": frame.dtypes.astype(str),
            "non_null": non_null,
            "null_count": rows - non_null,
            "null_pct": (rows - non_null) / rows * 100 if rows else 0.0,
            "unique": unique,
            "unique_pct": (unique / non_null.where(non_null > 0)) * 100,
            "is_constant": unique <= 1,
            "is_binary": unique == 2,
            "top_value_pct": top_share,
        }
    )
    stats = pd.DataFrame(
        {
            "mean": numeric.mean(),
            "std": numeric.std(),
            "min": numeric.min(),
            "q25": numeric.quantile(0.25),
            "median": numeric.median(),
            "q75": numeric.quantile(0.75),
            "max": numeric.max(),
            "zeros_pct": numeric.eq(0).sum() / rows * 100 if rows else 0.0,
        }
    )
    table = table.join(stats)
    table.index.name = "column"
    return table


def dataset_overview_table(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per dataset: shape, memory, dtype mix, nulls, duplicates, constant columns."""
    records = []
    for name, frame in datasets.items():
        cells = frame.size
        nulls = int(frame.isna().sum().sum())
        unique = frame.nunique(dropna=True)
        dtype_counts = frame.dtypes.astype(str).value_counts()
        records.append(
            {
                "dataset": name,
                "rows": frame.shape[0],
                "columns": frame.shape[1],
                "cells": cells,
                "memory_mb": round(frame.memory_usage(deep=True).sum() / 1e6, 2),
                "dtypes": ", ".join(f"{dtype}: {count}" for dtype, count in dtype_counts.items()),
                "null_cells": nulls,
                "null_pct": round(nulls / cells * 100, 2) if cells else 0.0,
                "columns_with_nulls": int(frame.isna().any().sum()),
                "rows_with_nulls": int(frame.isna().any(axis=1).sum()),
                "complete_rows": int(frame.notna().all(axis=1).sum()),
                "duplicate_rows": int(frame.duplicated().sum()),
                "constant_columns": int((unique <= 1).sum()),
            }
        )
    return pd.DataFrame(records).set_index("dataset")


def _bucket_counts(values: pd.Series, edges: list[float], labels: list[str]) -> pd.Series:
    bucketed = pd.cut(values, bins=edges, labels=labels, include_lowest=True, right=True)
    return bucketed.value_counts().reindex(labels, fill_value=0)


def _md_table(frame: pd.DataFrame, floatfmt: str = ".2f") -> str:
    """Markdown table without the optional ``tabulate`` dependency."""
    header = "| " + " | ".join([str(frame.index.name or "")] + [str(c) for c in frame.columns])
    lines = [header + " |", "|" + "---|" * (frame.shape[1] + 1)]
    for index, row in frame.astype(object).iterrows():
        cells = [str(index)]
        for value in row:
            if isinstance(value, float) and not pd.isna(value):
                cells.append(format(value, floatfmt))
            elif pd.isna(value):
                cells.append("")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_descriptive_report(
    output_path: Path,
    overview: pd.DataFrame,
    feature_summary: pd.DataFrame,
    labels: pd.DataFrame,
    target: pd.Series,
) -> None:
    """Write a Markdown report: shape, dtypes, nulls, unique values, label file, statistics."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fs = feature_summary
    n_features = len(fs)
    null_buckets = _bucket_counts(
        fs["null_pct"],
        [-0.001, 0, 5, 20, 50, 60, 100],
        ["0%", "(0, 5]%", "(5, 20]%", "(20, 50]%", "(50, 60]%", "(60, 100]%"],
    )
    unique_buckets = _bucket_counts(
        fs["unique"].astype(float),
        [-0.5, 1, 2, 10, 50, 500, float("inf")],
        ["1 (constant)", "2 (binary)", "3-10", "11-50", "51-500", "> 500"],
    )
    numeric = fs.dropna(subset=["mean"])
    top_null = fs.sort_values("null_pct", ascending=False).head(15)[
        ["null_count", "null_pct", "unique", "mean", "std"]
    ]
    low_unique = fs[(fs["unique"] > 1) & (fs["unique"] <= 10)].sort_values("unique")[
        ["unique", "top_value_pct", "null_pct", "min", "max"]
    ]
    dominated = fs[~fs["is_constant"] & (fs["top_value_pct"] >= 99)]
    varying = numeric[~numeric["is_constant"]]
    label_summary = column_summary(labels)[["dtype", "non_null", "null_count", "unique"]]
    class_counts = target.value_counts().sort_index()
    timestamps = labels["timestamp"].dropna()

    sections = [
        "# SECOM descriptive statistics report",
        "",
        (
            "Generated by `python -m secom_analysis.cli`. Full data (before the train/test split) "
            "unless stated. Per-column details for all 590 features are in "
            "`column_summary.csv`."
        ),
        "",
        "## 1. Shape and size",
        "",
        _md_table(overview),
        "",
        "## 2. Data types",
        "",
        (
            f"- **Features:** all {n_features} columns are "
            f"`{', '.join(sorted(fs['dtype'].unique()))}` (continuous sensor readings; "
            "the raw file is space-separated text, `NaN` marks a missing value)."
        ),
        "- **Labels file:** a pass/fail code and a test timestamp per wafer:",
        "",
        _md_table(label_summary),
        "",
        "## 3. Target",
        "",
        "| code | meaning | wafers | share |",
        "|---|---|---|---|",
        *[
            f"| {code} | {'defective (raw 1)' if code == 1 else 'good (raw −1)'} | {count:,} "
            f"| {count / len(target):.2%} |"
            for code, count in class_counts.items()
        ],
        "",
        (
            f"Timestamps run from {timestamps.min():%Y-%m-%d} to {timestamps.max():%Y-%m-%d} "
            f"({timestamps.nunique():,} distinct values for {len(labels):,} wafers)."
        ),
        "",
        "## 4. Null values per feature",
        "",
        (
            f"- Features with no nulls: **{int((fs['null_count'] == 0).sum())}**; "
            f"with at least one: **{int((fs['null_count'] > 0).sum())}**."
        ),
        (
            f"- Median null share per feature: {fs['null_pct'].median():.2f}%; "
            f"max: {fs['null_pct'].max():.2f}%."
        ),
        "",
        _md_table(null_buckets.rename("features").to_frame().rename_axis("null share")),
        "",
        "Top 15 features by null share:",
        "",
        _md_table(top_null),
        "",
        "## 5. Unique values per feature",
        "",
        (
            "Counted over non-null values. Low counts flag constant, binary or "
            "discrete-looking sensors."
        ),
        "",
        _md_table(unique_buckets.rename("features").to_frame().rename_axis("distinct values")),
        "",
        f"Non-constant features with at most 10 distinct values ({len(low_unique)}):",
        "",
        _md_table(low_unique, floatfmt=".4g") if len(low_unique) else "_none_",
        "",
        (
            f"**Near-constant:** {len(dominated)} non-constant features have one value on at "
            "least 99% of their non-null rows (`top_value_pct` in the CSV). Such a feature "
            "varies on only a handful of wafers and is almost constant in practice; on a "
            "different split it can become exactly constant."
        ),
        "",
        "## 6. Value ranges",
        "",
        (
            f"- Smallest feature minimum: {numeric['min'].min():.4g}; largest maximum: "
            f"{numeric['max'].max():.4g}."
        ),
        (
            f"- Standard deviation of the non-constant features ranges from "
            f"{varying['std'].min():.3g} to {varying['std'].max():.3g}: features need scaling "
            "before any distance-based method."
        ),
        (
            f"- Features with negative values: {int((numeric['min'] < 0).sum())}; non-constant "
            f"features that are more than 50% zeros: {int((varying['zeros_pct'] > 50).sum())}."
        ),
        "",
    ]
    output_path.write_text("\n".join(sections), encoding="utf-8")


BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#7b7a75"


def _style(axis: plt.Axes) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#ecebe7")
    axis.set_axisbelow(True)


def _missingness_hist(axis: plt.Axes, missing_percentage: pd.Series, threshold: float) -> None:
    bins = np.arange(0, 105, 5)
    counts, edges = np.histogram(missing_percentage, bins=bins)
    colors = [ORANGE if left >= threshold else BLUE for left in edges[:-1]]
    axis.bar(edges[:-1], counts, width=4.6, align="edge", color=colors)
    axis.set_yscale("log")
    axis.axvline(threshold, color=INK, ls="--", lw=1)
    axis.text(threshold + 1, axis.get_ylim()[1] * 0.5, f"threshold {threshold:g}%", color=INK)
    removed = int(missing_percentage.gt(threshold).sum())
    axis.set(
        title=f"Missing values per feature ({removed} features > {threshold:g}% in orange)",
        xlabel="missing values (%)",
        ylabel="number of features (log scale)",
    )
    _style(axis)


def save_missingness_plot(
    features: pd.DataFrame, output_path: Path, threshold: float = 60.0
) -> None:
    """Save a histogram of feature missing-value percentages (log y, threshold marked)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(9, 5))
    _missingness_hist(axis, features.isna().mean().mul(100), threshold)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def save_eda_overview(
    features: pd.DataFrame,
    target: pd.Series,
    train_features: pd.DataFrame,
    train_target: pd.Series,
    test_target: pd.Series,
    constants: list[str],
    high_missing: list[str],
    threshold: float,
    output_path: Path,
) -> None:
    """One-page EDA overview: class balance, missingness, variance, reduction funnel.

    Missingness, variance and the funnel use the training split, because those are the
    statistics the reduction decisions are fitted on.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 2, figsize=(13, 8.5))

    # (a) class balance and stratification
    axis = axes[0, 0]
    names = ["full data", "train (80%)", "test (20%)"]
    targets = [target, train_target, test_target]
    rates = [t.mean() * 100 for t in targets]
    bars = axis.bar(names, rates, color=ORANGE, width=0.55)
    for bar, t in zip(bars, targets):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.15,
            f"{t.mean():.2%}\n{int(t.sum())} of {len(t):,}",
            ha="center",
            va="bottom",
            color=INK,
        )
    axis.set_ylim(0, max(rates) * 1.45)
    axis.set(
        title="Defective wafers: imbalanced (about 1 in 15) and stratified",
        ylabel="defective wafers (%)",
    )
    _style(axis)

    # (b) missingness per feature (train)
    _missingness_hist(axes[0, 1], train_features.isna().mean().mul(100), threshold)

    # (c) variance per feature (train)
    axis = axes[1, 0]
    variance = train_features.var(axis=0, skipna=True)
    positive = variance[variance > 0]
    counts, _, _ = axis.hist(np.log10(positive), bins=40, color=BLUE)
    axis.set_ylim(0, counts.max() * 1.25)
    axis.text(
        0.02,
        0.95,
        f"{len(constants)} features have variance exactly 0 (not shown; removed)",
        transform=axis.transAxes,
        va="top",
        color=INK,
    )
    axis.set(
        title=f"Variance per feature: {len(positive)} non-constant features",
        xlabel="log10(variance)",
        ylabel="number of features",
    )
    _style(axis)

    # (d) reduction funnel
    axis = axes[1, 1]
    after_constant = train_features.shape[1] - len(constants)
    after_missing = after_constant - len(high_missing)
    steps = ["raw features", "variance = 0 removed", f"> {threshold:g}% missing removed"]
    counts = [train_features.shape[1], after_constant, after_missing]
    bars = axis.barh(steps[::-1], counts[::-1], color=BLUE, height=0.55)
    for bar, count, previous in zip(bars, counts[::-1], [counts[1], counts[0], None]):
        removed = "" if previous is None else f"  (-{previous - count})"
        axis.text(
            count + 5,
            bar.get_y() + bar.get_height() / 2,
            f"{count}{removed}",
            va="center",
            color=INK,
        )
    axis.set_xlim(0, train_features.shape[1] * 1.2)
    axis.set(title="Rough feature reduction (fitted on training split)", xlabel="features")
    axis.spines[["top", "right"]].set_visible(False)

    figure.suptitle(
        f"SECOM EDA overview: {features.shape[0]:,} wafers x {features.shape[1]} features, "
        f"{int(features.isna().sum().sum()):,} missing cells "
        f"({features.isna().mean().mean():.2%})",
        fontsize=13,
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)

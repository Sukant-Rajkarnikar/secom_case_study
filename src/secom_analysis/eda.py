"""Exploratory analysis helpers used by the first two project weeks."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def descriptive_statistics(features: pd.DataFrame, target: pd.Series) -> dict[str, object]:
    """Return dataset shape, class balance, and per-feature missingness."""
    class_counts = target.value_counts().sort_index()
    return {
        "rows": int(features.shape[0]),
        "features": int(features.shape[1]),
        "missing_cells": int(features.isna().sum().sum()),
        "class_counts": {str(index): int(value) for index, value in class_counts.items()},
        "class_percentages": {
            str(index): float(value / len(target) * 100)
            for index, value in class_counts.items()
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


def save_missingness_plot(features: pd.DataFrame, output_path: Path) -> None:
    """Save a histogram of feature missing-value percentages."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    missing_percentage = features.isna().mean().mul(100)
    figure, axis = plt.subplots(figsize=(9, 5))
    sns.histplot(missing_percentage, bins=20, ax=axis)
    axis.set(xlabel="Missing values (%)", ylabel="Number of features")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)

"""Command-line entry point for the SECOM scaffold."""

import argparse
import json
from pathlib import Path

from .data import load_labels, load_secom, stratified_split
from .eda import (
    column_summary,
    constant_features,
    dataset_overview_table,
    descriptive_statistics,
    remove_features_above_missing_threshold,
    save_eda_overview,
    save_missingness_plot,
    write_descriptive_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument("--missing-threshold", type=float, default=60.0)
    args = parser.parse_args()

    features, target = load_secom(args.data_dir)
    train_features, test_features, train_target, test_target = stratified_split(features, target)
    constants = constant_features(train_features)
    reduced_features, high_missing = remove_features_above_missing_threshold(
        train_features.drop(columns=constants),
        threshold_percentage=args.missing_threshold,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = descriptive_statistics(features, target)
    summary.update(
        {
            "train_rows": len(train_features),
            "test_rows": len(test_features),
            "train_defect_percentage": float(train_target.mean() * 100),
            "test_defect_percentage": float(test_target.mean() * 100),
            "constant_features": constants,
            "high_missing_features": high_missing,
            "features_after_reduction": reduced_features.shape[1],
        }
    )
    (args.output_dir / "descriptive_statistics.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    labels = load_labels(args.data_dir)
    overview = dataset_overview_table(
        {
            "features (full)": features,
            "labels file": labels,
            "train features": train_features,
            "test features": test_features,
            "train reduced": reduced_features,
        }
    )
    feature_summary = column_summary(features)
    overview.to_csv(args.output_dir / "dataset_overview.csv")
    feature_summary.to_csv(args.output_dir / "column_summary.csv")
    write_descriptive_report(
        args.output_dir / "descriptive_report.md", overview, feature_summary, labels, target
    )
    save_missingness_plot(
        train_features,
        args.output_dir / "missingness_histogram.png",
        threshold=args.missing_threshold,
    )
    save_eda_overview(
        features,
        target,
        train_features,
        train_target,
        test_target,
        constants,
        high_missing,
        args.missing_threshold,
        args.output_dir / "eda_overview.png",
    )


if __name__ == "__main__":
    main()

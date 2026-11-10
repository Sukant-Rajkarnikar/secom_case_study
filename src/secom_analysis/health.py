"""Descriptive profiling and data-health checks for every SECOM dataset.

Datasets covered:

* ``raw_files``  - the UCI text files themselves (line counts, field counts, tokens).
* ``labels``     - pass/fail labels and production timestamps.
* ``features``   - the full 1567 x 590 measurement matrix.
* ``train`` / ``test`` - the stratified split used by the modelling workflow.
* ``split``      - cross-dataset checks: leakage, prevalence and distribution drift.

Each check yields a ``CheckResult`` with status PASS / WARN / FAIL. FAIL means the data
cannot be trusted as loaded (structural problem); WARN flags a quality issue that later
preprocessing must handle (missingness, constants, imbalance, drift ...).

Run with::

    python -m secom_analysis.health --data-dir data/raw --output-dir reports/health
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .data import (
    LABELS_FILE,
    MEASUREMENTS_FILE,
    load_labels,
    load_secom,
    resolve_data_dir,
    stratified_split,
)
from .report import write_descriptive_report

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
EXPECTED_ROWS = 1567
EXPECTED_FEATURES = 590


@dataclass(frozen=True)
class Thresholds:
    """Tunable limits for the health checks."""

    high_missing_pct: float = 50.0  # column missingness that warrants a warning
    row_missing_pct: float = 50.0  # row missingness that warrants a warning
    near_constant_share: float = 0.95  # share of the dominant value among non-missing
    outlier_sigma: float = 3.0  # 3-sigma rule: |x - mean| > k * std
    extreme_skew: float = 10.0  # |skew| above this is "extreme"
    min_minority_pct: float = 10.0  # minority class share below this is "imbalanced"
    prevalence_tolerance_pct: float = 1.0  # allowed train/test defect-rate gap
    drift_alpha: float = 0.01  # KS-test significance for train/test drift
    drift_max_share: float = 0.05  # share of drifted features tolerated before WARN


DEFAULT_THRESHOLDS = Thresholds()


@dataclass
class CheckResult:
    dataset: str
    check: str
    status: str
    value: object
    detail: str
    items: list[str] = field(default_factory=list)


def _result(dataset, check, ok, value, detail, items=None, *, severity=WARN) -> CheckResult:
    return CheckResult(dataset, check, PASS if ok else severity, value, detail, list(items or []))


# --------------------------------------------------------------------------------------
# Descriptive profiles
# --------------------------------------------------------------------------------------


def profile_features(
    features: pd.DataFrame,
    target: pd.Series | None = None,
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
) -> pd.DataFrame:
    """Return one row of descriptive statistics per feature.

    When ``target`` is given, per-class mean and missingness are added so pass/fail
    differences are visible at a glance.
    """
    numeric = features.apply(pd.to_numeric, errors="coerce")
    finite = numeric.replace([np.inf, -np.inf], np.nan)
    quantiles = finite.quantile([0.01, 0.25, 0.5, 0.75, 0.99]).T
    quantiles.columns = ["p01", "p25", "median", "p75", "p99"]
    iqr = quantiles["p75"] - quantiles["p25"]
    count = finite.count()

    def dominant_share(column: pd.Series) -> float:
        counts = column.value_counts(dropna=True)
        return float(counts.iloc[0] / counts.sum()) if len(counts) else np.nan

    mean = finite.mean()
    std = finite.std()
    lower = mean - thresholds.outlier_sigma * std
    upper = mean + thresholds.outlier_sigma * std
    outliers = finite.lt(lower, axis=1).sum() + finite.gt(upper, axis=1).sum()
    profile = pd.DataFrame(
        {
            "dtype": features.dtypes.astype(str),
            "count": count,
            "missing": features.isna().sum(),
            "missing_pct": features.isna().mean() * 100,
            "non_numeric": (numeric.isna() & features.notna()).sum(),
            "infinite": np.isinf(numeric).sum(),
            "n_unique": finite.nunique(dropna=True),
            "dominant_share": finite.apply(dominant_share),
            "mean": mean,
            "std": std,
            "cv": std / mean.abs().replace(0, np.nan),
            "min": finite.min(),
            **{column: quantiles[column] for column in quantiles.columns},
            "max": finite.max(),
            "iqr": iqr,
            "skew": finite.skew(),
            "kurtosis": finite.kurt(),
            "zero_pct": finite.eq(0).sum() / count.replace(0, np.nan) * 100,
            "negative_pct": finite.lt(0).sum() / count.replace(0, np.nan) * 100,
            "sigma3_outliers": outliers,
            "sigma3_outlier_pct": outliers / count.replace(0, np.nan) * 100,
        }
    )
    profile["is_constant"] = profile["n_unique"].le(1)
    profile["is_near_constant"] = ~profile["is_constant"] & profile["dominant_share"].ge(
        thresholds.near_constant_share
    )

    if target is not None:
        target = target.loc[features.index]
        for label, name in ((0, "pass"), (1, "fail")):
            subset = finite.loc[target.eq(label)]
            profile[f"mean_{name}"] = subset.mean()
            profile[f"missing_pct_{name}"] = subset.isna().mean() * 100
        pooled = profile["std"].replace(0, np.nan)
        profile["mean_diff_fail_vs_pass_in_std"] = (
            profile["mean_fail"] - profile["mean_pass"]
        ) / pooled

    profile.index.name = "feature"
    return profile


def dataset_overview(features: pd.DataFrame, target: pd.Series | None = None) -> dict[str, object]:
    """Return dataset-level descriptive numbers."""
    total_cells = features.size
    missing_cells = int(features.isna().sum().sum())
    row_missing = features.isna().mean(axis=1) * 100
    overview: dict[str, object] = {
        "rows": int(features.shape[0]),
        "columns": int(features.shape[1]),
        "cells": int(total_cells),
        "missing_cells": missing_cells,
        "missing_cell_pct": round(missing_cells / total_cells * 100, 3) if total_cells else 0.0,
        "complete_rows": int(features.notna().all(axis=1).sum()),
        "complete_columns": int(features.notna().all(axis=0).sum()),
        "row_missing_pct": {
            "min": float(row_missing.min()),
            "median": float(row_missing.median()),
            "max": float(row_missing.max()),
        },
        "memory_mb": round(features.memory_usage(deep=True).sum() / 1e6, 3),
    }
    if target is not None:
        counts = target.value_counts().sort_index()
        overview["class_counts"] = {str(k): int(v) for k, v in counts.items()}
        overview["defect_pct"] = round(float(target.mean() * 100), 3)
    return overview


# --------------------------------------------------------------------------------------
# Health checks
# --------------------------------------------------------------------------------------


def check_raw_files(data_dir: Path) -> list[CheckResult]:
    """Validate the UCI text files before pandas touches them."""
    name = "raw_files"
    data_dir = resolve_data_dir(data_dir)
    results: list[CheckResult] = []
    paths = {"measurements": data_dir / MEASUREMENTS_FILE, "labels": data_dir / LABELS_FILE}
    for label, path in paths.items():
        results.append(
            _result(
                name,
                f"{label}_file_exists",
                path.exists(),
                str(path),
                "file present",
                severity=FAIL,
            )
        )
    if not all(path.exists() for path in paths.values()):
        return results

    lines = paths["measurements"].read_text(encoding="utf-8").splitlines()
    field_counts = pd.Series([len(line.split()) for line in lines])
    blank = int(field_counts.eq(0).sum())
    uneven = field_counts[field_counts.ne(EXPECTED_FEATURES)]
    results += [
        _result(
            name,
            "measurement_row_count",
            len(lines) == EXPECTED_ROWS,
            len(lines),
            f"expected {EXPECTED_ROWS} lines",
        ),
        _result(
            name,
            "measurement_field_count",
            uneven.empty,
            sorted(field_counts.unique().tolist()),
            f"every line should have {EXPECTED_FEATURES} fields",
            [f"line {i + 1}: {n} fields" for i, n in uneven.head(20).items()],
            severity=FAIL,
        ),
        _result(name, "measurement_blank_lines", blank == 0, blank, "blank lines in file"),
    ]

    tokens = pd.Series(" ".join(lines).split())
    is_number = pd.to_numeric(tokens, errors="coerce").notna()
    bad_tokens = tokens[~is_number & tokens.ne("NaN")]
    results.append(
        _result(
            name,
            "measurement_tokens_parseable",
            bad_tokens.empty,
            len(bad_tokens),
            "tokens that are neither numbers nor 'NaN'",
            bad_tokens.value_counts().head(20).index.tolist(),
            severity=FAIL,
        )
    )

    label_lines = paths["labels"].read_text(encoding="utf-8").splitlines()
    results.append(
        _result(
            name,
            "label_row_count_matches",
            len(label_lines) == len(lines),
            {"labels": len(label_lines), "measurements": len(lines)},
            "label and measurement files must have the same number of lines",
            severity=FAIL,
        )
    )
    return results


def check_labels(
    labels: pd.DataFrame, thresholds: Thresholds = DEFAULT_THRESHOLDS
) -> list[CheckResult]:
    """Validate the label domain, class balance and timestamps."""
    name = "labels"
    invalid = labels.loc[~labels["label"].isin([-1, 1]), "label"]
    counts = labels["label"].value_counts()
    minority_pct = float(counts.min() / counts.sum() * 100) if len(counts) else 0.0
    ts = labels["timestamp"]
    unparsed = labels.loc[ts.isna(), "timestamp_raw"]
    duplicated_ts = ts[ts.duplicated(keep=False) & ts.notna()]
    backwards = int((ts.dropna().diff().dt.total_seconds() < 0).sum())
    gaps = ts.sort_values().diff().dropna()
    return [
        _result(
            name,
            "label_domain",
            invalid.empty,
            len(invalid),
            "labels must be -1 (pass) or 1 (fail)",
            invalid.astype(str).unique().tolist(),
            severity=FAIL,
        ),
        _result(
            name,
            "label_missing",
            labels["label"].notna().all(),
            int(labels["label"].isna().sum()),
            "missing labels",
            severity=FAIL,
        ),
        _result(
            name,
            "class_balance",
            minority_pct >= thresholds.min_minority_pct,
            {str(k): int(v) for k, v in counts.sort_index().items()},
            f"minority class is {minority_pct:.2f}% (warn below {thresholds.min_minority_pct}%)",
        ),
        _result(
            name,
            "timestamp_parseable",
            unparsed.empty,
            len(unparsed),
            "timestamps not matching dd/mm/YYYY HH:MM:SS",
            unparsed.head(20).astype(str).tolist(),
            severity=FAIL,
        ),
        _result(
            name,
            "timestamp_duplicates",
            duplicated_ts.empty,
            len(duplicated_ts),
            "rows sharing an identical timestamp",
            sorted({str(value) for value in duplicated_ts.head(20)}),
        ),
        _result(
            name,
            "timestamp_monotonic",
            backwards == 0,
            backwards,
            "times a row's timestamp is earlier than the previous row's",
        ),
        _result(
            name,
            "timestamp_range",
            True,
            {
                "start": str(ts.min()),
                "end": str(ts.max()),
                "span_days": round((ts.max() - ts.min()).total_seconds() / 86400, 2),
                "median_gap_minutes": round(gaps.median().total_seconds() / 60, 2)
                if len(gaps)
                else None,
                "max_gap_hours": round(gaps.max().total_seconds() / 3600, 2) if len(gaps) else None,
            },
            "informational: production window covered",
        ),
    ]


def check_features(
    features: pd.DataFrame,
    profile: pd.DataFrame,
    dataset: str,
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
) -> list[CheckResult]:
    """Run structural and quality checks on a feature matrix."""
    row_missing = features.isna().mean(axis=1) * 100
    duplicate_rows = features[features.duplicated(keep=False)]
    varying = profile.index[~profile["is_constant"]]
    duplicate_cols = features[varying].T.duplicated(keep="first")
    duplicate_col_names = duplicate_cols[duplicate_cols].index.tolist()
    all_missing = profile.index[profile["missing"].eq(len(features))].tolist()
    high_missing = profile.index[profile["missing_pct"].gt(thresholds.high_missing_pct)].tolist()
    any_missing = profile.index[profile["missing"].gt(0)].tolist()
    constant = profile.index[profile["is_constant"]].tolist()
    near_constant = profile.index[profile["is_near_constant"]].tolist()
    extreme_skew = profile.index[profile["skew"].abs().gt(thresholds.extreme_skew)].tolist()
    non_numeric = profile.index[profile["non_numeric"].gt(0)].tolist()
    infinite = profile.index[profile["infinite"].gt(0)].tolist()
    heavy_rows = row_missing[row_missing.gt(thresholds.row_missing_pct)].index.tolist()
    empty_rows = row_missing[row_missing.eq(100)].index.tolist()
    n_cols = max(len(profile), 1)

    def pct(items: list) -> str:
        return f"{len(items)} of {len(profile)} features ({len(items) / n_cols * 100:.1f}%)"

    return [
        _result(
            dataset,
            "not_empty",
            features.shape[0] > 0 and features.shape[1] > 0,
            list(features.shape),
            "dataset has rows and columns",
            severity=FAIL,
        ),
        _result(
            dataset,
            "numeric_values",
            not non_numeric,
            len(non_numeric),
            "features containing non-numeric values",
            non_numeric,
            severity=FAIL,
        ),
        _result(
            dataset,
            "infinite_values",
            not infinite,
            len(infinite),
            "features containing +/-inf",
            infinite,
            severity=FAIL,
        ),
        _result(
            dataset,
            "duplicate_rows",
            duplicate_rows.empty,
            len(duplicate_rows),
            "rows identical to another row",
            [str(i) for i in duplicate_rows.index[:20]],
        ),
        _result(
            dataset,
            "duplicate_columns",
            not duplicate_col_names,
            len(duplicate_col_names),
            f"{pct(duplicate_col_names)} are exact copies of an earlier non-constant feature",
            duplicate_col_names,
        ),
        _result(
            dataset,
            "all_missing_columns",
            not all_missing,
            len(all_missing),
            pct(all_missing) + " are entirely missing",
            all_missing,
        ),
        _result(
            dataset,
            "high_missing_columns",
            not high_missing,
            len(high_missing),
            pct(high_missing) + f" are >{thresholds.high_missing_pct:g}% missing",
            high_missing,
        ),
        _result(
            dataset,
            "columns_with_missing",
            not any_missing,
            len(any_missing),
            pct(any_missing) + " have at least one missing value",
            any_missing,
        ),
        _result(
            dataset,
            "empty_rows",
            not empty_rows,
            len(empty_rows),
            "rows with every feature missing",
            [str(i) for i in empty_rows[:20]],
        ),
        _result(
            dataset,
            "high_missing_rows",
            not heavy_rows,
            len(heavy_rows),
            f"rows >{thresholds.row_missing_pct:g}% missing",
            [str(i) for i in heavy_rows[:20]],
        ),
        _result(
            dataset,
            "constant_columns",
            not constant,
            len(constant),
            pct(constant) + " have a single observed value (zero variance)",
            constant,
        ),
        _result(
            dataset,
            "near_constant_columns",
            not near_constant,
            len(near_constant),
            pct(near_constant)
            + f" have one value in >={thresholds.near_constant_share:.0%} of rows",
            near_constant,
        ),
        _result(
            dataset,
            "extreme_skew_columns",
            not extreme_skew,
            len(extreme_skew),
            pct(extreme_skew) + f" have |skew| > {thresholds.extreme_skew:g}",
            extreme_skew,
        ),
        _result(
            dataset,
            "sigma3_outliers",
            True,
            {
                "cells": int(profile["sigma3_outliers"].sum()),
                "median_pct_per_feature": round(float(profile["sigma3_outlier_pct"].median()), 3),
                "features_over_5pct": int(profile["sigma3_outlier_pct"].gt(5).sum()),
            },
            f"informational: values beyond mean +/- {thresholds.outlier_sigma:g} std (3-sigma rule)",
        ),
    ]


def check_split(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    train_target: pd.Series,
    test_target: pd.Series,
    total_rows: int,
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
) -> tuple[list[CheckResult], pd.DataFrame]:
    """Check the train/test split for leakage, prevalence and distribution drift."""
    name = "split"
    overlap = train_features.index.intersection(test_features.index)
    covered = len(train_features) + len(test_features)
    prevalence_gap = abs(train_target.mean() - test_target.mean()) * 100

    drift_rows = []
    for column in train_features.columns:
        train_values = train_features[column].dropna()
        test_values = test_features[column].dropna()
        if train_values.nunique() <= 1 and test_values.nunique() <= 1:
            statistic, p_value = 0.0, 1.0
        elif len(train_values) < 2 or len(test_values) < 2:
            statistic, p_value = np.nan, np.nan
        else:
            statistic, p_value = stats.ks_2samp(train_values, test_values)
        drift_rows.append(
            {
                "feature": column,
                "ks_statistic": statistic,
                "ks_p_value": p_value,
                "missing_pct_train": train_features[column].isna().mean() * 100,
                "missing_pct_test": test_features[column].isna().mean() * 100,
            }
        )
    drift = pd.DataFrame(drift_rows).set_index("feature")
    drift["missing_pct_gap"] = (drift["missing_pct_train"] - drift["missing_pct_test"]).abs()
    drifted = drift.index[drift["ks_p_value"].lt(thresholds.drift_alpha)].tolist()
    drift_share = len(drifted) / max(len(drift), 1)
    missing_gap = drift.index[drift["missing_pct_gap"].gt(10)].tolist()

    results = [
        _result(
            name,
            "no_row_overlap",
            overlap.empty,
            len(overlap),
            "rows present in both train and test (leakage)",
            [str(i) for i in overlap[:20]],
            severity=FAIL,
        ),
        _result(
            name,
            "rows_covered",
            covered == total_rows,
            {"train": len(train_features), "test": len(test_features), "total": total_rows},
            "train + test must equal the full dataset",
            severity=FAIL,
        ),
        _result(
            name,
            "same_columns",
            list(train_features.columns) == list(test_features.columns),
            train_features.shape[1],
            "train and test share identical columns",
            severity=FAIL,
        ),
        _result(
            name,
            "stratified_prevalence",
            prevalence_gap <= thresholds.prevalence_tolerance_pct,
            {
                "train_defect_pct": round(float(train_target.mean() * 100), 3),
                "test_defect_pct": round(float(test_target.mean() * 100), 3),
            },
            f"defect-rate gap {prevalence_gap:.2f} pp "
            f"(tolerance {thresholds.prevalence_tolerance_pct} pp)",
        ),
        _result(
            name,
            "distribution_drift",
            drift_share <= thresholds.drift_max_share,
            len(drifted),
            f"{len(drifted)} features ({drift_share:.1%}) differ train vs test "
            f"(KS p < {thresholds.drift_alpha}); ~{thresholds.drift_alpha:.0%} expected "
            "by chance",
            drifted,
        ),
        _result(
            name,
            "missingness_drift",
            not missing_gap,
            len(missing_gap),
            "features whose missing % differs by >10 pp between train and test",
            missing_gap,
        ),
    ]
    return results, drift


# --------------------------------------------------------------------------------------
# Orchestration and reporting
# --------------------------------------------------------------------------------------


def run_health_checks(
    data_dir: Path,
    output_dir: Path,
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
    test_size: float = 0.2,
    random_state: int = 42,
) -> list[CheckResult]:
    """Profile and check every dataset, writing CSV / JSON / Markdown / HTML to ``output_dir``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    results = check_raw_files(data_dir)
    if any(r.status == FAIL and r.check.endswith("_file_exists") for r in results):
        _write_reports(output_dir, results, {}, {})
        return results

    labels = load_labels(data_dir)
    features, target = load_secom(data_dir)
    train_x, test_x, train_y, test_y = stratified_split(
        features, target, test_size=test_size, random_state=random_state
    )
    results += check_labels(labels, thresholds)

    overviews: dict[str, dict] = {"labels": _label_overview(labels)}
    profiles: dict[str, pd.DataFrame] = {}
    for dataset, (x, y) in {
        "features": (features, target),
        "train": (train_x, train_y),
        "test": (test_x, test_y),
    }.items():
        profile = profile_features(x, y, thresholds)
        profiles[dataset] = profile
        overviews[dataset] = dataset_overview(x, y)
        results += check_features(x, profile, dataset, thresholds)

    split_results, drift = check_split(train_x, test_x, train_y, test_y, len(features), thresholds)
    results += split_results
    profiles["split_drift"] = drift

    _write_reports(output_dir, results, overviews, profiles, labels)
    return results


def _label_overview(labels: pd.DataFrame) -> dict[str, object]:
    ts = labels["timestamp"]
    daily = labels.assign(day=ts.dt.date, fail=labels["label"].eq(1)).groupby("day")["fail"]
    return {
        "rows": len(labels),
        "class_counts": {str(k): int(v) for k, v in labels["label"].value_counts().items()},
        "defect_pct": round(float(labels["label"].eq(1).mean() * 100), 3),
        "first_timestamp": str(ts.min()),
        "last_timestamp": str(ts.max()),
        "production_days": int(daily.ngroups),
        "units_per_day_median": float(daily.size().median()),
        "daily_defect_pct_max": round(float(daily.mean().max() * 100), 2),
    }


def _write_reports(
    output_dir: Path,
    results: list[CheckResult],
    overviews: dict[str, dict],
    profiles: dict[str, pd.DataFrame],
    labels: pd.DataFrame | None = None,
) -> None:
    for dataset, profile in profiles.items():
        profile.to_csv(output_dir / f"{dataset}_profile.csv", float_format="%.6g")

    checks = pd.DataFrame([asdict(r) for r in results])
    if not checks.empty:
        checks["items"] = checks["items"].map(lambda items: ";".join(map(str, items)))
        checks["value"] = checks["value"].map(lambda v: json.dumps(v, default=str))
    checks.to_csv(output_dir / "health_checks.csv", index=False)

    status_counts = pd.Series([r.status for r in results]).value_counts().to_dict()
    summary = {
        "status": FAIL if status_counts.get(FAIL) else WARN if status_counts.get(WARN) else PASS,
        "status_counts": status_counts,
        "datasets": overviews,
        "checks": [asdict(r) for r in results],
    }
    (output_dir / "health_summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
    )
    (output_dir / "health_report.md").write_text(
        _markdown_report(summary, profiles), encoding="utf-8"
    )
    if labels is not None and "features" in profiles:
        write_descriptive_report(output_dir / "descriptive_report.html", summary, profiles, labels)


def _markdown_report(summary: dict, profiles: dict[str, pd.DataFrame]) -> str:
    lines = [
        "# SECOM data health report",
        "",
        f"**Overall status: {summary['status']}** — "
        + ", ".join(f"{k}: {v}" for k, v in sorted(summary["status_counts"].items())),
        "",
        "## Dataset overview",
        "",
        "| dataset | rows | columns | missing cells % | complete rows | defect % |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, ov in summary["datasets"].items():
        lines.append(
            f"| {name} | {ov.get('rows', '')} | {ov.get('columns', '')} | "
            f"{ov.get('missing_cell_pct', '')} | {ov.get('complete_rows', '')} | "
            f"{ov.get('defect_pct', '')} |"
        )
    lines += [
        "",
        "## Checks",
        "",
        "| dataset | check | status | value | detail |",
        "|---|---|---|---|---|",
    ]
    for check in summary["checks"]:
        value = json.dumps(check["value"], default=str)
        value = value if len(value) <= 60 else value[:57] + "..."
        lines.append(
            f"| {check['dataset']} | {check['check']} | {check['status']} | "
            f"`{value}` | {check['detail']} |"
        )

    profile = profiles.get("features")
    if profile is not None:
        lines += ["", "## Feature highlights (full dataset)", ""]
        top_missing = profile["missing_pct"].sort_values(ascending=False).head(10)
        lines += ["**Most missing features**", "", "| feature | missing % |", "|---|---:|"]
        lines += [f"| {f} | {v:.1f} |" for f, v in top_missing.items()]
        if "mean_diff_fail_vs_pass_in_std" in profile:
            shift = profile["mean_diff_fail_vs_pass_in_std"].abs().sort_values(ascending=False)
            lines += [
                "",
                "**Largest pass/fail mean shift (in feature std units)**",
                "",
                "| feature | shift | mean pass | mean fail |",
                "|---|---:|---:|---:|",
            ]
            for feature in shift.head(10).index:
                row = profile.loc[feature]
                lines.append(
                    f"| {feature} | {row['mean_diff_fail_vs_pass_in_std']:+.3f} | "
                    f"{row['mean_pass']:.4g} | {row['mean_fail']:.4g} |"
                )
    footer = (
        "Full per-feature statistics: `*_profile.csv`; every check: "
        "`health_checks.csv` / `health_summary.json`."
    )
    lines += ["", footer, ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run SECOM data-health checks.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/health"))
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--fail-on-warn", action="store_true", help="exit non-zero on WARN as well as FAIL"
    )
    args = parser.parse_args(argv)

    results = run_health_checks(
        args.data_dir, args.output_dir, test_size=args.test_size, random_state=args.random_state
    )
    width = max(len(f"{r.dataset}.{r.check}") for r in results)
    for r in results:
        value = "" if r.status == PASS else f"  -> {json.dumps(r.value, default=str)}"
        print(f"[{r.status}] {f'{r.dataset}.{r.check}':<{width}}  {r.detail}{value}")
    statuses = {r.status for r in results}
    print(f"\nReports written to {args.output_dir}/")
    if FAIL in statuses or (args.fail_on_warn and WARN in statuses):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

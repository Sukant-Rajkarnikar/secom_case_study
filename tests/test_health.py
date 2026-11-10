import numpy as np
import pandas as pd

from secom_analysis.health import (
    FAIL,
    PASS,
    WARN,
    check_features,
    check_labels,
    check_split,
    profile_features,
)


def _by_check(results):
    return {r.check: r for r in results}


def test_profile_and_feature_checks_flag_quality_issues() -> None:
    features = pd.DataFrame(
        {
            "constant": [1.0] * 6,
            "copy_a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "copy_b": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "missing": [np.nan] * 4 + [1.0, 2.0],
        }
    )
    target = pd.Series([0, 0, 0, 1, 1, 1])
    profile = profile_features(features, target)

    assert profile.loc["constant", "is_constant"]
    assert profile.loc["missing", "missing_pct"] == 4 / 6 * 100
    assert profile.loc["copy_a", "mean_fail"] == 5.0

    checks = _by_check(check_features(features, profile, "toy"))
    assert checks["constant_columns"].items == ["constant"]
    assert checks["duplicate_columns"].items == ["copy_b"]  # constants excluded
    assert checks["high_missing_columns"].items == ["missing"]
    assert checks["infinite_values"].status == PASS


def test_label_checks_reject_bad_domain_and_timestamps() -> None:
    labels = pd.DataFrame({"label": [-1, 1, 2], "timestamp_raw": ["a", "b", "c"]})
    labels["timestamp"] = pd.to_datetime(["2008-07-19 11:00", None, "2008-07-19 10:00"])
    checks = _by_check(check_labels(labels))

    assert checks["label_domain"].status == FAIL
    assert checks["timestamp_parseable"].status == FAIL
    assert checks["timestamp_monotonic"].status == WARN


def test_split_check_detects_leakage() -> None:
    features = pd.DataFrame({"a": range(10)})
    target = pd.Series([0, 1] * 5)
    train, test = features.iloc[:7], features.iloc[5:]
    checks, drift = check_split(train, test, target.iloc[:7], target.iloc[5:], 10)
    checks = _by_check(checks)

    assert checks["no_row_overlap"].status == FAIL
    assert checks["rows_covered"].status == FAIL
    assert "ks_p_value" in drift

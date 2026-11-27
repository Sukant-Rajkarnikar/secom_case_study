"""Week 2: Data health checks, stratified hold-out sealing, rough feature reduction, and 3σ outliers."""

from .health import FAIL, PASS, WARN, CheckResult, Thresholds, run_health_checks
from .preparation import (
    cap_outliers,
    flag_outliers,
    load_split,
    missing_percentage,
    missing_threshold_pareto,
    remove_outliers,
    save_split,
    save_state,
    sip_table,
    three_sigma_bounds,
    verify_sealed,
    zero_variance_features,
)
from .report import write_descriptive_report

__all__ = [
    "PASS",
    "WARN",
    "FAIL",
    "CheckResult",
    "Thresholds",
    "run_health_checks",
    "cap_outliers",
    "flag_outliers",
    "load_split",
    "missing_percentage",
    "missing_threshold_pareto",
    "remove_outliers",
    "save_split",
    "save_state",
    "sip_table",
    "three_sigma_bounds",
    "verify_sealed",
    "zero_variance_features",
    "write_descriptive_report",
]

"""Week 4: Feature selection (Boruta), class balancing (SMOTE/ROSE/ADASYN), and initial confusion matrix metrics."""

from .balancing_and_eval import (
    BALANCER_LABELS,
    COST_FN,
    COST_FP,
    compute_error_metrics,
    evaluate_balancers_cv,
    make_sampler,
    resampling_counts,
)
from .feature_selection import (
    BorutaSelector,
    compare_all_vs_selected_cv,
    cross_check_rankings,
    ensure_prepared_training_split,
    run_boruta_selection,
    save_perc_comparison_plot,
)

__all__ = [
    "BALANCER_LABELS",
    "COST_FN",
    "COST_FP",
    "BorutaSelector",
    "compare_all_vs_selected_cv",
    "compute_error_metrics",
    "cross_check_rankings",
    "ensure_prepared_training_split",
    "evaluate_balancers_cv",
    "make_sampler",
    "resampling_counts",
    "run_boruta_selection",
    "save_perc_comparison_plot",
]

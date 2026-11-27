"""Week 5: Normality proof, Min-Max scaling, leakage-free pipeline, and Training vs. Validation error."""

from .pipeline_and_error import (
    ColumnSelector,
    FrameMinMaxScaler,
    OutlierTreatment,
    build_pipeline,
    complexity_curve,
    normality_audit,
    normality_cv_proof,
    run_pipeline_checks,
)

__all__ = [
    "ColumnSelector",
    "FrameMinMaxScaler",
    "OutlierTreatment",
    "build_pipeline",
    "complexity_curve",
    "normality_audit",
    "normality_cv_proof",
    "run_pipeline_checks",
]

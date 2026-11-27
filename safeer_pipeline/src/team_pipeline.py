"""The team's course pipeline (SECOM project, master), rebuilt as one imblearn Pipeline.

Order and settings are copied from the team code (notebook 02, section 7.2):
  column drops -> 3-sigma capping -> kNN imputer (k=5, standardised) -> Boruta
  -> min-max scaling -> SMOTE -> random forest (500 trees), seed 42 everywhere.

One deliberate difference: the team drops constant and >60%-missing columns once, on all
1,253 training wafers, before cross-validation. Here that step is inside the pipeline, so it
is refitted on each fold's training rows only. It uses no labels, so the effect is small,
but it means no row that is scored has influenced any step.
"""

import numpy as np
import pandas as pd
from boruta import BorutaPy
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler

RANDOM_STATE = 42
MISSING_THRESHOLD_PCT = 60   # team decision: drop columns with more than 60% missing


class ColumnDropper(BaseEstimator, TransformerMixin):
    """Purpose: drop constant columns (variance 0 or undefined) and columns with more than
    `missing_pct`% missing values, judged on the training rows only.
    Inputs: DataFrame with NaN. Outputs: DataFrame with the kept columns."""

    def __init__(self, missing_pct=MISSING_THRESHOLD_PCT):
        self.missing_pct = missing_pct

    def fit(self, X, y=None):
        variance = X.var(axis=0, skipna=True)
        constant = variance.eq(0) | variance.isna()
        too_missing = X.isna().mean().mul(100) > self.missing_pct
        self.keep_ = X.columns[~(constant | too_missing)]
        return self

    def transform(self, X):
        return X[self.keep_]


class ThreeSigmaCapper(BaseEstimator, TransformerMixin):
    """Purpose: team decision for outliers. Values outside mean +/- 3 SD (learned on training
    rows, NaN ignored) are replaced by that boundary. NaN stays NaN.
    Inputs/outputs: DataFrame, same columns."""

    def __init__(self, k=3.0):
        self.k = k

    def fit(self, X, y=None):
        mean, std = X.mean(), X.std()
        self.lower_, self.upper_ = mean - self.k * std, mean + self.k * std
        return self

    def transform(self, X):
        return X.clip(lower=self.lower_, upper=self.upper_, axis=1)


class ScaledKNNImputer(BaseEstimator, TransformerMixin):
    """Purpose: team decision for missing values. Standardise each column (training mean/SD),
    fill gaps from the 5 most similar wafers (distance-weighted), map back to original units,
    clip to the training range. Observed values are never changed. No mean imputation.
    Inputs: DataFrame with NaN. Outputs: complete DataFrame, same columns."""

    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors

    def fit(self, X, y=None):
        values = X.to_numpy(dtype=float)
        self.mean_ = np.nanmean(values, axis=0)
        std = np.nanstd(values, axis=0, ddof=1)
        self.scale_ = np.where(std > 0, std, 1.0)
        self.min_, self.max_ = np.nanmin(values, axis=0), np.nanmax(values, axis=0)
        self.knn_ = KNNImputer(n_neighbors=self.n_neighbors, weights="distance").fit(
            (values - self.mean_) / self.scale_)
        return self

    def transform(self, X):
        values = X.to_numpy(dtype=float)
        filled = self.knn_.transform((values - self.mean_) / self.scale_)
        filled = np.clip(filled * self.scale_ + self.mean_, self.min_, self.max_)
        return pd.DataFrame(np.where(np.isnan(values), filled, values),
                            index=X.index, columns=X.columns)


class BorutaSelector(BaseEstimator, TransformerMixin):
    """Purpose: team decision for feature selection. Boruta keeps a sensor only if it beats
    shuffled copies of the sensors in a random forest (max_depth 5, class-weighted).
    Confirmed and tentative features are kept, as in the team code.
    Inputs: complete DataFrame and labels. Outputs: DataFrame with the selected columns."""

    def __init__(self, seed=RANDOM_STATE, max_iter=100):
        self.seed = seed
        self.max_iter = max_iter

    def fit(self, X, y):
        forest = RandomForestClassifier(max_depth=5, class_weight="balanced", n_jobs=-1,
                                        random_state=self.seed)
        self.boruta_ = BorutaPy(forest, n_estimators="auto", max_iter=self.max_iter,
                                random_state=self.seed).fit(X.to_numpy(float), np.asarray(y))
        support = self.boruta_.support_ | self.boruta_.support_weak_
        if not support.any():                     # never hand the model an empty matrix
            support[np.argmin(self.boruta_.ranking_)] = True
        self.support_ = support
        self.selected_ = list(X.columns[support])
        return self

    def transform(self, X):
        return X.iloc[:, self.support_]


def build_team_pipeline():
    """Purpose: the full pipeline, unfitted. Every step is refitted on whatever rows fit()
    gets, so in cross-validation nothing is learned from the held-out fold.
    SMOTE sits in an imblearn Pipeline, so it runs only during fit, never on scored rows."""
    return Pipeline([
        ("drop_columns", ColumnDropper()),
        ("outliers", ThreeSigmaCapper(k=3.0)),
        ("imputer", ScaledKNNImputer(n_neighbors=5)),
        ("selector", BorutaSelector(seed=RANDOM_STATE)),
        ("scaler", MinMaxScaler().set_output(transform="pandas")),
        ("balancer", SMOTE(k_neighbors=5, random_state=RANDOM_STATE)),
        ("model", RandomForestClassifier(n_estimators=500, n_jobs=-1,
                                         random_state=RANDOM_STATE)),
    ])


# Manual choices this pipeline encodes (counted for the effort comparison). Each one is a
# setting the team had to decide and defend; the list is the source for that number.
MANUAL_DECISIONS = [
    "constant-column rule (variance exactly 0)",
    "missing-value threshold (60%)",
    "outlier rule (3 sigma)",
    "outlier treatment (cap, not remove or keep)",
    "imputation method (kNN, not mean or median)",
    "k for kNN (5) and distance weighting",
    "standardise before kNN",
    "feature selection method (Boruta)",
    "Boruta settings (forest depth 5, class weights, 100 iterations, keep tentative)",
    "scaling method (min-max) and its position after selection",
    "balancing method (SMOTE) and its k (5)",
    "classifier (random forest) and size (500 trees)",
    "step order",
]

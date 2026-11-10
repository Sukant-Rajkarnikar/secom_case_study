import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestClassifier
from plots import plot_section2_visualizations, plot_low_vola, plot_missingness


# Loading SECOM dataset in real time from UCI repository
def load_secom_data():
    print("Downloading SECOM dataset via pandas...")
    base = "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/"

    X = pd.read_csv(base + "secom.data", sep=" ", header=None)
    labels = pd.read_csv(
        base + "secom_labels.data", sep=" ", header=None, names=["target", "timestamp"]
    )

    y = labels["target"]
    X.columns = [f"sensor_{i}" for i in range(X.shape[1])]

    print(f"\nTotal wafers (rows): {len(X)}")
    print(f"Total labels (rows): {len(y)}")
    print(f"Total sensors (columns): {len(X.columns)}")
    return X, y

def describe_secom_data(X,y):
    print("\n--- Descriptive Statistics & Initial Inspection ---")

    print("\nColumn Data Types:")
    print(X.dtypes.value_counts())
    print(f"Duplicate wafer rows: {X.duplicated().sum()}")

    class_counts = y.value_counts()
    class_pct = (y.value_counts(normalize=True) * 100).round(2)
    print("\nTarget Class Distribution (Pass = -1, Fail = 1):")
    for label in [-1, 1]:
        print(f"  Class {label:2d}: {class_counts[label]} wafers ({class_pct[label]}%)")

    feature_summary = X.describe().T[
        ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
    ]
    print("\nSample Sensor Descriptive Statistics (First 10 Sensors):")
    print(feature_summary.head(10).round(3))
    
    total_missing = X.isna().sum().sum()
    missing_pct_overall = (total_missing / X.size) * 100
    missing_by_feature = (X.isna().mean() * 100).sort_values(ascending=False)
    missing_by_class = X.isna().sum(axis=1).groupby(y).mean()
    print(f"\nTotal missing cells: {total_missing} ({missing_pct_overall:.2f}% of dataset)")
    print("\nTop 10 Sensors by Missing Percentage (%):")
    print(missing_by_feature.head(10).round(2))
    print("\nAverage missing sensors per wafer by class (Full Dataset):")
    print(f"Pass (-1): {missing_by_class[-1]:.2f} | Fail (1): {missing_by_class[1]:.2f}")
    plot_section2_visualizations(X, y)
    return feature_summary


# Splitting dataset into training and test sets with stratification to maintain class distribution
def split_training_test_data(X, y):

    print("\nPerforming Stratified 80/20 Split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print(f"Training set: {len(X_train)} wafers")
    print(f"Test set: {len(X_test)} wafers")
    
    comparison = pd.DataFrame({
        "Original (%)": (y.value_counts(normalize=True) * 100).round(2),
        "Training (%)": (y_train.value_counts(normalize=True) * 100).round(2),
        "Test (%)": (y_test.value_counts(normalize=True) * 100).round(2),
        "Original Count": y.value_counts(),
        "Training Count": y_train.value_counts(),
        "Test Count": y_test.value_counts(),
    })
    print("\nTarget Distribution Before and After Split:")
    print(comparison.to_string())

    return X_train, X_test, y_train, y_test


# Reducing features
def reduce_features(X_train, y_train):
    variances = X_train.var(numeric_only=True)
    constant_cols = variances[variances == 0].index.tolist()

    X_train_reduced = X_train.drop(columns=constant_cols)
    # X_test_reduced = X_test.drop(columns=constant_cols)
    print(f"Dropped {len(constant_cols)} constant sensors.")

    missing_pct = X_train_reduced.isna().mean() * 100
    plot_missingness(missing_pct)
    high_missing_cols = missing_pct[missing_pct > 60].index.tolist()

    print("\nMissingness Breakdown for High-Missing Columns")
    for col in high_missing_cols:
        col_missing = X_train_reduced[col].isna().groupby(y_train).mean() * 100
        print(
            f"{col} -> Pass (-1): {col_missing[-1]:.1f}% | Fail (1): {col_missing[1]:.1f}%"
        )

    missing_by_class = X_train_reduced.isna().sum(axis=1).groupby(y_train).mean()
    print("\nAverage missing sensors per wafer:")
    print(f"Pass (-1): {missing_by_class[-1]:.2f}")
    print(f"Fail (1): {missing_by_class[1]:.2f}")

    print(f"Duplicate rows in X_train: {X_train_reduced.duplicated().sum()}")
    duplicate_cols = X_train_reduced.columns[X_train_reduced.T.duplicated()].tolist()
    X_train_reduced = X_train_reduced.drop(columns=duplicate_cols)
    print(f"Dropped {len(duplicate_cols)} duplicate sensors.")

    non_zero_var = X_train_reduced.var(numeric_only=True)
    low_var_subset = non_zero_var[non_zero_var < 0.01]
    print(f"Sensors with low variance (0 < var < 0.01): {len(low_var_subset)} (Action: Keep all)")
    plot_low_vola(non_zero_var)

    X_train_reduced = X_train_reduced.drop(columns=high_missing_cols)
    # X_test_reduced = X_test_reduced.drop(columns=high_missing_cols)
    print(f"Dropped {len(high_missing_cols)} sensors with >60% missing data.")
    print(f"Sensors remaining: {X_train_reduced.shape[1]}")

    return X_train_reduced


# Outlier detection
def treat_outliers(X_train):
    col_mean = X_train.mean(numeric_only=True)
    col_std = X_train.std(numeric_only=True)
    lower_bound = col_mean - 3 * col_std
    upper_bound = col_mean + 3 * col_std

    outlier_mask_train = (X_train < lower_bound) | (X_train > upper_bound)
    total_flagged_train = outlier_mask_train.sum().sum()
    print(f"Total outliers flagged in training set: {total_flagged_train}")

    X_train_cleaned = X_train.clip(lower=lower_bound, upper=upper_bound, axis=1)
    # X_test_cleaned = X_test.clip(lower=lower_bound, upper=upper_bound, axis=1)
    return X_train_cleaned


# KNN imputation
def knn_imputation(X_train):
    print("\nPerforming KNN Imputation...")

    scaler = MinMaxScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    # X_test_scaled = pd.DataFrame(
    #     scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    # )

    imputer = KNNImputer(n_neighbors=5)
    X_train_imputed = pd.DataFrame(
        imputer.fit_transform(X_train_scaled),
        columns=X_train.columns,
        index=X_train.index,
    )
    # X_test_imputed = pd.DataFrame(
    #     imputer.transform(X_test_scaled), columns=X_test.columns, index=X_test.index
    # )

    print(f"Missing values remaining in Train: {X_train_imputed.isna().sum().sum()}")
    # print(f"Missing values remaining in Test: {X_test_imputed.isna().sum().sum()}")

    return X_train_imputed


def evaluate_baseline(X_train, y_train):
    print("\nBaseline Random Forest (Pre-Feature Selection)...")

    rf_baseline = RandomForestClassifier(random_state=42)
    rf_baseline.fit(X_train, y_train)

    train_accuracy = rf_baseline.score(X_train, y_train) * 100
    print(f"Baseline Training Accuracy: {train_accuracy:.2f}%")

    return rf_baseline


if __name__ == "__main__":
    X, y = load_secom_data()
    feature_summary = describe_secom_data(X, y)
    feature_summary.to_csv("results/descriptive_statistics.csv")
    X_train, X_test, y_train, y_test = split_training_test_data(X, y)
    X_train_reduced = reduce_features(X_train, y_train)
    X_train_cleaned = treat_outliers(X_train_reduced)
    X_train_imputed = knn_imputation(X_train_cleaned)
    rf_baseline = evaluate_baseline(X_train_imputed, y_train)

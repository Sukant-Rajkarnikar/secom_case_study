import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestClassifier


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
    print(f"Total sensors (columns): {len(X.columns)}")
    return X, y


# Splitting dataset into training and test sets with stratification to maintain class distribution
def split_training_test_data(X, y):

    print("\nPerforming Stratified 80/20 Split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print(f"Training set: {len(X_train)} wafers")
    print(f"Test set: {len(X_test)} wafers")
    print("\nTraining Set Imbalance (%):")
    print((y_train.value_counts(normalize=True) * 100).round(2))

    return X_train, X_test, y_train, y_test


# Reducing features
def reduce_features(X_train, X_test, y_train):
    variances = X_train.var(numeric_only=True)
    constant_cols = variances[variances == 0].index.tolist()

    X_train_reduced = X_train.drop(columns=constant_cols)
    X_test_reduced = X_test.drop(columns=constant_cols)
    print(f"Dropped {len(constant_cols)} constant sensors.")

    missing_pct = X_train_reduced.isna().mean() * 100
    high_missing_cols = missing_pct[missing_pct > 60].index.tolist()

    print("\nMissingness Breakdown for High-Missing Columns")
    for col in high_missing_cols:
        col_missing = X_train_reduced[col].isna().groupby(y_train).mean() * 100
        print(f"{col} -> Pass (-1): {col_missing[-1]:.1f}% | Fail (1): {col_missing[1]:.1f}%")

    missing_by_class = X_train_reduced.isna().sum(axis=1).groupby(y_train).mean()
    print("\nAverage missing sensors per wafer:")
    print(f"Pass (-1): {missing_by_class[-1]:.2f}")
    print(f"Fail (1): {missing_by_class[1]:.2f}")

    X_train_reduced = X_train_reduced.drop(columns=high_missing_cols)
    X_test_reduced = X_test_reduced.drop(columns=high_missing_cols)
    print(f"Dropped {len(high_missing_cols)} sensors with >60% missing data.")
    print(f"Sensors remaining: {X_train_reduced.shape[1]}")

    return X_train_reduced, X_test_reduced

#Outlier detection
def treat_outliers(X_train, X_test):
    col_mean = X_train.mean(numeric_only=True)
    col_std = X_train.std(numeric_only=True)
    lower_bound = col_mean - 3 * col_std
    upper_bound = col_mean + 3 * col_std

    outlier_mask_train = (X_train < lower_bound) | (X_train > upper_bound)
    total_flagged_train = outlier_mask_train.sum().sum()
    print(f"Total outliers flagged in training set: {total_flagged_train}")

    X_train_cleaned = X_train.clip(lower=lower_bound, upper=upper_bound, axis=1)
    X_test_cleaned = X_test.clip(lower=lower_bound, upper=upper_bound, axis=1)
    return X_train_cleaned, X_test_cleaned

#KNN imputation
def knn_imputation(X_train, X_test):
    print("\nPerforming KNN Imputation...")

    scaler = MinMaxScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

    imputer = KNNImputer(n_neighbors=5)
    X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train_scaled), columns=X_train.columns, index=X_train.index)
    X_test_imputed = pd.DataFrame(imputer.transform(X_test_scaled), columns=X_test.columns, index=X_test.index)

    print(f"Missing values remaining in Train: {X_train_imputed.isna().sum().sum()}")
    print(f"Missing values remaining in Test: {X_test_imputed.isna().sum().sum()}")

    return X_train_imputed, X_test_imputed

def evaluate_baseline(X_train, y_train):
    print("\nBaseline Random Forest (Pre-Feature Selection)...")
    
    rf_baseline = RandomForestClassifier(random_state=42)
    rf_baseline.fit(X_train, y_train)
    
    train_accuracy = rf_baseline.score(X_train, y_train) * 100
    print(f"Baseline Training Accuracy: {train_accuracy:.2f}%")
    
    return rf_baseline

if __name__ == "__main__":
    X, y = load_secom_data()
    X_train, X_test, y_train, y_test = split_training_test_data(X, y)
    X_train_reduced, X_test_reduced = reduce_features(X_train, X_test, y_train)
    X_train_cleaned, X_test_cleaned = treat_outliers(X_train_reduced, X_test_reduced)
    X_train_imputed, X_test_imputed = knn_imputation(X_train_cleaned, X_test_cleaned)
    rf_baseline = evaluate_baseline(X_train_imputed, y_train)
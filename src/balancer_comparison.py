import argparse
from pathlib import Path

import pandas as pd
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import MinMaxScaler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    args = parser.parse_args()

    train = pd.read_csv(args.input, index_col=0)
    y = train["defective"].astype(int)
    X = train.drop(columns=["defective"])

    balancers = {
        "none": None,
        "rose": RandomOverSampler,
        "smote": SMOTE,
        "adasyn": ADASYN,
    }
    rows = []
    for seed in args.seeds:
        cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=seed)
        for name, balancer_type in balancers.items():
            balancer = None if balancer_type is None else balancer_type(random_state=seed)
            model = RandomForestClassifier(n_estimators=300, n_jobs=-1,
                                           random_state=seed)
            steps = [
                ("knn", KNNImputer(n_neighbors=5, weights="distance")),
                ("scale", MinMaxScaler()),
            ]
            if balancer is not None:
                steps.append(("balance", balancer))
            steps.append(("model", model))
            pipe = Pipeline(steps)
            scores = cross_validate(
                pipe, X, y, cv=cv,
                scoring=["balanced_accuracy", "recall", "f1", "roc_auc"],
                n_jobs=-1,
            )
            rows.append({
                "balancer": name,
                "seed": seed,
                "balanced_accuracy": scores["test_balanced_accuracy"].mean(),
                "recall": scores["test_recall"].mean(),
                "f1": scores["test_f1"].mean(),
                "roc_auc": scores["test_roc_auc"].mean(),
            })

    output = Path("reports") / "balancer_comparison.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(output)


if __name__ == "__main__":
    main()

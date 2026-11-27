import argparse
from pathlib import Path

import pandas as pd
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(42, 52)))
    args = parser.parse_args()

    train = pd.read_csv(args.input, index_col=0)
    y = train["defective"].astype(int)
    X = train.drop(columns=["defective"])

    from sklearn.impute import KNNImputer
    X = pd.DataFrame(KNNImputer(n_neighbors=5, weights="distance").fit_transform(X),
                     columns=X.columns)

    stability = pd.DataFrame(0, index=X.columns,
                             columns=["confirmed_count", "tentative_count", "rejected_count"])
    stability.index.name = "feature"

    for seed in args.seeds:
        rf = RandomForestClassifier(n_estimators=300, n_jobs=-1,
                                    class_weight="balanced", random_state=seed)
        selector = BorutaPy(rf, n_estimators="auto", max_iter=50,
                            random_state=seed, verbose=0)
        selector.fit(X.to_numpy(), y.to_numpy())
        for column, confirmed, tentative in zip(X.columns,
                                                selector.support_,
                                                selector.support_weak_):
            if confirmed:
                stability.loc[column, "confirmed_count"] += 1
            elif tentative:
                stability.loc[column, "tentative_count"] += 1
            else:
                stability.loc[column, "rejected_count"] += 1

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    stability.reset_index().to_csv(output, index=False)
    print(output)


if __name__ == "__main__":
    main()

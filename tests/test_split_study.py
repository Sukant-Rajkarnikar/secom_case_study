import numpy as np
import pandas as pd

from secom_analysis.split_study import (
    FoldPreparation,
    holdouts,
    leave_one_month_out,
    time_forward,
    time_holdout,
)


def test_fold_preparation_learns_only_from_fit_rows():
    train = pd.DataFrame(
        {
            "constant": [1.0] * 10,
            "mostly_missing": [np.nan] * 7 + [1.0, 2.0, 3.0],
            "signal": np.arange(10, dtype=float),
        }
    )
    prep = FoldPreparation().fit(train)
    assert prep.columns_ == ["signal"]

    test = pd.DataFrame({"constant": [5.0], "mostly_missing": [9.0], "signal": [1000.0]})
    out = prep.transform(test)
    assert list(out.columns) == ["signal"]
    # Capped at the train-fitted 3σ bound, not at anything learned from the test row.
    assert out.iloc[0, 0] == prep.bounds_.loc["signal", "upper"]


def test_stratified_holdout_fixes_minority_count():
    target = pd.Series([1] * 10 + [0] * 90)
    counts = {
        int(target.iloc[test].sum()) for _, _, _, test in holdouts(target, range(20), stratify=True)
    }
    assert counts == {2}


def test_time_schemes_never_train_on_the_future():
    order = np.array([3, 0, 4, 1, 2, 5, 9, 6, 8, 7])  # row ids sorted by timestamp
    position = {row: rank for rank, row in enumerate(order)}
    for _, _, train, test in [*time_holdout(order), *time_forward(order, folds=3)]:
        assert max(position[r] for r in train) < min(position[r] for r in test)


def test_leave_one_month_out_covers_every_row_once():
    month = pd.Series(["2008-07", "2008-08", "2008-08", "2008-09"])
    tests = [test for _, _, _, test in leave_one_month_out(month)]
    assert sorted(np.concatenate(tests).tolist()) == [0, 1, 2, 3]

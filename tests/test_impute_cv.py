import numpy as np

from secom_analysis.impute_cv import corrected_ttest, score


def test_corrected_ttest_is_more_conservative_than_naive() -> None:
    diffs = np.array([0.02, 0.01, 0.03, 0.00, 0.02, 0.01, 0.02, 0.03, 0.01, 0.02])
    t_corr, p_corr = corrected_ttest(diffs, n_train=800, n_test=200)
    t_naive = diffs.mean() / (diffs.std(ddof=1) / np.sqrt(len(diffs)))
    assert 0 < t_corr < t_naive
    assert 0 < p_corr < 1


def test_score_perfect_ranking() -> None:
    y = np.array([0] * 90 + [1] * 10)
    proba = np.linspace(0, 1, 100)
    result = score(y, proba)
    assert np.isclose(result["pr_auc"], 1.0)
    assert np.isclose(result["roc_auc"], 1.0)
    assert result["recall_at_fpr10"] == 1.0

"""Integration tests of the whole training run on the synthetic customers (about 3.3 s for all candidates)."""

import pandas as pd
import pytest

from campaign_response.modeling.candidates import CANDIDATE_NAMES, UnknownCandidateError
from campaign_response.modeling import training
from campaign_response.modeling.training import TrainingResult, evaluate_model, train_candidates

ALL_BUT_BASELINE = ["logistic_regression", "random_forest", "hist_gradient_boosting"]


@pytest.fixture(scope="module")
def data(customers: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return customers.drop(columns=["ID", "Response"]), customers["Response"].astype(int)


@pytest.fixture(scope="module")
def trained(data: tuple[pd.DataFrame, pd.Series]) -> TrainingResult:
    return train_candidates(*data, ALL_BUT_BASELINE, n_iter=1, cv_folds=3, seed=42)


def test_train_baseline_always_searched_first(trained: TrainingResult) -> None:
    assert tuple(result.name for result in trained.candidates) == CANDIDATE_NAMES


def test_train_baseline_never_selected(trained: TrainingResult) -> None:
    assert trained.best.name != "dummy"
    assert trained.best.best.mean_pr_auc > trained.baseline.best.mean_pr_auc


def test_train_model_fitted_with_best_parameters(trained: TrainingResult) -> None:
    fitted = trained.model.estimator_.named_steps["model"].get_params()

    assert all(fitted[key] == value for key, value in trained.best.best.parameters.items())
    assert 0 < trained.threshold < 1


def test_train_summary_values(trained: TrainingResult) -> None:
    summary = trained.summary()

    assert summary["best_candidate"] == trained.best.name
    assert summary["cv_pr_auc"] == trained.best.best.mean_pr_auc
    assert summary["baseline_cv_pr_auc"] == trained.baseline.best.mean_pr_auc
    assert summary["threshold"] == trained.threshold


def test_evaluate_model_uses_model_threshold(
    trained: TrainingResult, data: tuple[pd.DataFrame, pd.Series]
) -> None:
    metrics = evaluate_model(trained.model, *data)

    assert metrics.threshold == trained.threshold
    assert metrics.n_samples == len(data[1])
    assert metrics.true_positives + metrics.false_positives == trained.model.predict(data[0]).sum()


def test_train_duplicates_and_explicit_baseline_once(data: tuple[pd.DataFrame, pd.Series]) -> None:
    result = train_candidates(*data, ["logistic_regression", "dummy", "logistic_regression"], n_iter=1, cv_folds=3,
                              seed=42)

    assert [candidate.name for candidate in result.candidates] == ["dummy", "logistic_regression"]


def test_train_unknown_name_rejected_before_training(
    data: tuple[pd.DataFrame, pd.Series], monkeypatch: pytest.MonkeyPatch
) -> None:
    searched: list[str] = []
    monkeypatch.setattr(training, "search_candidate", lambda candidate, *args, **kwargs: searched.append(candidate))

    with pytest.raises(UnknownCandidateError, match="svm"):
        train_candidates(*data, ["logistic_regression", "svm"], n_iter=1, cv_folds=3, seed=42)

    assert searched == []


@pytest.mark.parametrize("names", [[], ["dummy"]])
def test_train_only_baseline_ValueError(data: tuple[pd.DataFrame, pd.Series], names: list[str]) -> None:
    with pytest.raises(ValueError, match="besides the baseline"):
        train_candidates(*data, names, n_iter=1, cv_folds=3, seed=42)

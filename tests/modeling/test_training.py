"""Tests for the cross-validated candidate search and the selection of the best candidate."""

import logging

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.model_selection import TunedThresholdClassifierCV

from campaign_response.modeling.candidates import CandidateName, get_candidate
from campaign_response.modeling.training import (
    CandidateResult,
    ParameterSetScore,
    build_pipeline,
    decision_threshold,
    fit_with_tuned_threshold,
    search_candidate,
    select_best,
)


@pytest.fixture
def data(customers: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return customers.drop(columns=["ID", "Response"]), customers["Response"].astype(int)


def stub(name: CandidateName, *means: float) -> CandidateResult:
    """A result whose parameter sets have the given mean PR-AUC (two identical folds each)."""
    scores = tuple(ParameterSetScore({"set": index}, (mean, mean)) for index, mean in enumerate(means))
    return CandidateResult(name, scores)


def test_build_pipeline_steps_and_seed() -> None:
    pipeline = build_pipeline(get_candidate("random_forest"), seed=11)

    assert [name for name, _ in pipeline.steps] == ["features", "preprocess", "model"]
    assert pipeline.get_params()["model__random_state"] == 11


def test_build_pipeline_applies_parameters() -> None:
    pipeline = build_pipeline(get_candidate("logistic_regression"), seed=1, parameters={"C": 0.1})

    assert pipeline.get_params()["model__C"] == 0.1


def test_build_pipeline_fits_raw_customer_columns(data: tuple[pd.DataFrame, pd.Series]) -> None:
    features, target = data

    fitted = clone(build_pipeline(get_candidate("logistic_regression"), 42)).fit(features, target)

    assert fitted.predict_proba(features).shape == (len(features), 2)


def test_parameter_set_score_mean_std() -> None:
    score = ParameterSetScore({}, (0.5, 0.7))

    assert score.mean_pr_auc == pytest.approx(0.6)
    assert score.std_pr_auc == pytest.approx(0.1)


def test_candidate_result_best_highest_mean() -> None:
    assert stub("random_forest", 0.5, 0.7, 0.6).best.parameters == {"set": 1}


def test_candidate_result_tie_keeps_defaults() -> None:
    assert stub("random_forest", 0.6, 0.6).best.parameters == {"set": 0}


def test_select_best_highest_candidate() -> None:
    results = [
        stub("logistic_regression", 0.61), stub("random_forest", 0.58, 0.64), stub("hist_gradient_boosting", 0.6),
    ]

    assert select_best(results).name == "random_forest"


def test_select_best_never_baseline() -> None:
    results = [stub("dummy", 0.9), stub("logistic_regression", 0.2)]

    assert select_best(results).name == "logistic_regression"


def test_select_best_tie_first_candidate() -> None:
    results = [stub("hist_gradient_boosting", 0.6), stub("logistic_regression", 0.6)]

    assert select_best(results).name == "hist_gradient_boosting"


@pytest.mark.parametrize("names", [[], ["dummy"]])
def test_select_best_no_contender_ValueError(names: list[CandidateName]) -> None:
    with pytest.raises(ValueError, match="besides the baseline"):
        select_best([stub(name, 0.5) for name in names])


def test_search_candidate_scores_every_set(data: tuple[pd.DataFrame, pd.Series]) -> None:
    candidate = get_candidate("logistic_regression")

    result = search_candidate(candidate, *data, n_iter=3, cv_folds=3, seed=42)

    assert result.name == "logistic_regression"
    assert [score.parameters for score in result.scores] == [{"C": 1.0}, {"C": 0.01}, {"C": 3.0}]
    assert all(len(score.fold_pr_aucs) == 3 for score in result.scores)
    assert result.best.mean_pr_auc > 0.5


def test_search_candidate_baseline_scores_prevalence(data: tuple[pd.DataFrame, pd.Series]) -> None:
    features, target = data

    result = search_candidate(get_candidate("dummy"), features, target, n_iter=5, cv_folds=3, seed=42)

    assert len(result.scores) == 1
    assert result.best.mean_pr_auc == pytest.approx(target.mean(), abs=0.01)


def test_search_candidate_same_seed_identical(data: tuple[pd.DataFrame, pd.Series]) -> None:
    candidate = get_candidate("hist_gradient_boosting")

    first = search_candidate(candidate, *data, n_iter=2, cv_folds=3, seed=5)
    second = search_candidate(candidate, *data, n_iter=2, cv_folds=3, seed=5)

    assert first == second


def test_search_candidate_logs_best(data: tuple[pd.DataFrame, pd.Series], caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="campaign_response.modeling.training"):
        result = search_candidate(get_candidate("logistic_regression"), *data, n_iter=1, cv_folds=3, seed=42)

    record = caplog.records[-1]
    assert record.__dict__["candidate"] == "logistic_regression"
    assert record.__dict__["best_cv_pr_auc"] == round(result.best.mean_pr_auc, 4)


def test_search_candidate_fit_error_raises_not_nan(data: tuple[pd.DataFrame, pd.Series]) -> None:
    features, target = data

    with pytest.raises(KeyError, match="Income"):
        search_candidate(get_candidate("logistic_regression"), features.drop(columns=["Income"]), target,
                         n_iter=1, cv_folds=3, seed=42)


@pytest.fixture(scope="module")
def tuned(customers: pd.DataFrame) -> TunedThresholdClassifierCV:
    features, target = customers.drop(columns=["ID", "Response"]), customers["Response"].astype(int)
    return fit_with_tuned_threshold(get_candidate("logistic_regression"), {"C": 0.3}, features, target,
                                    cv_folds=3, seed=42)


def test_tuned_threshold_inside_unit_interval(tuned: TunedThresholdClassifierCV) -> None:
    assert 0 < decision_threshold(tuned) < 1


def test_tuned_predict_uses_threshold(tuned: TunedThresholdClassifierCV, data: tuple[pd.DataFrame, pd.Series]) -> None:
    features = data[0]

    scores = tuned.predict_proba(features)[:, 1]

    np.testing.assert_array_equal(tuned.predict(features), (scores >= decision_threshold(tuned)).astype(int))


def test_tuned_refitted_with_chosen_parameters(tuned: TunedThresholdClassifierCV) -> None:
    assert tuned.estimator_.get_params()["model__C"] == 0.3


def test_tuned_same_seed_same_threshold(
    tuned: TunedThresholdClassifierCV, data: tuple[pd.DataFrame, pd.Series]
) -> None:
    again = fit_with_tuned_threshold(get_candidate("logistic_regression"), {"C": 0.3}, *data, cv_folds=3, seed=42)

    assert decision_threshold(again) == decision_threshold(tuned)


def test_tuned_logs_threshold(data: tuple[pd.DataFrame, pd.Series], caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="campaign_response.modeling.training"):
        model = fit_with_tuned_threshold(get_candidate("logistic_regression"), {}, *data, cv_folds=3, seed=42)

    assert caplog.records[-1].__dict__["threshold"] == round(decision_threshold(model), 4)

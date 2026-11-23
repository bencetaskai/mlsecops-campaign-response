"""Tests for the candidate allow-list and the parameter sets tried per candidate."""

import numpy as np
import pytest
from sklearn.base import clone, is_classifier
from sklearn.model_selection import ParameterGrid

from campaign_response.modeling.candidates import (
    BASELINE,
    CANDIDATE_NAMES,
    CANDIDATES,
    UnknownCandidateError,
    get_candidate,
    parameter_sets,
)


def test_candidates_names_match_literal() -> None:
    assert tuple(CANDIDATES) == CANDIDATE_NAMES
    assert all(CANDIDATES[name].name == name for name in CANDIDATE_NAMES)
    assert BASELINE in CANDIDATE_NAMES


@pytest.mark.parametrize("name", CANDIDATE_NAMES)
def test_candidate_builds_fittable_classifier(name: str) -> None:
    features = np.random.default_rng(0).normal(size=(40, 3))
    target = np.array([0, 1] * 20)

    estimator = get_candidate(name).build(42)

    assert is_classifier(estimator)
    assert clone(estimator).fit(features, target).predict_proba(features).shape == (40, 2)


@pytest.mark.parametrize("name", CANDIDATE_NAMES)
def test_candidate_space_contains_defaults(name: str) -> None:
    candidate = get_candidate(name)

    defaults = candidate.default_parameters()

    assert all(value in candidate.search_space[key] for key, value in defaults.items())


@pytest.mark.parametrize("name", CANDIDATE_NAMES)
def test_candidate_space_valid_parameters(name: str) -> None:
    candidate = get_candidate(name)

    for point in ParameterGrid(dict(candidate.search_space)):
        candidate.build(42).set_params(**point)


def test_candidate_seed_reaches_estimator() -> None:
    assert get_candidate("random_forest").build(7).get_params()["random_state"] == 7


@pytest.mark.parametrize("name", ["svm", "sklearn.svm.SVC", "", "Dummy"])
def test_get_candidate_unknown_name_rejected(name: str) -> None:
    with pytest.raises(UnknownCandidateError, match="allowed: dummy, logistic_regression"):
        get_candidate(name)


def test_parameter_sets_defaults_first() -> None:
    candidate = get_candidate("hist_gradient_boosting")

    sets = parameter_sets(candidate, n_iter=5, seed=42)

    assert sets[0] == candidate.default_parameters()
    assert len(sets) == 5
    assert candidate.default_parameters() not in sets[1:]


def test_parameter_sets_no_duplicates() -> None:
    sets = parameter_sets(get_candidate("random_forest"), n_iter=20, seed=42)

    assert len({tuple(sorted(point.items(), key=str)) for point in sets}) == 20


def test_parameter_sets_capped_at_grid_size() -> None:
    assert len(parameter_sets(get_candidate("logistic_regression"), n_iter=100, seed=42)) == 7
    assert parameter_sets(get_candidate("dummy"), n_iter=10, seed=42) == [{}]


def test_parameter_sets_same_seed_same_sets() -> None:
    candidate = get_candidate("hist_gradient_boosting")

    assert parameter_sets(candidate, 10, seed=1) == parameter_sets(candidate, 10, seed=1)
    assert parameter_sets(candidate, 10, seed=1) != parameter_sets(candidate, 10, seed=2)


def test_parameter_sets_one_iteration_defaults_only() -> None:
    candidate = get_candidate("random_forest")

    assert parameter_sets(candidate, n_iter=1, seed=42) == [candidate.default_parameters()]


@pytest.mark.parametrize("n_iter", [0, -3])
def test_parameter_sets_below_one_ValueError(n_iter: int) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        parameter_sets(get_candidate("logistic_regression"), n_iter=n_iter, seed=42)

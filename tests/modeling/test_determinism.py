"""Round-trip determinism of the whole training run: the property a reproducible MLflow run depends on."""

import pickle

import numpy as np
import pandas as pd
import pytest

from campaign_response.modeling.training import TrainingResult, evaluate_model, train_candidates

# The random forest is left out for speed; its seed handling is tested in test_training.py.
CANDIDATES = ["logistic_regression", "hist_gradient_boosting"]


@pytest.fixture(scope="module")
def data(customers: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return customers.drop(columns=["ID", "Response"]), customers["Response"].astype(int)


def train(data: tuple[pd.DataFrame, pd.Series], seed: int) -> TrainingResult:
    return train_candidates(*data, CANDIDATES, n_iter=2, cv_folds=3, seed=seed)


@pytest.fixture(scope="module")
def first(data: tuple[pd.DataFrame, pd.Series]) -> TrainingResult:
    return train(data, seed=42)


def test_same_seed_identical_run(first: TrainingResult, data: tuple[pd.DataFrame, pd.Series]) -> None:
    second = train(data, seed=42)

    assert second.summary() == first.summary()
    assert second.candidates == first.candidates
    assert evaluate_model(second.model, *data) == evaluate_model(first.model, *data)
    np.testing.assert_array_equal(second.model.predict_proba(data[0]), first.model.predict_proba(data[0]))


def test_other_seed_other_folds(first: TrainingResult, data: tuple[pd.DataFrame, pd.Series]) -> None:
    other = train(data, seed=7)

    assert [result.scores for result in other.candidates] != [result.scores for result in first.candidates]


def test_pickled_model_same_predictions(first: TrainingResult, data: tuple[pd.DataFrame, pd.Series]) -> None:
    restored = pickle.loads(pickle.dumps(first.model))

    np.testing.assert_array_equal(restored.predict_proba(data[0]), first.model.predict_proba(data[0]))
    np.testing.assert_array_equal(restored.predict(data[0]), first.model.predict(data[0]))
    assert restored.best_threshold_ == first.threshold

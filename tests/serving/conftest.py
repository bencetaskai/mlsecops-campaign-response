"""Fixtures shared by the API tests."""

import pandas as pd
import pytest
from sklearn.model_selection import TunedThresholdClassifierCV

from campaign_response.modeling.candidates import get_candidate
from campaign_response.modeling.training import fit_with_tuned_threshold
from campaign_response.pipeline.train_pipeline import split_features


@pytest.fixture(scope="session")
def model(customers: pd.DataFrame) -> TunedThresholdClassifierCV:
    """A real tuned model on the synthetic customers, fitted once per test session."""
    features, target = split_features(customers)
    return fit_with_tuned_threshold(get_candidate("logistic_regression"), {}, features, target, cv_folds=3, seed=42)

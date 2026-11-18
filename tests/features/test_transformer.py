"""Tests for the sklearn wrapper of the feature engineering."""

import pickle
from datetime import date

import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from campaign_response.data.cleaning import clean
from campaign_response.features.engineering import MODEL_FEATURES, add_engineered_features
from campaign_response.features.transformer import FeatureEngineer


@pytest.fixture
def inputs(raw_frame: pd.DataFrame) -> pd.DataFrame:
    """What /predict will receive: cleaned customer columns without ID and target."""
    cleaned, _ = clean(raw_frame)
    return cleaned.drop(columns=["ID", "Response"])


def test_transformer_fit_returns_self(inputs: pd.DataFrame) -> None:
    engineer = FeatureEngineer()

    assert engineer.fit(inputs) is engineer


def test_transformer_output_equals_function(inputs: pd.DataFrame) -> None:
    result = FeatureEngineer().fit_transform(inputs)

    expected = add_engineered_features(inputs)[list(MODEL_FEATURES)]
    pd.testing.assert_frame_equal(result, expected)


def test_transformer_columns_model_features(inputs: pd.DataFrame) -> None:
    engineer = FeatureEngineer().fit(inputs)

    assert tuple(engineer.transform(inputs).columns) == MODEL_FEATURES
    assert tuple(engineer.get_feature_names_out()) == MODEL_FEATURES


def test_transformer_clone_keeps_reference_date() -> None:
    engineer = FeatureEngineer(reference_date=date(2015, 12, 31))

    copy = clone(engineer)

    assert copy is not engineer
    assert copy.get_params() == {"reference_date": date(2015, 12, 31)}


def test_transformer_reference_date_used(inputs: pd.DataFrame) -> None:
    later = FeatureEngineer(reference_date=date(2015, 12, 31)).fit_transform(inputs)
    default = FeatureEngineer().fit_transform(inputs)

    assert (later["age"] - default["age"]).tolist() == [1, 1, 1]


def test_transformer_inside_pipeline(inputs: pd.DataFrame) -> None:
    pipeline = Pipeline([("features", FeatureEngineer())])

    result = pipeline.fit_transform(inputs)

    assert tuple(pipeline.get_feature_names_out()) == MODEL_FEATURES
    assert len(result) == len(inputs)


def test_transformer_pickle_round_trip(inputs: pd.DataFrame) -> None:
    engineer = FeatureEngineer().fit(inputs)

    restored = pickle.loads(pickle.dumps(engineer))

    pd.testing.assert_frame_equal(restored.transform(inputs), engineer.transform(inputs))

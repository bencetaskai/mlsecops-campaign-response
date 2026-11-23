"""Tests for the imputation, scaling and encoding step."""

import numpy as np
import pandas as pd
import pytest

from campaign_response.data.cleaning import clean
from campaign_response.features.engineering import MODEL_FEATURES
from campaign_response.features.preprocessing import build_preprocessor
from campaign_response.features.transformer import FeatureEngineer

CATEGORICAL = ("Education", "Marital_Status")


@pytest.fixture
def features(raw_frame: pd.DataFrame) -> pd.DataFrame:
    """The three fixture customers ten times each, so every category reaches the minimum count of 10."""
    cleaned, _ = clean(raw_frame)
    repeated = pd.concat([cleaned] * 10, ignore_index=True)
    return FeatureEngineer().fit_transform(repeated.drop(columns=["ID", "Response"]))


def test_preprocessor_feature_names_fixed(features: pd.DataFrame) -> None:
    preprocessor = build_preprocessor().fit(features)

    numeric = [column for column in MODEL_FEATURES if column not in CATEGORICAL]
    encoded = ["Education_2n Cycle", "Education_Graduation", "Education_PhD",
               "Marital_Status_Single", "Marital_Status_Together"]
    assert preprocessor.get_feature_names_out().tolist() == numeric + encoded


def test_preprocessor_missing_values_no_nan(features: pd.DataFrame) -> None:
    preprocessor = build_preprocessor().fit(features)
    gaps = features.head(3).assign(Income=np.nan, store_share=np.nan, Marital_Status=np.nan)

    assert not np.isnan(preprocessor.transform(gaps)).any()


def test_preprocessor_imputes_training_median(features: pd.DataFrame) -> None:
    preprocessor = build_preprocessor().fit(features)
    row = features.head(1)

    missing = preprocessor.transform(row.assign(Income=np.nan))
    median = preprocessor.transform(row.assign(Income=features["Income"].median()))

    np.testing.assert_array_equal(missing, median)


def test_preprocessor_numeric_scaled(features: pd.DataFrame) -> None:
    preprocessor = build_preprocessor().fit(features)
    names = preprocessor.get_feature_names_out().tolist()

    scaled = preprocessor.transform(features)[:, names.index("total_spend")]

    assert scaled.mean() == pytest.approx(0)
    assert scaled.std() == pytest.approx(1)


def test_preprocessor_unknown_category_all_zero(features: pd.DataFrame) -> None:
    preprocessor = build_preprocessor().fit(features)
    names = preprocessor.get_feature_names_out().tolist()

    encoded = preprocessor.transform(features.head(1).assign(Education="Doctorate"))

    education = [names.index(name) for name in names if name.startswith("Education_")]
    assert encoded[0, education].tolist() == [0, 0, 0]


def test_preprocessor_rare_category_grouped(features: pd.DataFrame) -> None:
    preprocessor = build_preprocessor(min_category_count=11).fit(features)

    names = preprocessor.get_feature_names_out().tolist()

    assert "Education_infrequent_sklearn" in names
    assert "Marital_Status_Single" in names

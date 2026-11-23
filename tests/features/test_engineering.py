"""Tests for the engineered features: each formula checked on hand-computed values."""

from datetime import date

import pandas as pd
import pytest

from campaign_response.data.cleaning import clean
from campaign_response.features.engineering import (
    ENGINEERED_FEATURES,
    MODEL_FEATURES,
    REPLACED_COLUMNS,
    add_engineered_features,
)


@pytest.fixture
def cleaned(raw_frame: pd.DataFrame) -> pd.DataFrame:
    """Cleaned customers 5524 (Single, 1 kid), 2174 (Alone -> Single, no income, no purchases), 4141 (Together,
    2 teens)."""
    frame, _ = clean(raw_frame)
    return frame.assign(
        Kidhome=pd.array([1, 0, 0], dtype="Int64"),
        Teenhome=pd.array([0, 0, 2], dtype="Int64"),
        MntFruits=pd.array([10, 0, 0], dtype="Int64"),
        AcceptedCmp1=pd.array([1, 0, 0], dtype="Int64"),
        AcceptedCmp5=pd.array([1, 0, 1], dtype="Int64"),
        NumWebPurchases=pd.array([8, 0, 8], dtype="Int64"),
        NumCatalogPurchases=pd.array([10, 0, 2], dtype="Int64"),
        NumStorePurchases=pd.array([4, 0, 10], dtype="Int64"),
        NumDealsPurchases=pd.array([3, 0, 1], dtype="Int64"),
    )


def test_features_hand_computed_values(cleaned: pd.DataFrame) -> None:
    result = add_engineered_features(cleaned)

    assert result["age"].tolist() == [57, 60, 49]
    assert result["tenure_days"].tolist() == [848, 298, 497]
    assert result["total_spend"].tolist() == [645, 11, 426]
    assert result["total_accepted_cmp"].tolist() == [2, 0, 1]
    assert result["household_size"].tolist() == [2, 1, 4]
    assert result.loc[0, "income_per_member"] == 29069
    assert result.loc[2, "income_per_member"] == pytest.approx(17903.25)
    assert result.loc[0, "store_share"] == pytest.approx(4 / 22)
    assert result.loc[2, "store_share"] == 0.5


def test_features_no_purchases_store_share_missing(cleaned: pd.DataFrame) -> None:
    result = add_engineered_features(cleaned)

    assert pd.isna(result.loc[1, "store_share"])
    assert str(result["store_share"].dtype) == "Float64"


def test_features_missing_income_propagates(cleaned: pd.DataFrame) -> None:
    result = add_engineered_features(cleaned)

    assert pd.isna(result.loc[1, "income_per_member"])
    assert result.loc[1, "household_size"] == 1


def test_features_missing_marital_household_missing(cleaned: pd.DataFrame) -> None:
    result = add_engineered_features(cleaned.assign(Marital_Status=pd.array([pd.NA, "Single", "Married"],
                                                                            dtype="string")))

    assert pd.isna(result.loc[0, "household_size"])
    assert pd.isna(result.loc[0, "income_per_member"])
    assert result.loc[2, "household_size"] == 4


def test_features_missing_birth_year_age_missing(cleaned: pd.DataFrame) -> None:
    result = add_engineered_features(cleaned.assign(Year_Birth=pd.array([pd.NA, 1954, 1965], dtype="Int64")))

    assert pd.isna(result.loc[0, "age"])


def test_features_custom_reference_date(cleaned: pd.DataFrame) -> None:
    result = add_engineered_features(cleaned, reference_date=date(2015, 12, 31))

    assert result.loc[0, "age"] == 58
    assert result.loc[0, "tenure_days"] == 848 + 365


def test_features_replace_raw_columns(cleaned: pd.DataFrame) -> None:
    result = add_engineered_features(cleaned)

    assert set(REPLACED_COLUMNS).isdisjoint(result.columns)
    assert set(MODEL_FEATURES) <= set(result.columns)
    assert {"ID", "Response"} <= set(result.columns)


def test_model_features_exclude_id_target() -> None:
    assert {"ID", "Response", *REPLACED_COLUMNS}.isdisjoint(MODEL_FEATURES)
    assert set(ENGINEERED_FEATURES) <= set(MODEL_FEATURES)


def test_features_input_unchanged(cleaned: pd.DataFrame) -> None:
    before = cleaned.copy()

    add_engineered_features(cleaned)

    pd.testing.assert_frame_equal(cleaned, before)

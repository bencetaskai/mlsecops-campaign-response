"""Consistency tests for the raw data contract."""

import pytest

from campaign_response.data import schema

KAGGLE_HEADER = (
    "ID\tYear_Birth\tEducation\tMarital_Status\tIncome\tKidhome\tTeenhome\tDt_Customer\tRecency\tMntWines\t"
    "MntFruits\tMntMeatProducts\tMntFishProducts\tMntSweetProducts\tMntGoldProds\tNumDealsPurchases\t"
    "NumWebPurchases\tNumCatalogPurchases\tNumStorePurchases\tNumWebVisitsMonth\tAcceptedCmp3\tAcceptedCmp4\t"
    "AcceptedCmp5\tAcceptedCmp1\tAcceptedCmp2\tComplain\tZ_CostContact\tZ_Revenue\tResponse"
)


def test_raw_columns_match_kaggle_header() -> None:
    assert schema.RAW_COLUMNS == tuple(KAGGLE_HEADER.split("\t"))


def test_raw_columns_unique() -> None:
    assert len(set(schema.RAW_COLUMNS)) == len(schema.RAW_COLUMNS)


def test_special_columns_in_raw_columns() -> None:
    special = {schema.ID_COLUMN, schema.TARGET_COLUMN, schema.DATE_COLUMN, *schema.CONSTANT_COLUMNS}

    assert special <= set(schema.RAW_COLUMNS)


def test_ranges_only_numeric_columns() -> None:
    for column in schema.VALUE_RANGES:
        assert schema.RAW_DTYPES[column] in {"Int64", "Float64"}, column


def test_ranges_min_below_max() -> None:
    for column, (low, high) in schema.VALUE_RANGES.items():
        assert low < high, column


def test_target_is_binary() -> None:
    assert schema.VALUE_RANGES[schema.TARGET_COLUMN] == (0, 1)


def test_features_exclude_id_target_constants() -> None:
    excluded = {schema.ID_COLUMN, schema.TARGET_COLUMN, *schema.CONSTANT_COLUMNS}

    assert excluded.isdisjoint(schema.FEATURE_COLUMNS)
    assert len(schema.FEATURE_COLUMNS) == len(schema.RAW_COLUMNS) - len(excluded)


def test_categories_cover_observed_values() -> None:
    assert schema.EDUCATION_LEVELS == {"Basic", "Graduation", "2n Cycle", "Master", "PhD"}
    assert {"Alone", "Absurd", "YOLO"} <= schema.MARITAL_STATUSES


def test_raw_dtypes_read_only() -> None:
    with pytest.raises(TypeError):
        schema.RAW_DTYPES["ID"] = "string"

"""Tests for the cleaning rules: each rule changes only what it is responsible for."""

import logging
from datetime import date

import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from campaign_response.data.cleaning import (
    CleaningParams,
    birth_year_bounds,
    drop_duplicate_customers,
    drop_unused_columns,
    mask_implausible_birth_year,
    mask_implausible_income,
    normalise_marital_status,
)
from campaign_response.data.schema import CONSTANT_COLUMNS, ID_COLUMN, RAW_COLUMNS


def with_copy_of_first_row(frame: pd.DataFrame, new_id: int) -> pd.DataFrame:
    """Append a copy of row 0 under another ID: the same customer registered twice."""
    copy = frame.iloc[[0]].assign(ID=pd.array([new_id], dtype="Int64"))
    return pd.concat([frame, copy], ignore_index=True)


def test_drop_unused_removes_id_and_constants(raw_frame: pd.DataFrame) -> None:
    cleaned = drop_unused_columns(raw_frame)

    assert {ID_COLUMN, *CONSTANT_COLUMNS}.isdisjoint(cleaned.columns)
    assert len(cleaned.columns) == len(RAW_COLUMNS) - 3


def test_drop_unused_keeps_other_columns(raw_frame: pd.DataFrame) -> None:
    cleaned = drop_unused_columns(raw_frame)

    pd.testing.assert_frame_equal(cleaned, raw_frame[list(cleaned.columns)])


def test_drop_unused_input_unchanged(raw_frame: pd.DataFrame) -> None:
    before = raw_frame.copy()

    drop_unused_columns(raw_frame)

    pd.testing.assert_frame_equal(raw_frame, before)


def test_drop_unused_missing_column_KeyError(raw_frame: pd.DataFrame) -> None:
    with pytest.raises(KeyError, match="Z_Revenue"):
        drop_unused_columns(raw_frame.drop(columns=["Z_Revenue"]))


def test_drop_duplicates_other_id_removed(raw_frame: pd.DataFrame) -> None:
    cleaned, removed = drop_duplicate_customers(with_copy_of_first_row(raw_frame, new_id=9999))

    assert removed == 1
    assert list(cleaned["ID"]) == [5524, 2174, 4141]


def test_drop_duplicates_none_unchanged(raw_frame: pd.DataFrame) -> None:
    cleaned, removed = drop_duplicate_customers(raw_frame)

    assert removed == 0
    pd.testing.assert_frame_equal(cleaned, raw_frame)


def test_drop_duplicates_one_difference_kept(raw_frame: pd.DataFrame) -> None:
    frame = with_copy_of_first_row(raw_frame, new_id=9999)
    frame.loc[3, "Recency"] = 1

    _, removed = drop_duplicate_customers(frame)

    assert removed == 0


def test_drop_duplicates_without_id_column(raw_frame: pd.DataFrame) -> None:
    frame = with_copy_of_first_row(raw_frame, new_id=9999).drop(columns=["ID"])

    _, removed = drop_duplicate_customers(frame)

    assert removed == 1


def test_drop_duplicates_keeps_original_index(raw_frame: pd.DataFrame) -> None:
    frame = pd.concat([with_copy_of_first_row(raw_frame, new_id=9999).iloc[[3]], raw_frame], ignore_index=True)

    cleaned, _ = drop_duplicate_customers(frame)

    assert list(cleaned.index) == [0, 2, 3]


def test_drop_duplicates_input_unchanged(raw_frame: pd.DataFrame) -> None:
    frame = with_copy_of_first_row(raw_frame, new_id=9999)
    before = frame.copy()

    drop_duplicate_customers(frame)

    pd.testing.assert_frame_equal(frame, before)


def test_drop_duplicates_logs_removed(raw_frame: pd.DataFrame, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="campaign_response.data.cleaning"):
        drop_duplicate_customers(with_copy_of_first_row(raw_frame, new_id=9999))

    assert caplog.records[-1].__dict__["removed"] == 1


def birth_years(*years: int | None) -> pd.DataFrame:
    return pd.DataFrame({"Year_Birth": pd.array(list(years), dtype="Int64")})


def test_birth_year_bounds_default_1914_1996() -> None:
    assert birth_year_bounds(CleaningParams()) == (1914, 1996)


@pytest.mark.parametrize(
    ("year", "kept"),
    [(1893, False), (1913, False), (1914, True), (1957, True), (1996, True), (1997, False), (2090, False)],
)
def test_mask_birth_year_bounds_inclusive(year: int, kept: bool) -> None:
    result = mask_implausible_birth_year(birth_years(year))

    assert (result.loc[0, "Year_Birth"] == year) if kept else pd.isna(result.loc[0, "Year_Birth"])


def test_mask_birth_year_missing_stays_missing() -> None:
    result = mask_implausible_birth_year(birth_years(None, 1957))

    assert pd.isna(result.loc[0, "Year_Birth"])
    assert result.loc[1, "Year_Birth"] == 1957


def test_mask_birth_year_custom_params() -> None:
    params = CleaningParams(reference_date=date(2020, 1, 1), min_age=30, max_age=60)

    result = mask_implausible_birth_year(birth_years(1959, 1960, 1990, 1991), params)

    assert result["Year_Birth"].isna().tolist() == [True, False, False, True]


def test_mask_birth_year_other_columns_and_input_unchanged(raw_frame: pd.DataFrame) -> None:
    frame = raw_frame.assign(Year_Birth=pd.array([1893, 1957, 1900], dtype="Int64"))
    before = frame.copy()

    result = mask_implausible_birth_year(frame)

    pd.testing.assert_frame_equal(frame, before)
    pd.testing.assert_frame_equal(result.drop(columns="Year_Birth"), frame.drop(columns="Year_Birth"))
    assert result["Year_Birth"].dtype == "Int64"


def test_mask_birth_year_logs_masked(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="campaign_response.data.cleaning"):
        mask_implausible_birth_year(birth_years(1893, 1899, 1957))

    assert caplog.records[-1].__dict__["masked"] == 2


@given(st.lists(st.one_of(st.none(), st.integers(min_value=1700, max_value=2300)), min_size=1, max_size=40))
def test_mask_birth_year_property_in_bounds_or_missing(years: list[int | None]) -> None:
    low, high = birth_year_bounds(CleaningParams())

    result = mask_implausible_birth_year(birth_years(*years))["Year_Birth"]

    for before, after in zip(years, result, strict=True):
        if before is not None and low <= before <= high:
            assert after == before
        else:
            assert pd.isna(after)


def incomes(*values: float | None) -> pd.DataFrame:
    return pd.DataFrame({"Income": pd.array(list(values), dtype="Float64")})


@pytest.mark.parametrize(
    ("income", "kept"),
    [(1730.0, True), (162397.0, True), (200000.0, True), (200000.01, False), (666666.0, False)],
)
def test_mask_income_threshold_inclusive(income: float, kept: bool) -> None:
    result = mask_implausible_income(incomes(income))

    assert (result.loc[0, "Income"] == income) if kept else pd.isna(result.loc[0, "Income"])


def test_mask_income_missing_stays_missing() -> None:
    result = mask_implausible_income(incomes(None, 666666.0, 58138.0))

    assert result["Income"].isna().tolist() == [True, True, False]


def test_mask_income_custom_threshold() -> None:
    result = mask_implausible_income(incomes(150000.0, 160000.0), CleaningParams(max_income=155000))

    assert result["Income"].isna().tolist() == [False, True]


def test_mask_income_other_columns_and_input_unchanged(raw_frame: pd.DataFrame) -> None:
    frame = raw_frame.assign(Income=pd.array([666666.0, None, 71613.0], dtype="Float64"))
    before = frame.copy()

    result = mask_implausible_income(frame)

    pd.testing.assert_frame_equal(frame, before)
    pd.testing.assert_frame_equal(result.drop(columns="Income"), frame.drop(columns="Income"))
    assert result["Income"].dtype == "Float64"


def test_mask_income_logs_masked(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="campaign_response.data.cleaning"):
        mask_implausible_income(incomes(666666.0, 58138.0))

    assert caplog.records[-1].__dict__["masked"] == 1


@given(st.lists(st.one_of(st.none(), st.floats(min_value=0, max_value=1e7)), min_size=1, max_size=40))
def test_mask_income_property_kept_or_missing(values: list[float | None]) -> None:
    result = mask_implausible_income(incomes(*values))["Income"]

    for before, after in zip(values, result, strict=True):
        if before is not None and before <= CleaningParams().max_income:
            assert after == before
        else:
            assert pd.isna(after)


def statuses(*values: str | None) -> pd.DataFrame:
    return pd.DataFrame({"Marital_Status": pd.array(list(values), dtype="string")})


def test_marital_alone_becomes_single() -> None:
    result = normalise_marital_status(statuses("Alone", "Single"))

    assert result["Marital_Status"].tolist() == ["Single", "Single"]


@pytest.mark.parametrize("junk", ["Absurd", "YOLO"])
def test_marital_junk_becomes_missing(junk: str) -> None:
    result = normalise_marital_status(statuses(junk, "Married"))

    assert pd.isna(result.loc[0, "Marital_Status"])
    assert result.loc[1, "Marital_Status"] == "Married"


def test_marital_known_and_unseen_unchanged() -> None:
    values = ("Married", "Together", "Single", "Divorced", "Widow", "Engaged", None)

    result = normalise_marital_status(statuses(*values))

    pd.testing.assert_frame_equal(result, statuses(*values))


def test_marital_other_columns_and_input_unchanged(raw_frame: pd.DataFrame) -> None:
    before = raw_frame.copy()

    result = normalise_marital_status(raw_frame)

    pd.testing.assert_frame_equal(raw_frame, before)
    pd.testing.assert_frame_equal(result.drop(columns="Marital_Status"), raw_frame.drop(columns="Marital_Status"))
    assert result["Marital_Status"].dtype == "string"


def test_marital_logs_mapped_and_masked(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="campaign_response.data.cleaning"):
        normalise_marital_status(statuses("Alone", "YOLO", "Absurd", "Single"))

    assert caplog.records[-1].__dict__["mapped"] == 1
    assert caplog.records[-1].__dict__["masked"] == 2

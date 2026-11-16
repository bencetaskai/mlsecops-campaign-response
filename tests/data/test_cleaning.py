"""Tests for the cleaning rules: each rule changes only what it is responsible for."""

import logging

import pandas as pd
import pytest

from campaign_response.data.cleaning import drop_duplicate_customers, drop_unused_columns
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

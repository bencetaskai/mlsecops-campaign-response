"""Tests for the cleaning rules: each rule changes only what it is responsible for."""

import pandas as pd
import pytest

from campaign_response.data.cleaning import drop_unused_columns
from campaign_response.data.schema import CONSTANT_COLUMNS, ID_COLUMN, RAW_COLUMNS


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

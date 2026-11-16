"""Tests for the raw data validation gate: one broken aspect per test."""

import json
import logging

import pandas as pd
import pytest

from campaign_response.data import schema
from campaign_response.data.validation import Thresholds, ValidationReport, validate_raw

ROWS = 120


def valid_frame(rows: int = ROWS) -> pd.DataFrame:
    values: dict[str, list[object]] = {column: [0] * rows for column in schema.RAW_COLUMNS}
    values.update({
        "ID": list(range(rows)),
        "Year_Birth": [1970] * rows,
        "Education": ["Graduation"] * rows,
        "Marital_Status": ["Married"] * rows,
        "Income": [50000.0] * rows,
        "Dt_Customer": ["04-09-2012"] * rows,
        "Z_CostContact": [3] * rows,
        "Z_Revenue": [11] * rows,
        "Response": [index % 2 for index in range(rows)],
    })
    return pd.DataFrame(values).astype(dict(schema.RAW_DTYPES))


def failed_checks(report: ValidationReport) -> set[str]:
    return {check.name for check in report.failures}


def test_validate_valid_frame_passes() -> None:
    report = validate_raw(valid_frame())

    assert report.passed
    assert report.rows == ROWS


@pytest.mark.parametrize("column", ["Income", "Education", "Year_Birth"])
def test_validate_missing_column_fails(column: str) -> None:
    report = validate_raw(valid_frame().drop(columns=[column]))

    assert failed_checks(report) == {"columns"}


def test_validate_unexpected_column_fails() -> None:
    frame = valid_frame().assign(Extra=1)

    assert failed_checks(validate_raw(frame)) == {"columns"}


@pytest.mark.parametrize("column", [schema.DATE_COLUMN, schema.ID_COLUMN, schema.TARGET_COLUMN])
def test_validate_key_column_missing_fails(column: str) -> None:
    report = validate_raw(valid_frame().drop(columns=[column]))

    assert "columns" in failed_checks(report)
    assert len(report.failures) == 2


def test_validate_wrong_dtype_fails() -> None:
    frame = valid_frame().astype({"Recency": "float64"})

    assert failed_checks(validate_raw(frame)) == {"dtypes"}


def test_validate_too_few_rows_fails() -> None:
    assert failed_checks(validate_raw(valid_frame(rows=10))) == {"row_count"}


def test_validate_empty_frame_fails() -> None:
    report = validate_raw(valid_frame(rows=0))

    assert failed_checks(report) == {"row_count", "target"}


def test_validate_missing_share_fails() -> None:
    frame = valid_frame()
    frame.loc[:11, "Income"] = pd.NA

    assert failed_checks(validate_raw(frame)) == {"missing_values"}


def test_validate_out_of_range_share_fails() -> None:
    frame = valid_frame()
    frame.loc[:4, "Year_Birth"] = 2090

    assert failed_checks(validate_raw(frame)) == {"value_ranges"}


def test_validate_single_outlier_tolerated() -> None:
    frame = valid_frame()
    frame.loc[0, "Year_Birth"] = 2090

    assert validate_raw(frame).passed


def test_validate_unknown_category_fails() -> None:
    frame = valid_frame()
    frame.loc[:4, "Education"] = "Bootcamp"

    assert failed_checks(validate_raw(frame)) == {"categories"}


def test_validate_bad_date_fails() -> None:
    frame = valid_frame()
    frame.loc[0, "Dt_Customer"] = "2012-09-04"

    assert failed_checks(validate_raw(frame)) == {"dates"}


def test_validate_duplicate_id_fails() -> None:
    frame = valid_frame()
    frame.loc[1, "ID"] = 0

    assert failed_checks(validate_raw(frame)) == {"ids_unique"}


def test_validate_single_class_target_fails() -> None:
    frame = valid_frame()
    frame["Response"] = pd.array([0] * ROWS, dtype="Int64")

    assert failed_checks(validate_raw(frame)) == {"target"}


def test_validate_custom_thresholds_apply() -> None:
    frame = valid_frame()
    frame.loc[:11, "Income"] = pd.NA

    assert validate_raw(frame, Thresholds(max_missing_share=0.2)).passed


def test_report_to_json_fields() -> None:
    frame = valid_frame()
    frame.loc[1, "ID"] = 0

    payload = json.loads(validate_raw(frame).to_json())

    assert payload["passed"] is False
    assert payload["rows"] == ROWS
    assert {"name": "ids_unique", "passed": False, "detail": "duplicate ids=1"} in payload["checks"]


def test_validate_logs_failed_checks(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="campaign_response.data.validation"):
        validate_raw(valid_frame(rows=10))

    assert caplog.records[-1].__dict__["failed"] == ["row_count"]

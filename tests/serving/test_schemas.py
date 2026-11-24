"""Tests for the API request and response models, validated in Python mode as FastAPI does with a decoded body."""

from datetime import date
from pathlib import Path
from typing import Any, get_args

import pytest
from pydantic import ValidationError

from campaign_response.data import schema
from campaign_response.params import load_params
from campaign_response.serving.schemas import (
    LATEST_ENROLLMENT,
    MAX_RECORDS,
    CustomerRecord,
    EducationLevel,
    Prediction,
    PredictRequest,
    PredictResponse,
)

REPO = Path(__file__).resolve().parents[2]
RECORD: dict[str, Any] = {
    "Year_Birth": 1970, "Education": "Graduation", "Marital_Status": "Married", "Income": 58138.0,
    "Kidhome": 0, "Teenhome": 1, "Dt_Customer": "2012-09-04", "Recency": 58,
    "MntWines": 635, "MntFruits": 88, "MntMeatProducts": 546, "MntFishProducts": 172,
    "MntSweetProducts": 88, "MntGoldProds": 88, "NumDealsPurchases": 3, "NumWebPurchases": 8,
    "NumCatalogPurchases": 10, "NumStorePurchases": 4, "NumWebVisitsMonth": 7,
    "AcceptedCmp1": 0, "AcceptedCmp2": 0, "AcceptedCmp3": 0, "AcceptedCmp4": 0, "AcceptedCmp5": 1,
    "Complain": 0,
}


def record(**changes: object) -> dict[str, Any]:
    return {**RECORD, **changes}


def test_record_valid_example_accepted() -> None:
    parsed = CustomerRecord.model_validate(RECORD)

    assert parsed.Dt_Customer == date(2012, 9, 4)
    assert parsed.ID is None
    assert parsed.AcceptedCmp5 == 1


def test_record_fields_match_feature_columns() -> None:
    assert tuple(CustomerRecord.model_fields) == ("ID", *schema.FEATURE_COLUMNS)


def test_education_levels_match_schema() -> None:
    assert set(get_args(EducationLevel)) == schema.EDUCATION_LEVELS


def test_latest_enrollment_matches_params() -> None:
    assert LATEST_ENROLLMENT == load_params(REPO / "params.yaml").cleaning.reference_date


@pytest.mark.parametrize("changes", [
    {"ID": 11191},
    {"Year_Birth": 1893},
    {"Income": None},
    {"Income": 666666.0},
    {"Income": 50000},
    {"Marital_Status": "YOLO"},
    {"Marital_Status": "Polyamorous"},
    {"Dt_Customer": date(2014, 12, 31)},
    {"MntWines": 5000, "NumWebVisitsMonth": 60},
], ids=["id", "old-birth-year", "income-missing", "income-placeholder", "income-int", "junk-status", "new-status",
        "date-object", "beyond-training-range"])
def test_record_unusual_but_possible_accepted(changes: dict[str, object]) -> None:
    CustomerRecord.model_validate(record(**changes))


@pytest.mark.parametrize("changes", [
    {"ID": 0},
    {"ID": 2**31},
    {"Year_Birth": 1889},
    {"Year_Birth": 2011},
    {"Education": "Doctorate"},
    {"Education": "phd"},
    {"Marital_Status": ""},
    {"Marital_Status": "A" * 21},
    {"Marital_Status": "Not Married"},
    {"Income": -1.0},
    {"Income": 10_000_001.0},
    {"Income": float("inf")},
    {"Income": float("nan")},
    {"Kidhome": -1},
    {"Teenhome": 11},
    {"Recency": 3651},
    {"MntMeatProducts": 100_001},
    {"MntGoldProds": -5},
    {"NumStorePurchases": 1001},
    {"NumWebVisitsMonth": 10**30},
    {"AcceptedCmp1": 2},
    {"Complain": -1},
    {"Dt_Customer": "1999-12-31"},
    {"Dt_Customer": "2015-01-01"},
], ids=lambda changes: next(iter(changes)))
def test_record_out_of_range_rejected(changes: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        CustomerRecord.model_validate(record(**changes))


@pytest.mark.parametrize("changes", [
    {"Year_Birth": "1970"},
    {"Kidhome": 1.0},
    {"Recency": True},
    {"AcceptedCmp2": True},
    {"Complain": "0"},
    {"Income": "58138"},
    {"Income": False},
    {"Education": 1},
], ids=["int-as-text", "int-as-float", "bool-count", "bool-flag", "flag-as-text", "income-as-text",
        "income-bool", "education-number"])
def test_record_wrong_type_rejected(changes: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        CustomerRecord.model_validate(record(**changes))


@pytest.mark.parametrize("value", ["04-09-2012", "2012/09/04", "2012-09-04T00:00:00", "2012-02-30", 1346716800, ""])
def test_record_enrollment_not_iso_rejected(value: object) -> None:
    with pytest.raises(ValidationError):
        CustomerRecord.model_validate(record(Dt_Customer=value))


@pytest.mark.parametrize("field", ["Year_Birth", "Marital_Status", "Dt_Customer", "MntWines", "Complain"])
def test_record_missing_or_null_field_rejected(field: str) -> None:
    without = {key: value for key, value in RECORD.items() if key != field}

    with pytest.raises(ValidationError):
        CustomerRecord.model_validate(without)
    with pytest.raises(ValidationError):
        CustomerRecord.model_validate(record(**{field: None}))


def test_record_income_required_but_nullable() -> None:
    without = {key: value for key, value in RECORD.items() if key != "Income"}

    with pytest.raises(ValidationError, match="Income"):
        CustomerRecord.model_validate(without)


def test_record_extra_field_rejected() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        CustomerRecord.model_validate(record(Response=1))


@pytest.mark.parametrize("changes", [
    {"Marital_Status": "Single'; DROP TABLE prediction_log;--"},
    {"Marital_Status": "<script>alert(1)</script>"},
    {"Education": "PhD' OR '1'='1"},
    {"Dt_Customer": "2012-09-04'; --"},
], ids=["sql-status", "html-status", "sql-education", "sql-date"])
def test_record_injection_string_rejected(changes: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        CustomerRecord.model_validate(record(**changes))


@pytest.mark.parametrize("count", [1, MAX_RECORDS])
def test_predict_request_size_accepted(count: int) -> None:
    request = PredictRequest.model_validate({"records": [RECORD] * count})

    assert len(request.records) == count


@pytest.mark.parametrize("count", [0, MAX_RECORDS + 1])
def test_predict_request_size_rejected(count: int) -> None:
    with pytest.raises(ValidationError):
        PredictRequest.model_validate({"records": [RECORD] * count})


def test_predict_request_extra_field_rejected() -> None:
    with pytest.raises(ValidationError):
        PredictRequest.model_validate({"records": [RECORD], "model_version": 1})


def test_predict_response_round_trip() -> None:
    response = PredictResponse(
        request_id="9f1c", model_name="campaign-response-classifier", model_version=7, model_alias="champion",
        threshold=0.31, predictions=[Prediction(ID=5524, probability=0.74, label=1)],
    )

    assert PredictResponse.model_validate_json(response.model_dump_json()) == response


@pytest.mark.parametrize("changes", [{"probability": 1.2}, {"label": 2}])
def test_prediction_invalid_value_rejected(changes: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        Prediction.model_validate({"ID": None, "probability": 0.5, "label": 0, **changes})

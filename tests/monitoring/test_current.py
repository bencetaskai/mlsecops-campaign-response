"""Tests for loading the current window from a real SQLite prediction log."""

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import create_engine

from campaign_response.data.cleaning import CleaningParams, clean
from campaign_response.data.io import load_processed, write_processed
from campaign_response.data.schema import FEATURE_COLUMNS
from campaign_response.monitoring.current import load_current
from campaign_response.serving.schemas import CustomerRecord
from campaign_response.smoke import records_from_raw
from campaign_response.storage.prediction_store import PredictionRecord, SqlPredictionStore

CLEANING = CleaningParams()


@pytest.fixture
def store(tmp_path: Path) -> Iterator[SqlPredictionStore]:
    engine = create_engine(f"sqlite:///{(tmp_path / 'monitoring.db').as_posix()}")
    store = SqlPredictionStore(engine)
    store.create_schema()
    yield store
    engine.dispose()


def log(store: SqlPredictionStore, records: list[CustomerRecord], model_version: int = 1) -> None:
    store.save([
        PredictionRecord(
            request_id=f"req-{index}", created_at=datetime(2026, 11, 25, tzinfo=UTC), model_version=model_version,
            source="test", customer_id=record.ID, features=record.model_dump(include=set(FEATURE_COLUMNS)),
            probability=0.5, label=0,
        )
        for index, record in enumerate(records)
    ])


def test_current_matches_training_cleaning(store: SqlPredictionStore, raw_customers: pd.DataFrame) -> None:
    """Logged requests must come back exactly as training's clean() leaves the same customers."""
    log(store, records_from_raw(raw_customers))
    cleaned, _ = clean(raw_customers, CLEANING)

    current = load_current(store, 1000, CLEANING)

    expected = cleaned[list(FEATURE_COLUMNS)].reset_index(drop=True)
    pd.testing.assert_frame_equal(current.loc[cleaned.index].reset_index(drop=True), expected)


def test_current_dtypes_match_reference(store: SqlPredictionStore, raw_frame: pd.DataFrame, tmp_path: Path) -> None:
    log(store, records_from_raw(raw_frame))
    write_processed(clean(raw_frame, CLEANING)[0], tmp_path / "reference.csv")
    reference = load_processed(tmp_path / "reference.csv")[list(FEATURE_COLUMNS)]

    current = load_current(store, 10, CLEANING)

    assert current.dtypes.to_dict() == reference.dtypes.to_dict()


def test_current_outliers_masked(store: SqlPredictionStore, raw_frame: pd.DataFrame) -> None:
    record = records_from_raw(raw_frame)[0]
    log(store, [record.model_copy(update={"Year_Birth": 1893, "Income": 666666.0, "Marital_Status": "YOLO"})])

    [row] = load_current(store, 10, CLEANING).to_dict("records")

    assert pd.isna(row["Year_Birth"]) and pd.isna(row["Income"]) and pd.isna(row["Marital_Status"])


def test_current_newest_window_and_version(store: SqlPredictionStore, raw_customers: pd.DataFrame) -> None:
    records = records_from_raw(raw_customers.head(6))
    log(store, records[:3], model_version=1)
    log(store, records[3:], model_version=2)

    newest = load_current(store, 2, CLEANING)
    second_model = load_current(store, 10, CLEANING, model_version=2)

    assert newest["MntWines"].tolist() == [record.MntWines for record in records[4:]]
    assert second_model["MntWines"].tolist() == [record.MntWines for record in records[3:]]


def test_current_empty_log_empty_frame(store: SqlPredictionStore) -> None:
    current = load_current(store, 200, CLEANING)

    assert current.empty
    assert tuple(current.columns) == FEATURE_COLUMNS

"""The prediction store contract, run against the in-memory fake and the SQL store on SQLite, plus SQL details."""

from collections.abc import Iterator
from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import pytest
from fakes import FakePredictionStore
from sqlalchemy import Engine, create_engine, inspect, text
from sqlalchemy.exc import IntegrityError

from campaign_response.data.schema import FEATURE_COLUMNS
from campaign_response.storage.prediction_store import (
    COLUMNS,
    MAX_WINDOW,
    TABLE_NAME,
    PredictionRecord,
    PredictionStore,
    SqlPredictionStore,
)

FEATURES: dict[str, object] = {
    "Year_Birth": 1970, "Education": "Graduation", "Marital_Status": "Married", "Income": 58138.0,
    "Kidhome": 0, "Teenhome": 1, "Dt_Customer": date(2012, 9, 4), "Recency": 58,
    "MntWines": 635, "MntFruits": 88, "MntMeatProducts": 546, "MntFishProducts": 172,
    "MntSweetProducts": 88, "MntGoldProds": 88, "NumDealsPurchases": 3, "NumWebPurchases": 8,
    "NumCatalogPurchases": 10, "NumStorePurchases": 4, "NumWebVisitsMonth": 7,
    "AcceptedCmp1": 0, "AcceptedCmp2": 0, "AcceptedCmp3": 0, "AcceptedCmp4": 0, "AcceptedCmp5": 1, "Complain": 0,
}
CREATED = datetime(2026, 11, 25, 12, 0, tzinfo=UTC)


def record(request_id: str = "req-1", model_version: int = 1, **features: object) -> PredictionRecord:
    return PredictionRecord(
        request_id=request_id, created_at=CREATED, model_version=model_version, source="test", customer_id=5524,
        features={**FEATURES, **features}, probability=0.74, label=1,
    )


@pytest.fixture
def engine(tmp_path: Path) -> Iterator[Engine]:
    engine = create_engine(f"sqlite:///{(tmp_path / 'monitoring.db').as_posix()}")
    yield engine
    engine.dispose()


@pytest.fixture
def sql_store(engine: Engine) -> SqlPredictionStore:
    store = SqlPredictionStore(engine)
    store.create_schema()
    return store


@pytest.fixture(params=["fake", "sql"])
def store(request: pytest.FixtureRequest) -> PredictionStore:
    if request.param == "fake":
        return FakePredictionStore()
    return request.getfixturevalue("sql_store")


def test_store_conforms_to_port(store: PredictionStore) -> None:
    assert isinstance(store, PredictionStore)


def test_store_empty_latest_has_columns(store: PredictionStore) -> None:
    frame = store.latest(10)

    assert frame.empty
    assert tuple(frame.columns) == COLUMNS


def test_store_saved_record_read_back(store: PredictionStore) -> None:
    store.save([record(Income=None, Marital_Status="YOLO")])

    [row] = store.latest(10).to_dict("records")

    meta = (row["log_id"], row["request_id"], row["model_version"], row["source"], row["ID"])
    assert meta == (1, "req-1", 1, "test", 5524)
    assert {column: row[column] for column in FEATURE_COLUMNS if column != "Income"} == {
        **{column: value for column, value in FEATURES.items() if column != "Income"}, "Marital_Status": "YOLO",
    }
    assert pd.isna(row["Income"])
    assert (row["probability"], row["label"]) == (0.74, 1)


def test_store_latest_newest_rows_oldest_first(store: PredictionStore) -> None:
    store.save([record(f"req-{number}") for number in range(5)])

    frame = store.latest(3)

    assert frame["request_id"].tolist() == ["req-2", "req-3", "req-4"]
    assert frame.index.tolist() == [0, 1, 2]


def test_store_latest_filters_model_version(store: PredictionStore) -> None:
    store.save([record("old-1", 1), record("new-1", 2), record("old-2", 1), record("new-2", 2)])

    assert store.latest(10, model_version=2)["request_id"].tolist() == ["new-1", "new-2"]


def test_store_empty_save_no_rows(store: PredictionStore) -> None:
    store.save([])

    assert store.latest(10).empty


@pytest.mark.parametrize("limit", [0, -1, MAX_WINDOW + 1])
def test_store_limit_out_of_range_ValueError(store: PredictionStore, limit: int) -> None:
    with pytest.raises(ValueError, match="limit must be between"):
        store.latest(limit)


@pytest.mark.parametrize("features", [
    {column: value for column, value in FEATURES.items() if column != "Recency"},
    {**FEATURES, "Response": 1},
], ids=["missing", "unexpected"])
def test_store_wrong_features_nothing_saved(store: PredictionStore, features: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="features do not match"):
        store.save([record("good"), replace(record("bad"), features=features)])

    assert store.latest(10).empty


def test_sql_failed_insert_rolled_back(sql_store: SqlPredictionStore) -> None:
    broken = replace(record("bad"), label=None)  # type: ignore[arg-type]  # NOT NULL violation on purpose

    with pytest.raises(IntegrityError):
        sql_store.save([record("good"), broken])

    assert sql_store.latest(10).empty


def test_sql_injection_string_stored_literally(sql_store: SqlPredictionStore, engine: Engine) -> None:
    payload = "x'); DROP TABLE prediction_log; --"

    sql_store.save([record(payload)])

    assert sql_store.latest(1)["request_id"].tolist() == [payload]
    assert TABLE_NAME in inspect(engine).get_table_names()


def test_sql_create_schema_twice_keeps_rows(sql_store: SqlPredictionStore) -> None:
    sql_store.save([record()])

    sql_store.create_schema()

    assert len(sql_store.latest(10)) == 1


def test_sql_table_has_created_at_index(sql_store: SqlPredictionStore, engine: Engine) -> None:
    indexes = inspect(engine).get_indexes(TABLE_NAME)

    assert [index["column_names"] for index in indexes] == [["created_at"]]


def test_sql_columns_keep_raw_names(sql_store: SqlPredictionStore, engine: Engine) -> None:
    with engine.connect() as connection:
        names = [column[1] for column in connection.execute(text(f"PRAGMA table_info({TABLE_NAME})"))]

    assert tuple(names) == COLUMNS

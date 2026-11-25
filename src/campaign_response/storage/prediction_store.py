"""The prediction log: one row per scored customer, with the raw features, the model version and the result.

The feature columns keep the raw file's names, so the latest rows compare directly with a champion's reference
sample in the drift report. The table is created on start-up when missing; there are no migrations (known limit).
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Final, Protocol, runtime_checkable

import pandas as pd
from sqlalchemy import (
    Column,
    Date,
    DateTime,
    Engine,
    Float,
    Index,
    Integer,
    MetaData,
    SmallInteger,
    String,
    Table,
    insert,
    select,
)
from sqlalchemy.types import TypeEngine

from campaign_response.data.schema import DATE_COLUMN, FEATURE_COLUMNS, ID_COLUMN, RAW_DTYPES

TABLE_NAME: Final = "prediction_log"
MAX_WINDOW: Final = 10_000
MAX_TEXT_LENGTH: Final = 64
META_COLUMNS: Final = ("log_id", "request_id", "created_at", "model_version", "source", ID_COLUMN)
RESULT_COLUMNS: Final = ("probability", "label")
COLUMNS: Final = (*META_COLUMNS, *FEATURE_COLUMNS, *RESULT_COLUMNS)


@dataclass(frozen=True)
class PredictionRecord:
    """One scored customer; features holds exactly the raw feature columns."""

    request_id: str
    created_at: datetime
    model_version: int
    source: str
    customer_id: int | None
    features: Mapping[str, object]
    probability: float
    label: int


@runtime_checkable
class PredictionStore(Protocol):
    def save(self, records: Sequence[PredictionRecord]) -> None:
        """Store all records of one request, or none of them."""
        ...

    def latest(self, limit: int, model_version: int | None = None) -> pd.DataFrame:
        """The newest rows, oldest first, with COLUMNS; only one model version's rows if given."""
        ...


def check_limit(limit: int) -> None:
    if not 1 <= limit <= MAX_WINDOW:
        raise ValueError(f"limit must be between 1 and {MAX_WINDOW}, got {limit}")


def to_row(record: PredictionRecord) -> dict[str, Any]:
    """The record as a table row without log_id; refuses records whose features are not exactly the raw columns."""
    if set(record.features) != set(FEATURE_COLUMNS):
        missing = sorted(set(FEATURE_COLUMNS) - set(record.features))
        unexpected = sorted(set(record.features) - set(FEATURE_COLUMNS))
        raise ValueError(f"features do not match the raw columns: missing {missing}, unexpected {unexpected}")
    return {
        "request_id": record.request_id,
        "created_at": record.created_at,
        "model_version": record.model_version,
        "source": record.source,
        ID_COLUMN: record.customer_id,
        **record.features,
        "probability": record.probability,
        "label": record.label,
    }


def _feature_type(column: str) -> TypeEngine[Any]:
    if column == DATE_COLUMN:
        return Date()
    return {"Int64": Integer(), "Float64": Float(), "string": String(MAX_TEXT_LENGTH)}[RAW_DTYPES[column]]


def build_table(metadata: MetaData) -> Table:
    return Table(
        TABLE_NAME,
        metadata,
        Column("log_id", Integer, primary_key=True, autoincrement=True),
        Column("request_id", String(MAX_TEXT_LENGTH), nullable=False),
        Column("created_at", DateTime(timezone=True), nullable=False),
        Column("model_version", Integer, nullable=False),
        Column("source", String(MAX_TEXT_LENGTH), nullable=False),
        Column(ID_COLUMN, Integer, nullable=True),
        *(Column(column, _feature_type(column), nullable=True) for column in FEATURE_COLUMNS),
        Column("probability", Float, nullable=False),
        Column("label", SmallInteger, nullable=False),
        Index(f"ix_{TABLE_NAME}_created_at", "created_at"),
    )


class SqlPredictionStore:
    """The PredictionStore port on SQLAlchemy Core: Postgres in compose, SQLite in the tests.

    Core statements bind every value as a parameter, so record contents never become SQL text.
    """

    def __init__(self, engine: Engine) -> None:
        self._engine = engine
        self._table = build_table(MetaData())

    def create_schema(self) -> None:
        """Create the table and index if they are missing; existing data is left alone."""
        self._table.metadata.create_all(self._engine, checkfirst=True)

    def save(self, records: Sequence[PredictionRecord]) -> None:
        if not records:
            return
        rows = [to_row(record) for record in records]
        with self._engine.begin() as connection:
            connection.execute(insert(self._table), rows)

    def latest(self, limit: int, model_version: int | None = None) -> pd.DataFrame:
        check_limit(limit)
        query = select(self._table).order_by(self._table.c.log_id.desc()).limit(limit)
        if model_version is not None:
            query = query.where(self._table.c.model_version == model_version)
        with self._engine.connect() as connection:
            rows = connection.execute(query).mappings().all()
        frame = pd.DataFrame([dict(row) for row in reversed(rows)], columns=list(COLUMNS))
        return frame.reset_index(drop=True)

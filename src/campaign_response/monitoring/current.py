"""The current side of the drift comparison: the latest requests from the prediction log, cleaned like training."""

from collections.abc import Hashable
from datetime import date

import pandas as pd

from campaign_response.data.cleaning import CleaningParams, clean_records
from campaign_response.data.schema import DATE_COLUMN, DATE_FORMAT, FEATURE_COLUMNS, RAW_DTYPES
from campaign_response.storage.prediction_store import PredictionStore


def load_current(
    store: PredictionStore, window: int, cleaning: CleaningParams, model_version: int | None = None
) -> pd.DataFrame:
    """The features of the last `window` logged records, oldest first, in the reference sample's shape.

    The log keeps the values as sent and the reference is cleaned, so the same per-record cleaning runs here;
    otherwise every masked outlier (a birth year of 1893, an income of 666,666) would show up as drift.
    """
    logged = store.latest(window, model_version)[list(FEATURE_COLUMNS)]
    raw = logged.assign(**{DATE_COLUMN: [_file_date(value) for value in logged[DATE_COLUMN]]})
    dtypes: dict[Hashable, str] = {column: RAW_DTYPES[column] for column in FEATURE_COLUMNS}
    return clean_records(raw.astype(dtypes), cleaning)


def _file_date(value: date) -> str:
    """The raw file's DD-MM-YYYY text, so clean_records parses the date with the rule training used."""
    return value.strftime(DATE_FORMAT)

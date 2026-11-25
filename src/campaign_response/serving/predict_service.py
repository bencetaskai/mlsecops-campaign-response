"""Scoring validated records with the cached champion, through the same cleaning rules as training."""

from collections.abc import Hashable, Sequence
from dataclasses import dataclass

import pandas as pd

from campaign_response.data.cleaning import CleaningParams, clean_records
from campaign_response.data.schema import DATE_COLUMN, DATE_FORMAT, FEATURE_COLUMNS, RAW_DTYPES
from campaign_response.registry.ports import ModelVersion
from campaign_response.serving.model_cache import ChampionCache
from campaign_response.serving.schemas import CustomerRecord


@dataclass(frozen=True)
class ScoredRecords:
    """Results in request order, with the model that produced them."""

    model_version: ModelVersion
    threshold: float
    probabilities: list[float]
    labels: list[int]


def records_to_raw_frame(records: Sequence[CustomerRecord]) -> pd.DataFrame:
    """The records as rows of the raw file: its columns, dtypes and DD-MM-YYYY date text.

    Turning the date back into the file's text lets clean_records parse it with the very rule training used,
    instead of a second conversion that could drift apart from it.
    """
    rows = [record.model_dump(include=set(FEATURE_COLUMNS)) for record in records]
    frame = pd.DataFrame(rows, columns=list(FEATURE_COLUMNS))
    frame[DATE_COLUMN] = [record.Dt_Customer.strftime(DATE_FORMAT) for record in records]
    dtypes: dict[Hashable, str] = {column: RAW_DTYPES[column] for column in FEATURE_COLUMNS}
    return frame.astype(dtypes)


class PredictService:
    def __init__(self, cache: ChampionCache, cleaning: CleaningParams) -> None:
        self._cache = cache
        self._cleaning = cleaning

    def predict(self, records: Sequence[CustomerRecord]) -> ScoredRecords:
        """Score the records; raises ModelUnavailableError (from the cache) when no model can be served."""
        champion = self._cache.get()
        features = clean_records(records_to_raw_frame(records), self._cleaning)
        probabilities = champion.model.predict_proba(features)[:, 1]
        labels = champion.model.predict(features)
        return ScoredRecords(
            model_version=champion.version,
            threshold=float(champion.model.best_threshold_),
            probabilities=[float(value) for value in probabilities],
            labels=[int(value) for value in labels],
        )

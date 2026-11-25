"""The Evidently drift report: champion reference against the current window, per column and for the dataset.

All of Evidently stays in this module. The settings come from a calibration on the real data (02 §6): distances at
threshold 0.2 give no false alarms on 200 undrifted rows and catch a realistic five-column shift.
"""

from dataclasses import dataclass
from datetime import date
from typing import Any, Final

import numpy as np
import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

from campaign_response.data.schema import BINARY_COLUMNS, DATE_COLUMN, FEATURE_COLUMNS

DEFAULT_THRESHOLD: Final = 0.2
CATEGORICAL_COLUMNS: Final = ("Education", "Marital_Status")
# Few distinct values: compared as distributions over those values, like the categories.
LOW_CARDINALITY_COLUMNS: Final = ("Kidhome", "Teenhome", *(c for c in BINARY_COLUMNS if c in FEATURE_COLUMNS))
NUMERIC_COLUMNS: Final = tuple(column for column in FEATURE_COLUMNS if column not in CATEGORICAL_COLUMNS)
# Evidently cannot compare dates; days since a fixed day carry the same information (tenure is derived from it).
DATE_ORIGIN: Final = pd.Timestamp(date(2012, 1, 1))


class EmptyWindowError(ValueError):
    """There is nothing to compare: the reference or the current window has no rows."""


@dataclass(frozen=True)
class DriftResult:
    n_reference: int
    n_current: int
    threshold: float
    column_scores: dict[str, float]
    drifted_columns: list[str]
    html: str

    @property
    def share_drifted(self) -> float:
        return len(self.drifted_columns) / len(self.column_scores)


def column_methods() -> dict[str, Any]:
    """The Evidently method per column, named explicitly instead of left to Evidently.

    Evidently's automatic choice depends on the reference size: above 1,000 rows it uses these distances, at or
    below 1,000 hypothesis tests whose value is a p-value, where drift means value BELOW the threshold. A smaller
    training set would silently turn our 0.2 into "p < 0.2". The values are typed Any because Evidently declares
    the parameter as a dict of a wider union, and dict does not accept a narrower value type.
    """
    return {
        column: "jensenshannon" if column in (*CATEGORICAL_COLUMNS, *LOW_CARDINALITY_COLUMNS) else "wasserstein"
        for column in FEATURE_COLUMNS
    }


def to_drift_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """The feature columns as Evidently needs them: plain float64 numbers, the date as a day count.

    With pandas' nullable dtypes, DataFrame.values is an object array, and Evidently's np.corrcoef then fails on
    any window without missing values.
    """
    result = frame[list(FEATURE_COLUMNS)].copy()
    result[DATE_COLUMN] = (result[DATE_COLUMN] - DATE_ORIGIN).dt.days
    numeric = list(NUMERIC_COLUMNS)
    result[numeric] = result[numeric].astype("float64")
    for column in CATEGORICAL_COLUMNS:
        result[column] = result[column].astype(object).where(result[column].notna(), np.nan)
    return result


def build_drift_report(
    reference: pd.DataFrame, current: pd.DataFrame, threshold: float = DEFAULT_THRESHOLD
) -> DriftResult:
    if reference.empty or current.empty:
        raise EmptyWindowError(f"cannot compare {len(reference)} reference rows with {len(current)} current rows")
    definition = DataDefinition(numerical_columns=list(NUMERIC_COLUMNS), categorical_columns=list(CATEGORICAL_COLUMNS))
    report = Report([DataDriftPreset(per_column_method=column_methods(), threshold=threshold)])
    # A column without variation in a window (e.g. no complaint among 200 requests) has a 0/0 correlation in
    # Evidently's correlation charts; numpy warns, Evidently handles the NaN. Only these warnings are silenced.
    with np.errstate(divide="ignore", invalid="ignore"):
        snapshot = report.run(
            Dataset.from_pandas(to_drift_frame(current), data_definition=definition),
            Dataset.from_pandas(to_drift_frame(reference), data_definition=definition),
        )
    scores = _column_scores(snapshot.dict()["metrics"])
    return DriftResult(
        n_reference=len(reference),
        n_current=len(current),
        threshold=threshold,
        column_scores=scores,
        drifted_columns=[column for column, score in scores.items() if score >= threshold],
        html=snapshot.get_html_str(as_iframe=False),
    )


def _column_scores(metrics: list[dict[str, Any]]) -> dict[str, float]:
    """The per-column distance from Evidently's metric list, in FEATURE_COLUMNS order."""
    found = {
        metric["config"]["column"]: float(metric["value"])
        for metric in metrics
        if metric["config"].get("type", "").endswith("ValueDrift")
    }
    return {column: found[column] for column in FEATURE_COLUMNS}

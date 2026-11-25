"""Tests for the Evidently drift report on the synthetic customers: no drift, injected drift, data preparation."""

import pandas as pd
import pytest

from campaign_response.data.schema import BINARY_COLUMNS, DATE_COLUMN, FEATURE_COLUMNS
from campaign_response.monitoring.drift import (
    CATEGORICAL_COLUMNS,
    DEFAULT_THRESHOLD,
    NUMERIC_COLUMNS,
    DriftResult,
    EmptyWindowError,
    build_drift_report,
    column_methods,
    to_drift_frame,
)

SHIFTED = ("Income", "Year_Birth", "MntWines", "Recency")


def shifted(frame: pd.DataFrame) -> pd.DataFrame:
    """A realistic multi-column shift: higher incomes, younger, less wine, longer since the last purchase."""
    result = frame.copy()
    result["Income"] = result["Income"] * 1.5
    result["Year_Birth"] = result["Year_Birth"] + 12
    result["MntWines"] = (result["MntWines"] * 0.4).round().astype("Int64")
    result["Recency"] = (result["Recency"] + 40).clip(upper=99)
    return result


@pytest.fixture(scope="module")
def same(customers: pd.DataFrame) -> DriftResult:
    return build_drift_report(customers, customers)


@pytest.fixture(scope="module")
def drifted(customers: pd.DataFrame) -> DriftResult:
    return build_drift_report(customers, shifted(customers))


def test_drift_identical_data_none(same: DriftResult, customers: pd.DataFrame) -> None:
    assert same.drifted_columns == []
    assert same.share_drifted == 0
    assert max(same.column_scores.values()) == pytest.approx(0)
    assert (same.n_reference, same.n_current, same.threshold) == (len(customers), len(customers), DEFAULT_THRESHOLD)


def test_drift_shifted_columns_found(drifted: DriftResult) -> None:
    assert set(SHIFTED) <= set(drifted.drifted_columns)
    assert drifted.share_drifted == len(drifted.drifted_columns) / len(FEATURE_COLUMNS)


def test_drift_scores_cover_every_feature(drifted: DriftResult) -> None:
    """Normalised Wasserstein is the distance in reference standard deviations, so it can exceed 1."""
    assert tuple(drifted.column_scores) == FEATURE_COLUMNS
    assert all(score >= 0 for score in drifted.column_scores.values())


def test_drift_report_html(drifted: DriftResult) -> None:
    assert drifted.html.lstrip().startswith("<!DOCTYPE html>") and "Income" in drifted.html


def test_drift_small_reference_still_distances(customers: pd.DataFrame) -> None:
    """Below 1,000 reference rows Evidently would switch to p-values; the explicit methods keep distances."""
    result = build_drift_report(customers.head(120), customers.head(120))

    assert result.drifted_columns == []
    assert max(result.column_scores.values()) == pytest.approx(0)


def test_drift_window_without_missing_values(customers: pd.DataFrame) -> None:
    """Regression: with nullable dtypes Evidently's np.corrcoef failed on windows without any missing value."""
    complete = customers.dropna()

    result = build_drift_report(customers, complete)

    assert result.n_current == len(complete) > 0


@pytest.mark.parametrize("threshold", [0.0, 100.0])
def test_drift_threshold_applied(customers: pd.DataFrame, threshold: float) -> None:
    result = build_drift_report(customers, shifted(customers), threshold=threshold)

    assert len(result.drifted_columns) == (len(FEATURE_COLUMNS) if threshold == 0 else 0)


@pytest.mark.parametrize("empty", ["reference", "current"])
def test_drift_empty_frame_EmptyWindowError(customers: pd.DataFrame, empty: str) -> None:
    frames = {"reference": customers, "current": customers, empty: customers.head(0)}

    with pytest.raises(EmptyWindowError, match="cannot compare"):
        build_drift_report(frames["reference"], frames["current"])


def test_column_methods_explicit_per_type() -> None:
    methods = column_methods()

    assert set(methods) == set(FEATURE_COLUMNS)
    distributions = {"Education", "Marital_Status", "Kidhome", "Teenhome",
                     *(column for column in BINARY_COLUMNS if column in FEATURE_COLUMNS)}
    assert {column for column, method in methods.items() if method == "jensenshannon"} == distributions
    assert methods["Income"] == methods[DATE_COLUMN] == "wasserstein"


def test_drift_frame_types_and_date(customers: pd.DataFrame) -> None:
    before = customers.copy()

    frame = to_drift_frame(customers)

    assert tuple(frame.columns) == FEATURE_COLUMNS
    assert all(frame[column].dtype == "float64" for column in NUMERIC_COLUMNS)
    assert all(frame[column].dtype == object for column in CATEGORICAL_COLUMNS)
    first = customers[DATE_COLUMN].iloc[0]
    assert frame[DATE_COLUMN].iloc[0] == (first - pd.Timestamp("2012-01-01")).days
    pd.testing.assert_frame_equal(customers, before)

"""Tests for the predict service: training/serving parity of the model input, and scoring with a fitted model."""

import numpy as np
import pandas as pd
import pytest
from fakes import FakeRegistry, FakeTracker
from serving_helpers import add_champion, payload_records
from sklearn.model_selection import TunedThresholdClassifierCV

from campaign_response.data.cleaning import CleaningParams, clean, clean_records
from campaign_response.data.schema import DATE_COLUMN, FEATURE_COLUMNS, RAW_DTYPES
from campaign_response.pipeline.train_pipeline import split_features
from campaign_response.serving.model_cache import ChampionCache, NoChampionError
from campaign_response.serving.predict_service import PredictService, records_to_raw_frame
from campaign_response.serving.schemas import CustomerRecord


def to_records(raw: pd.DataFrame) -> list[CustomerRecord]:
    return [CustomerRecord.model_validate(values) for values in payload_records(raw)]


@pytest.fixture
def registry() -> FakeRegistry:
    return FakeRegistry(FakeTracker())


@pytest.fixture
def service(registry: FakeRegistry, model: TunedThresholdClassifierCV) -> PredictService:
    add_champion(registry, model)
    return PredictService(ChampionCache(registry), CleaningParams())


def test_raw_frame_has_raw_columns_and_dtypes(raw_frame: pd.DataFrame) -> None:
    frame = records_to_raw_frame(to_records(raw_frame))

    assert tuple(frame.columns) == FEATURE_COLUMNS
    assert {column: str(dtype) for column, dtype in frame.dtypes.items()} == {
        column: RAW_DTYPES[column] for column in FEATURE_COLUMNS
    }
    assert frame[DATE_COLUMN].tolist() == raw_frame[DATE_COLUMN].tolist()


def test_records_cleaned_like_training(raw_customers: pd.DataFrame) -> None:
    """Training/serving skew guard: API records must reach the model exactly as training rows did."""
    cleaned, _ = clean(raw_customers)

    served = clean_records(records_to_raw_frame(to_records(raw_customers)))

    pd.testing.assert_frame_equal(
        served.loc[cleaned.index].reset_index(drop=True), split_features(cleaned)[0].reset_index(drop=True)
    )


def test_service_scores_like_model(
    service: PredictService, model: TunedThresholdClassifierCV, customers: pd.DataFrame, raw_customers: pd.DataFrame
) -> None:
    result = service.predict(to_records(raw_customers.head(20)))

    expected = model.predict_proba(split_features(customers.head(20))[0])[:, 1]
    assert result.probabilities == pytest.approx(expected.tolist())
    assert result.labels == model.predict(split_features(customers.head(20))[0]).tolist()
    assert result.model_version.version == 1
    assert result.threshold == pytest.approx(model.best_threshold_)


def test_service_labels_follow_threshold(service: PredictService, raw_customers: pd.DataFrame) -> None:
    result = service.predict(to_records(raw_customers))

    assert result.labels == [int(probability >= result.threshold) for probability in result.probabilities]
    assert 0 < sum(result.labels) < len(result.labels)


def test_service_identical_records_both_scored(service: PredictService, raw_frame: pd.DataFrame) -> None:
    record = to_records(raw_frame)[0]

    result = service.predict([record, record])

    assert len(result.probabilities) == 2
    assert result.probabilities[0] == pytest.approx(result.probabilities[1])


def test_service_implausible_birth_years_masked(service: PredictService, raw_frame: pd.DataFrame) -> None:
    record = to_records(raw_frame)[0]

    result = service.predict([record.model_copy(update={"Year_Birth": year}) for year in (1893, 1905, 1913)])

    assert result.probabilities == pytest.approx([result.probabilities[0]] * 3)


def test_service_types_are_plain_python(service: PredictService, raw_frame: pd.DataFrame) -> None:
    result = service.predict(to_records(raw_frame))

    assert all(type(value) is float for value in [*result.probabilities, result.threshold])
    assert all(type(label) is int for label in result.labels)
    assert not any(isinstance(value, np.generic) for value in result.probabilities)


def test_service_no_champion_NoChampionError(registry: FakeRegistry, raw_frame: pd.DataFrame) -> None:
    service = PredictService(ChampionCache(registry), CleaningParams())

    with pytest.raises(NoChampionError):
        service.predict(to_records(raw_frame))

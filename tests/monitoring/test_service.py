"""Tests for one drift check on the fakes: reference, current window, retraining rule and the MLflow run."""

import logging
from contextlib import AbstractContextManager
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest
from fakes import FakePredictionStore, FakeRegistry, FakeTracker

from campaign_response.data.cleaning import CleaningParams
from campaign_response.data.io import write_processed
from campaign_response.data.schema import FEATURE_COLUMNS
from campaign_response.monitoring.reference import REFERENCE_ARTIFACT, ReferenceUnavailableError
from campaign_response.monitoring.retraining import MonitoringParams
from campaign_response.monitoring.service import (
    MONITORING_EXPERIMENT,
    NO_PREDICTIONS,
    REPORT_FILE,
    SUMMARY_FILE,
    DriftService,
)
from campaign_response.registry.ports import CHAMPION
from campaign_response.serving.schemas import CustomerRecord
from campaign_response.smoke import records_from_raw
from campaign_response.storage.prediction_store import PredictionRecord

PARAMS = MonitoringParams()


class BrokenTracker(FakeTracker):
    def run(self, experiment: str, run_name: str, parent_run_id: str | None = None) -> AbstractContextManager[str]:
        raise ConnectionError("mlflow at http://secret-host:5000 unreachable")


def add_champion(registry: FakeRegistry, customers: pd.DataFrame, tmp_path: Path) -> int:
    """A champion whose run logged the customers as its reference sample, as the training pipeline does."""
    path = tmp_path / "reference_sample.csv"
    write_processed(customers, path)
    with registry.tracker.run("training", "train") as run_id:
        registry.tracker.log_file(run_id, path, REFERENCE_ARTIFACT.rsplit("/", 1)[0])
        uri = registry.tracker.log_model(run_id, object(), customers.head(1))
    version = registry.register(uri, run_id).version
    registry.set_alias(CHAMPION, version)
    return version


def log(store: FakePredictionStore, records: list[CustomerRecord], model_version: int = 1) -> None:
    store.save([
        PredictionRecord(
            request_id=f"req-{index}", created_at=datetime(2026, 11, 25, tzinfo=UTC), model_version=model_version,
            source="test", customer_id=record.ID, features=record.model_dump(include=set(FEATURE_COLUMNS)),
            probability=0.5, label=0,
        )
        for index, record in enumerate(records)
    ])


def shifted(records: list[CustomerRecord]) -> list[CustomerRecord]:
    """Richer customers who buy less wine and come back later: the kind of shift the calibration used."""
    return [
        record.model_copy(update={
            "Income": None if record.Income is None else record.Income * 1.5,
            "MntWines": round(record.MntWines * 0.4),
            "Recency": min(record.Recency + 40, 99),
            "NumWebVisitsMonth": record.NumWebVisitsMonth + 6,
        })
        for record in records
    ]


@pytest.fixture
def registry() -> FakeRegistry:
    return FakeRegistry(FakeTracker())


@pytest.fixture
def store() -> FakePredictionStore:
    return FakePredictionStore()


def service(registry: FakeRegistry, store: FakePredictionStore) -> DriftService:
    return DriftService(registry, registry.tracker, store, CleaningParams(), PARAMS)


def test_check_same_customers_no_drift(
    registry: FakeRegistry, store: FakePredictionStore, raw_customers: pd.DataFrame, customers: pd.DataFrame,
    tmp_path: Path,
) -> None:
    version = add_champion(registry, customers, tmp_path)
    log(store, records_from_raw(raw_customers))

    check = service(registry, store).check(window=500)

    assert (check.reference_version, check.n_reference, check.n_current) == (version, len(customers), 240)
    assert check.drifted_columns == [] and not check.dataset_drift and not check.retrain_recommended
    assert check.reason == "0 of 25 columns drifted (alarm at 3)"
    assert tuple(check.column_scores) == FEATURE_COLUMNS and check.threshold == PARAMS.drift_threshold


def test_check_shifted_traffic_retrain(
    registry: FakeRegistry, store: FakePredictionStore, raw_customers: pd.DataFrame, customers: pd.DataFrame,
    tmp_path: Path,
) -> None:
    add_champion(registry, customers, tmp_path)
    log(store, shifted(records_from_raw(raw_customers)))

    check = service(registry, store).check(window=500)

    assert {"Income", "MntWines", "Recency", "NumWebVisitsMonth"} <= set(check.drifted_columns)
    assert check.dataset_drift and check.retrain_recommended
    assert check.share_drifted == len(check.drifted_columns) / len(FEATURE_COLUMNS)


def test_check_small_window_no_retrain(
    registry: FakeRegistry, store: FakePredictionStore, raw_customers: pd.DataFrame, customers: pd.DataFrame,
    tmp_path: Path,
) -> None:
    add_champion(registry, customers, tmp_path)
    log(store, shifted(records_from_raw(raw_customers)))

    check = service(registry, store).check(window=150)

    assert check.n_current == 150
    assert check.dataset_drift and not check.retrain_recommended
    assert check.reason.startswith("only 150 predictions in the window, at least 200 needed")


def test_check_logs_mlflow_run(
    registry: FakeRegistry, store: FakePredictionStore, raw_customers: pd.DataFrame, customers: pd.DataFrame,
    tmp_path: Path,
) -> None:
    version = add_champion(registry, customers, tmp_path)
    log(store, records_from_raw(raw_customers), model_version=version)

    check = service(registry, store).check(window=300, model_version=version)

    assert check.report_run_id is not None
    run = registry.tracker.runs[check.report_run_id]
    assert (run.experiment, run.name) == (MONITORING_EXPERIMENT, f"drift-v{version}")
    assert run.tags == {
        "reference_version": str(version), "window": "300", "model_version_filter": str(version),
        "dataset_drift": "false", "retrain_recommended": "false",
    }
    assert run.metrics["n_current"] == 240 and run.metrics["n_drifted"] == 0
    assert run.metrics["drift_Income"] == check.column_scores["Income"]
    assert registry.tracker.json_artifact(check.report_run_id, SUMMARY_FILE)["reason"] == check.reason
    assert run.artifacts[REPORT_FILE].lstrip().startswith(b"<!DOCTYPE html>")


def test_check_model_version_filter(
    registry: FakeRegistry, store: FakePredictionStore, raw_customers: pd.DataFrame, customers: pd.DataFrame,
    tmp_path: Path,
) -> None:
    add_champion(registry, customers, tmp_path)
    records = records_from_raw(raw_customers)
    log(store, records[:100], model_version=1)
    log(store, records[100:], model_version=2)

    assert service(registry, store).check(window=500, model_version=1).n_current == 100
    assert service(registry, store).check(window=500).n_current == 240


def test_check_empty_log_no_report(
    registry: FakeRegistry, store: FakePredictionStore, customers: pd.DataFrame, tmp_path: Path
) -> None:
    version = add_champion(registry, customers, tmp_path)

    check = service(registry, store).check(window=500)

    assert (check.reference_version, check.n_current, check.reason) == (version, 0, NO_PREDICTIONS)
    assert not check.retrain_recommended and check.report_run_id is None
    assert [run.experiment for run in registry.tracker.runs.values()] == ["training"]


def test_check_no_champion_ReferenceUnavailableError(registry: FakeRegistry, store: FakePredictionStore) -> None:
    with pytest.raises(ReferenceUnavailableError, match="no champion"):
        service(registry, store).check(window=500)


def test_check_logging_failure_result_kept(
    store: FakePredictionStore, raw_customers: pd.DataFrame, customers: pd.DataFrame, tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    registry = FakeRegistry(FakeTracker())
    add_champion(registry, customers, tmp_path)
    log(store, records_from_raw(raw_customers))
    drift = DriftService(registry, BrokenTracker(), store, CleaningParams(), PARAMS)

    with caplog.at_level(logging.ERROR):
        check = drift.check(window=500)

    assert check.n_current == 240 and check.report_run_id is None
    [record] = caplog.records
    assert record.error_type == "ConnectionError"  # type: ignore[attr-defined]  # set through extra=
    assert "secret-host" not in caplog.text


def test_summary_without_run_id(
    registry: FakeRegistry, store: FakePredictionStore, customers: pd.DataFrame, tmp_path: Path
) -> None:
    add_champion(registry, customers, tmp_path)

    summary = service(registry, store).check(window=10).summary()

    assert "report_run_id" not in summary and summary["reason"] == NO_PREDICTIONS

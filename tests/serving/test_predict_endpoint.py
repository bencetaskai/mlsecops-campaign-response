"""Tests for POST /predict: the response, the prediction log rows and the MLflow inference run, plus error paths."""

import io
import json
import logging
from typing import Any

import pandas as pd
import pytest
from fakes import FakePredictionStore, FakeRegistry, FakeTracker
from fastapi.testclient import TestClient
from serving_helpers import AUTH, MODEL_NAME, NOW, add_champion, make_dependencies, payload_records
from sklearn.model_selection import TunedThresholdClassifierCV
from sqlalchemy.exc import OperationalError

from campaign_response.data.schema import FEATURE_COLUMNS
from campaign_response.logging_setup import CorrelationIdFilter
from campaign_response.registry.ports import CHAMPION
from campaign_response.serving.app import create_app
from campaign_response.serving.middleware import REQUEST_ID_HEADER
from campaign_response.serving.prediction_logging import INFERENCE_EXPERIMENT, PREDICTIONS_FILE, REQUEST_ARTIFACT


@pytest.fixture
def registry() -> FakeRegistry:
    return FakeRegistry(FakeTracker())


@pytest.fixture
def store() -> FakePredictionStore:
    return FakePredictionStore()


@pytest.fixture
def client(registry: FakeRegistry, store: FakePredictionStore) -> TestClient:
    return TestClient(create_app(make_dependencies(registry, store=store)))


@pytest.fixture
def records(raw_customers: pd.DataFrame) -> list[dict[str, Any]]:
    first = payload_records(raw_customers.head(3))
    return [{**first[0], "ID": 101}, {**first[1], "Income": None}, first[2]]


def inference_runs(registry: FakeRegistry) -> list[Any]:
    return [run for run in registry.tracker.runs.values() if run.experiment == INFERENCE_EXPERIMENT]


def test_predict_200_one_result_per_record(
    client: TestClient, registry: FakeRegistry, model: TunedThresholdClassifierCV, records: list[dict[str, Any]]
) -> None:
    add_champion(registry, model)

    response = client.post("/predict", json={"records": records}, headers={**AUTH, REQUEST_ID_HEADER: "req-7"})

    assert response.status_code == 200
    body = response.json()
    assert (body["request_id"], body["model_name"], body["model_version"], body["model_alias"]) == (
        "req-7", MODEL_NAME, 1, CHAMPION,
    )
    assert body["threshold"] == pytest.approx(model.best_threshold_)
    assert [prediction["ID"] for prediction in body["predictions"]] == [101, None, None]
    assert all(
        prediction["label"] == int(prediction["probability"] >= body["threshold"])
        for prediction in body["predictions"]
    )


def test_predict_rows_written_to_log(
    client: TestClient, registry: FakeRegistry, store: FakePredictionStore, model: TunedThresholdClassifierCV,
    records: list[dict[str, Any]],
) -> None:
    add_champion(registry, model)

    body = client.post("/predict", json={"records": records, "source": "simulator"}, headers=AUTH).json()

    assert len(store.rows) == 3
    first = store.rows[0]
    assert (first["request_id"], first["created_at"], first["model_version"], first["source"], first["ID"]) == (
        body["request_id"], NOW, 1, "simulator", 101,
    )
    assert first["Year_Birth"] == records[0]["Year_Birth"]
    assert first["Dt_Customer"].isoformat() == records[0]["Dt_Customer"]
    assert set(first) >= set(FEATURE_COLUMNS)
    assert store.rows[1]["Income"] is None
    assert [row["probability"] for row in store.rows] == [p["probability"] for p in body["predictions"]]


def test_predict_inference_run_logged(
    client: TestClient, registry: FakeRegistry, model: TunedThresholdClassifierCV, records: list[dict[str, Any]]
) -> None:
    add_champion(registry, model)

    body = client.post("/predict", json={"records": records}, headers={**AUTH, REQUEST_ID_HEADER: "req-8"}).json()

    [run] = inference_runs(registry)
    assert (run.name, run.status) == ("predict-req-8", "FINISHED")
    assert run.tags == {"request_id": "req-8", "model_name": MODEL_NAME, "model_version": "1", "source": "api"}
    probabilities = [prediction["probability"] for prediction in body["predictions"]]
    assert run.metrics["n_records"] == 3
    assert run.metrics["mean_probability"] == pytest.approx(sum(probabilities) / 3)
    assert run.metrics["threshold"] == pytest.approx(body["threshold"])
    assert json.loads(run.artifacts[REQUEST_ARTIFACT])["records"][0]["ID"] == 101
    logged = pd.read_csv(io.BytesIO(run.artifacts[PREDICTIONS_FILE]))
    assert logged.columns.tolist() == ["ID", "probability", "label"]
    assert logged["probability"].tolist() == pytest.approx(probabilities)


def test_predict_no_key_401_nothing_logged(
    client: TestClient, registry: FakeRegistry, store: FakePredictionStore, model: TunedThresholdClassifierCV,
    records: list[dict[str, Any]],
) -> None:
    add_champion(registry, model)

    response = client.post("/predict", json={"records": records})

    assert response.status_code == 401
    assert store.rows == [] and inference_runs(registry) == []


@pytest.mark.parametrize("payload", [
    {"records": []},
    {"records": "not a list"},
    {"records": [{"Year_Birth": 1970}]},
    {"source": "unknown"},
], ids=["empty", "not-list", "incomplete-record", "unknown-source"])
def test_predict_invalid_body_422_nothing_logged(
    client: TestClient, registry: FakeRegistry, store: FakePredictionStore, model: TunedThresholdClassifierCV,
    records: list[dict[str, Any]], payload: dict[str, Any],
) -> None:
    add_champion(registry, model)

    response = client.post("/predict", json={"records": records, **payload}, headers=AUTH)

    assert response.status_code == 422
    assert store.rows == [] and inference_runs(registry) == []


def test_predict_no_champion_503(
    client: TestClient, registry: FakeRegistry, store: FakePredictionStore, records: list[dict[str, Any]]
) -> None:
    response = client.post("/predict", json={"records": records}, headers=AUTH)

    assert response.status_code == 503
    assert "POST /train" in response.json()["detail"]
    assert store.rows == [] and inference_runs(registry) == []


def test_predict_in_openapi_with_security(client: TestClient) -> None:
    operation = client.get("/openapi.json").json()["paths"]["/predict"]["post"]

    assert operation["security"] == [{"APIKeyHeader": []}]
    assert {"200", "401", "422", "503"} <= set(operation["responses"])


def fail_with_customer_data(*args: object) -> None:
    raise OperationalError("INSERT INTO prediction_log ...", {"Income": 777777}, ConnectionError("db down"))


def errors(logs: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [record for record in logs.records if record.levelno == logging.ERROR]


@pytest.fixture
def logs(caplog: pytest.LogCaptureFixture) -> pytest.LogCaptureFixture:
    caplog.handler.addFilter(CorrelationIdFilter())
    caplog.set_level(logging.INFO)
    return caplog


def test_predict_log_write_fails_still_200(
    client: TestClient, registry: FakeRegistry, store: FakePredictionStore, model: TunedThresholdClassifierCV,
    records: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch, logs: pytest.LogCaptureFixture,
) -> None:
    add_champion(registry, model)
    monkeypatch.setattr(store, "save", fail_with_customer_data)

    response = client.post("/predict", json={"records": records}, headers={**AUTH, REQUEST_ID_HEADER: "req-9"})

    assert response.status_code == 200
    assert len(response.json()["predictions"]) == 3
    [error] = errors(logs)
    assert error.__dict__["correlation_id"] == "req-9"
    assert (error.__dict__["error_type"], error.__dict__["error_code"], error.__dict__["n_records"]) == (
        "OperationalError", "e3q8", 3,
    )
    assert "777777" not in logs.text
    assert len(inference_runs(registry)) == 1


def test_predict_mlflow_unreachable_still_200(
    client: TestClient, registry: FakeRegistry, store: FakePredictionStore, model: TunedThresholdClassifierCV,
    records: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch, logs: pytest.LogCaptureFixture,
) -> None:
    add_champion(registry, model)

    def refuse(*args: object, **kwargs: object) -> None:
        raise ConnectionError("cannot reach http://user:secret@mlflow:5000")

    monkeypatch.setattr(registry.tracker, "run", refuse)

    response = client.post("/predict", json={"records": records}, headers={**AUTH, REQUEST_ID_HEADER: "req-10"})

    assert response.status_code == 200
    assert len(store.rows) == 3
    [error] = errors(logs)
    assert (error.message, error.__dict__["error_type"], error.__dict__["error_code"]) == (
        "inference run logging failed", "ConnectionError", None,
    )
    assert error.__dict__["correlation_id"] == "req-10"
    assert "secret" not in logs.text


def test_predict_artifact_upload_fails_run_marked_failed(
    client: TestClient, registry: FakeRegistry, model: TunedThresholdClassifierCV, records: list[dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch, logs: pytest.LogCaptureFixture,
) -> None:
    add_champion(registry, model)
    monkeypatch.setattr(registry.tracker, "log_file", fail_with_customer_data)

    response = client.post("/predict", json={"records": records}, headers=AUTH)

    assert response.status_code == 200
    [run] = inference_runs(registry)
    assert run.status == "FAILED"
    assert [error.message for error in errors(logs)] == ["inference run logging failed"]

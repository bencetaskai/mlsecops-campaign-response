"""Tests for POST /predict: the response, the prediction log rows and the MLflow inference run, plus error paths."""

import io
import json
from typing import Any

import pandas as pd
import pytest
from fakes import FakePredictionStore, FakeRegistry, FakeTracker
from fastapi.testclient import TestClient
from serving_helpers import AUTH, MODEL_NAME, NOW, add_champion, make_dependencies, payload_records
from sklearn.model_selection import TunedThresholdClassifierCV

from campaign_response.data.schema import FEATURE_COLUMNS
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

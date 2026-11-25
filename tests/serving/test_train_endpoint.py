"""Tests for POST /train: real training runs on the synthetic raw file with the fakes, plus the error paths."""

from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from fakes import FakeRegistry, FakeTracker
from fastapi.testclient import TestClient
from serving_helpers import AUTH, REQUIREMENTS, make_dependencies, payload_records

from campaign_response.data.cleaning import CleaningParams
from campaign_response.data.splitting import SplitParams
from campaign_response.params import PipelineParams
from campaign_response.pipeline.train_pipeline import TrainingContext
from campaign_response.registry.ports import CHAMPION
from campaign_response.registry.promotion import GateParams
from campaign_response.serving.app import TRAINING_DATA_INVALID, TRAINING_DATA_MISSING, create_app
from campaign_response.serving.model_cache import ChampionCache
from campaign_response.serving.training_service import TrainingInProgressError, TrainingService

FAST: dict[str, Any] = {"candidates": ["logistic_regression"], "n_iter": 1, "cv_folds": 3, "seed": 1}


@pytest.fixture
def registry() -> FakeRegistry:
    return FakeRegistry(FakeTracker())


@pytest.fixture
def client(registry: FakeRegistry, raw_customers_file: Path) -> TestClient:
    return TestClient(create_app(make_dependencies(registry, raw_customers_file)))


def strict_client(registry: FakeRegistry, raw_file: Path) -> TestClient:
    """An app whose gate cannot be passed, so every model is rejected."""
    cache = ChampionCache(registry)
    context = TrainingContext(
        tracker=registry.tracker, registry=registry,
        params=PipelineParams(CleaningParams(), SplitParams(), GateParams(min_pr_auc=1.0)),
        raw_data_path=raw_file, requirements_path=REQUIREMENTS, environ={"GIT_SHA": "abc123"},
    )
    return TestClient(create_app(make_dependencies(registry, cache=cache, training=TrainingService(context, cache))))


def test_train_first_model_champion(client: TestClient, registry: FakeRegistry) -> None:
    response = client.post("/train", json=FAST, headers=AUTH)

    assert response.status_code == 200
    body = response.json()
    assert (body["model_version"], body["best_candidate"], body["promoted"]) == (1, "logistic_regression", True)
    assert (body["champion_before"], body["champion_after"]) == (None, 1)
    assert set(body["holdout_metrics"]) == {"pr_auc", "roc_auc", "f1", "precision", "recall"}
    assert all(0 <= value <= 1 for value in body["holdout_metrics"].values())
    assert body["duration_s"] >= 0 and body["gate_reason"]
    champion = registry.get_by_alias(CHAMPION)
    assert champion is not None and champion.version == 1
    assert registry.tracker.runs[body["run_id"]].tags["trained_by"] == "api"


def test_train_second_model_served_at_once(client: TestClient, raw_customers: pd.DataFrame) -> None:
    client.post("/train", json=FAST, headers=AUTH)
    records = payload_records(raw_customers.head(2))
    assert client.post("/predict", json={"records": records}, headers=AUTH).json()["model_version"] == 1

    body = client.post("/train", json={**FAST, "trained_by": "airflow"}, headers=AUTH).json()

    assert (body["model_version"], body["promoted"], body["champion_before"], body["champion_after"]) == (2, True, 1, 2)
    assert client.post("/predict", json={"records": records}, headers=AUTH).json()["model_version"] == 2


def test_train_trigger_recorded(client: TestClient, registry: FakeRegistry) -> None:
    body = client.post("/train", json={**FAST, "trained_by": "bootstrap"}, headers=AUTH).json()

    assert registry.tracker.runs[body["run_id"]].tags["trained_by"] == "bootstrap"


def test_train_rejected_model_200_not_promoted(registry: FakeRegistry, raw_customers_file: Path) -> None:
    response = strict_client(registry, raw_customers_file).post("/train", json=FAST, headers=AUTH)

    assert response.status_code == 200
    body = response.json()
    assert (body["promoted"], body["champion_before"], body["champion_after"]) == (False, None, None)
    assert body["gate_reason"].startswith("floor: pr_auc")
    assert registry.get_by_alias(CHAMPION) is None


def test_train_busy_409(client: TestClient, registry: FakeRegistry, monkeypatch: pytest.MonkeyPatch) -> None:
    def busy(request: object) -> None:
        raise TrainingInProgressError("a training run is already in progress; try again when it has finished")

    monkeypatch.setattr(TrainingService, "train", lambda self, request: busy(request))

    response = client.post("/train", json=FAST, headers=AUTH)

    assert response.status_code == 409
    assert "already in progress" in response.json()["detail"]


def test_train_data_missing_503(registry: FakeRegistry) -> None:
    client = TestClient(create_app(make_dependencies(registry)))

    response = client.post("/train", json=FAST, headers=AUTH)

    assert response.status_code == 503
    assert response.json() == {"detail": TRAINING_DATA_MISSING}
    assert registry.get_by_alias(CHAMPION) is None


def test_train_data_invalid_503(registry: FakeRegistry, raw_customers: pd.DataFrame, tmp_path: Path) -> None:
    raw_file = tmp_path / "marketing_campaign.csv"
    raw_customers.head(50).to_csv(raw_file, sep="\t", index=False, lineterminator="\r\n")
    client = TestClient(create_app(make_dependencies(registry, raw_file)))

    response = client.post("/train", json=FAST, headers=AUTH)

    assert response.status_code == 503
    assert response.json() == {"detail": TRAINING_DATA_INVALID}


@pytest.mark.parametrize("payload", [
    {"candidates": ["dummy"]}, {"n_iter": 11}, {"cv_folds": 2}, {"trained_by": "cli"}, {"promote": True},
], ids=["baseline", "n_iter", "cv_folds", "trigger", "extra-field"])
def test_train_invalid_body_422(client: TestClient, registry: FakeRegistry, payload: dict[str, Any]) -> None:
    response = client.post("/train", json={**FAST, **payload}, headers=AUTH)

    assert response.status_code == 422
    assert registry.tracker.runs == {}


def test_train_no_key_401(client: TestClient, registry: FakeRegistry) -> None:
    response = client.post("/train", json=FAST)

    assert response.status_code == 401
    assert registry.tracker.runs == {}


def test_train_in_openapi_with_security(client: TestClient) -> None:
    operation = client.get("/openapi.json").json()["paths"]["/train"]["post"]

    assert operation["security"] == [{"APIKeyHeader": []}]
    assert {"200", "401", "409", "422", "503"} <= set(operation["responses"])

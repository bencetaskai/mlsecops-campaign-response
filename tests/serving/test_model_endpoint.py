"""Tests for GET /model: the aliased versions after real training runs, and versions made outside the pipeline."""

from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from fakes import FakeRegistry, FakeTracker
from fastapi.testclient import TestClient
from serving_helpers import AUTH, MODEL_NAME, make_dependencies

from campaign_response.registry.ports import CHALLENGER, CHAMPION
from campaign_response.serving.app import REGISTRY_UNREACHABLE, create_app

FAST: dict[str, Any] = {"candidates": ["logistic_regression"], "n_iter": 1, "cv_folds": 3, "seed": 1}


@pytest.fixture
def registry() -> FakeRegistry:
    return FakeRegistry(FakeTracker())


@pytest.fixture
def client(registry: FakeRegistry, raw_customers_file: Path) -> TestClient:
    return TestClient(create_app(make_dependencies(registry, raw_customers_file)))


def test_model_empty_registry_no_versions(client: TestClient) -> None:
    response = client.get("/model")

    assert response.status_code == 200
    assert response.json() == {"model_name": MODEL_NAME, "versions": []}


def test_model_after_two_runs_three_aliases(client: TestClient) -> None:
    first = client.post("/train", json=FAST, headers=AUTH).json()
    second = client.post("/train", json=FAST, headers=AUTH).json()

    versions = client.get("/model").json()["versions"]

    assert [(entry["alias"], entry["version"]) for entry in versions] == [
        ("champion", 2), ("challenger", 2), ("previous", 1),
    ]
    champion = versions[0]
    assert (champion["run_id"], champion["best_candidate"], champion["promoted"]) == (
        second["run_id"], "logistic_regression", True,
    )
    assert champion["holdout_metrics"] == pytest.approx(second["holdout_metrics"])
    assert champion["threshold"] == pytest.approx(second["threshold"])
    assert champion["gate_reason"] == second["gate_reason"]
    assert versions[2]["run_id"] == first["run_id"]


def test_model_version_outside_pipeline_nulls(client: TestClient, registry: FakeRegistry) -> None:
    with registry.tracker.run("manual", "upload") as run_id:
        uri = registry.tracker.log_model(run_id, object(), pd.DataFrame({"x": [1]}))
    registry.set_alias(CHALLENGER, registry.register(uri, run_id).version)

    [entry] = client.get("/model").json()["versions"]

    assert entry == {
        "alias": CHALLENGER, "version": 1, "run_id": run_id, "best_candidate": None, "threshold": None,
        "holdout_metrics": None, "promoted": None, "gate_reason": None,
    }


def test_model_registry_unreachable_503(
    client: TestClient, registry: FakeRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(alias: str) -> None:
        raise ConnectionError("cannot reach http://user:secret@mlflow:5000")

    monkeypatch.setattr(registry, "get_by_alias", refuse)

    response = client.get("/model")

    assert response.status_code == 503
    assert response.json() == {"detail": REGISTRY_UNREACHABLE}


def test_model_open_without_key(client: TestClient) -> None:
    operation = client.get("/openapi.json").json()["paths"]["/model"]["get"]

    assert "security" not in operation
    assert CHAMPION in operation["description"]

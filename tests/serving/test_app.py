"""Tests for the app factory and the health and version endpoints, with the in-memory registry."""

import logging

import pandas as pd
import pytest
from fakes import FakeRegistry, FakeTracker
from fastapi.testclient import TestClient
from serving_helpers import make_dependencies

from campaign_response import __version__
from campaign_response.registry.ports import CHAMPION
from campaign_response.serving.app import NO_CHAMPION, REGISTRY_UNREACHABLE, create_app

SECRET_URI = "http://user:secret@mlflow:5000"


@pytest.fixture
def registry() -> FakeRegistry:
    return FakeRegistry(FakeTracker())


@pytest.fixture
def client(registry: FakeRegistry) -> TestClient:
    return TestClient(create_app(make_dependencies(registry, git_sha="fbdb428")))


def add_champion(registry: FakeRegistry) -> None:
    with registry.tracker.run("training", "train") as run_id:
        uri = registry.tracker.log_model(run_id, object(), pd.DataFrame({"x": [1]}))
    registry.set_alias(CHAMPION, registry.register(uri, run_id).version)


def make_unreachable(registry: FakeRegistry, monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(alias: str) -> None:
        raise ConnectionError(f"cannot reach {SECRET_URI}")

    monkeypatch.setattr(registry, "get_by_alias", refuse)


def test_live_always_ok(client: TestClient, registry: FakeRegistry, monkeypatch: pytest.MonkeyPatch) -> None:
    make_unreachable(registry, monkeypatch)

    response = client.get("/health/live")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "detail": None}


def test_ready_no_champion_503(client: TestClient) -> None:
    response = client.get("/health/ready")

    assert response.status_code == 503
    assert response.json() == {"status": "unavailable", "detail": NO_CHAMPION}


def test_ready_with_champion_200(client: TestClient, registry: FakeRegistry) -> None:
    add_champion(registry)

    response = client.get("/health/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "detail": "champion version 1"}


def test_ready_registry_unreachable_503_uri_hidden(
    client: TestClient, registry: FakeRegistry, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    make_unreachable(registry, monkeypatch)

    with caplog.at_level(logging.WARNING):
        response = client.get("/health/ready")

    assert response.status_code == 503
    assert response.json()["detail"] == REGISTRY_UNREACHABLE
    assert [(record.message, record.__dict__["error_type"]) for record in caplog.records] == [
        (REGISTRY_UNREACHABLE, "ConnectionError")
    ]
    assert "secret" not in response.text + caplog.text


def test_version_with_champion(client: TestClient, registry: FakeRegistry) -> None:
    add_champion(registry)

    response = client.get("/version")

    assert response.status_code == 200
    assert response.json() == {
        "app_version": __version__, "git_sha": "fbdb428",
        "model_name": "campaign-response-classifier", "champion_version": 1,
    }


def test_version_registry_unreachable_null_champion(
    client: TestClient, registry: FakeRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_unreachable(registry, monkeypatch)

    response = client.get("/version")

    assert response.status_code == 200
    assert response.json()["champion_version"] is None


def test_openapi_lists_endpoints(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()

    assert schema["info"]["version"] == __version__
    assert {"/health/live", "/health/ready", "/version"} <= set(schema["paths"])
    assert "503" in schema["paths"]["/health/ready"]["get"]["responses"]

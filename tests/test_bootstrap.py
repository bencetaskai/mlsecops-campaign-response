"""Tests for the bootstrap job against a scripted stand-in API: skip, train, wait and fail."""

import logging
from pathlib import Path

import httpx2
import pytest
from api_stub import Answer, StubApi, ok, overview, status, trained
from pydantic import SecretStr

from campaign_response import bootstrap
from campaign_response.api_client import ApiClient
from campaign_response.bootstrap import ApiNotReadyError, ensure_champion, wait_for_registry
from campaign_response.serving.schemas import ModelOverview

KEY = SecretStr("k" * 40)


def client_for(stub: StubApi) -> ApiClient:
    return ApiClient("http://api:8000", KEY, stub.transport)


def no_wait(client: ApiClient) -> ModelOverview:
    return wait_for_registry(client, attempts=3, delay=0, sleep=lambda seconds: None)


def test_bootstrap_champion_exists_no_training() -> None:
    stub = StubApi({"GET /model": [ok(overview("champion", "previous"))]})

    assert ensure_champion(client_for(stub), wait=no_wait)
    assert stub.calls() == ["GET /model"]


def test_bootstrap_only_challenger_trains() -> None:
    stub = StubApi({"GET /model": [ok(overview("challenger"))], "POST /train": [ok(trained())]})

    assert ensure_champion(client_for(stub), wait=no_wait)
    assert stub.calls() == ["GET /model", "POST /train"]
    assert stub.body(1) == {**stub.body(1), "trained_by": "bootstrap", "n_iter": 10, "cv_folds": 5}


def test_bootstrap_first_model_rejected_false(caplog: pytest.LogCaptureFixture) -> None:
    stub = StubApi({"GET /model": [ok(overview())], "POST /train": [ok(trained(promoted=False))]})

    with caplog.at_level(logging.ERROR):
        assert not ensure_champion(client_for(stub), wait=no_wait)

    assert caplog.records[0].__dict__["gate_reason"].startswith("floor:")


@pytest.mark.parametrize("failures", [
    [httpx2.ConnectError("connection refused")] * 2,
    [status(503), httpx2.ReadTimeout("slow")],
], ids=["api-starting", "registry-starting"])
def test_wait_retries_until_answer(failures: list[Answer]) -> None:
    stub = StubApi({"GET /model": [*failures, ok(overview())]})
    sleeps: list[float] = []

    result = wait_for_registry(client_for(stub), attempts=5, delay=2.5, sleep=sleeps.append)

    assert result.versions == []
    assert sleeps == [2.5, 2.5]


def test_wait_gives_up_after_attempts() -> None:
    stub = StubApi({"GET /model": [httpx2.ConnectError("connection refused")]})
    sleeps: list[float] = []

    with pytest.raises(ApiNotReadyError, match="3 attempts"):
        wait_for_registry(client_for(stub), attempts=3, delay=1, sleep=sleeps.append)

    assert len(stub.requests) == 3 and sleeps == [1, 1]


def test_wait_other_status_not_retried() -> None:
    stub = StubApi({"GET /model": [status(401)]})

    with pytest.raises(httpx2.HTTPStatusError):
        wait_for_registry(client_for(stub), attempts=3, delay=0, sleep=lambda seconds: None)

    assert len(stub.requests) == 1


@pytest.fixture
def environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[StubApi]:
    """Settings from the environment, logging untouched, and the job's client served by the stub put in the list."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CRP_API_URL", "http://api:8000")
    monkeypatch.setenv("CRP_API_KEY", KEY.get_secret_value())
    monkeypatch.setattr(bootstrap, "configure_logging", lambda level, log_format: None)
    stubs: list[StubApi] = []
    monkeypatch.setattr(bootstrap, "ApiClient", lambda url, key: ApiClient(url, key, stubs[0].transport))
    return stubs


def test_main_champion_exists_exit_0(environment: list[StubApi]) -> None:
    environment.append(StubApi({"GET /model": [ok(overview("champion"))]}))

    assert bootstrap.main() == 0
    assert environment[0].requests[0].headers["X-API-Key"] == KEY.get_secret_value()


def test_main_rejected_first_model_exit_1(environment: list[StubApi]) -> None:
    environment.append(StubApi({"GET /model": [ok(overview())], "POST /train": [ok(trained(promoted=False))]}))

    assert bootstrap.main() == 1


def test_main_training_error_exit_1(environment: list[StubApi], caplog: pytest.LogCaptureFixture) -> None:
    environment.append(StubApi({"GET /model": [ok(overview())], "POST /train": [status(503)]}))

    with caplog.at_level(logging.ERROR):
        assert bootstrap.main() == 1

    assert caplog.records[-1].__dict__["error_type"] == "HTTPStatusError"


def test_main_api_never_ready_exit_1(environment: list[StubApi], monkeypatch: pytest.MonkeyPatch) -> None:
    environment.append(StubApi({}))

    def never(client: ApiClient) -> bool:
        raise ApiNotReadyError("the API did not answer within 30 attempts")

    monkeypatch.setattr(bootstrap, "ensure_champion", never)

    assert bootstrap.main() == 1

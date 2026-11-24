"""Tests for the request id and timing middleware, on the real app with two extra test routes."""

import logging

import pytest
from fakes import FakeRegistry, FakeTracker
from fastapi.testclient import TestClient

from campaign_response.logging_setup import NO_CORRELATION_ID, CorrelationIdFilter, correlation_id_var
from campaign_response.serving.app import ApiDependencies, create_app
from campaign_response.serving.middleware import REQUEST_ID_HEADER, request_id_from


@pytest.fixture
def client() -> TestClient:
    app = create_app(ApiDependencies(FakeRegistry(FakeTracker()), "campaign-response-classifier", "abc123"))

    @app.get("/echo")
    def echo() -> dict[str, str]:
        logging.getLogger("test.echo").info("inside endpoint")
        return {"correlation_id": correlation_id_var.get()}

    @app.get("/boom")
    def boom() -> None:
        raise RuntimeError("secret detail")

    return TestClient(app)


@pytest.fixture
def logs(caplog: pytest.LogCaptureFixture) -> pytest.LogCaptureFixture:
    caplog.handler.addFilter(CorrelationIdFilter())
    caplog.set_level(logging.INFO)
    return caplog


def finished(logs: pytest.LogCaptureFixture) -> logging.LogRecord:
    [record] = [record for record in logs.records if record.message == "request finished"]
    return record


@pytest.mark.parametrize("value", ["abc-123", "9f1c.b2_X", "a" * 64])
def test_request_id_safe_header_reused(value: str) -> None:
    assert request_id_from(value) == value


@pytest.mark.parametrize("value", [None, "", "a" * 65, "id\nlevel=ERROR forged", "id with spaces", "<script>"])
def test_request_id_unsafe_header_replaced(value: str | None) -> None:
    generated = request_id_from(value)

    assert generated != value
    assert len(generated) == 16 and generated.isalnum()


def test_middleware_header_echoed(client: TestClient) -> None:
    response = client.get("/echo", headers={REQUEST_ID_HEADER: "client-id-1"})

    assert response.headers[REQUEST_ID_HEADER] == "client-id-1"
    assert response.json() == {"correlation_id": "client-id-1"}


def test_middleware_missing_header_generated(client: TestClient) -> None:
    first = client.get("/echo").headers[REQUEST_ID_HEADER]
    second = client.get("/echo").headers[REQUEST_ID_HEADER]

    assert len(first) == 16 and first != second


def test_middleware_every_log_line_tagged(client: TestClient, logs: pytest.LogCaptureFixture) -> None:
    client.get("/echo?customer=5524", headers={REQUEST_ID_HEADER: "trace-me"})

    server = [record for record in logs.records if not record.name.startswith("httpx")]

    assert {record.message: record.__dict__["correlation_id"] for record in server} == {
        "inside endpoint": "trace-me", "request finished": "trace-me",
    }
    record = finished(logs)
    assert (record.__dict__["method"], record.__dict__["path"], record.__dict__["status"]) == ("GET", "/echo", 200)
    assert record.__dict__["duration_ms"] >= 0


def test_middleware_context_reset_after_request(client: TestClient) -> None:
    client.get("/echo", headers={REQUEST_ID_HEADER: "short-lived"})

    assert correlation_id_var.get() == NO_CORRELATION_ID


def test_middleware_not_found_still_logged(client: TestClient, logs: pytest.LogCaptureFixture) -> None:
    response = client.get("/missing")

    assert response.status_code == 404
    assert REQUEST_ID_HEADER in response.headers
    assert finished(logs).__dict__["status"] == 404


def test_middleware_unhandled_error_generic_500(client: TestClient, logs: pytest.LogCaptureFixture) -> None:
    response = client.get("/boom", headers={REQUEST_ID_HEADER: "broken-1"})

    assert response.status_code == 500
    assert response.json() == {"detail": "internal server error", "request_id": "broken-1"}
    assert response.headers[REQUEST_ID_HEADER] == "broken-1"
    assert "secret detail" not in response.text
    [error] = [record for record in logs.records if record.message == "unhandled error"]
    assert error.exc_info is not None and error.__dict__["correlation_id"] == "broken-1"
    assert finished(logs).__dict__["status"] == 500

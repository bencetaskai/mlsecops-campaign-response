"""Tests for the API key dependency, on a minimal app with one protected route."""

import logging

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from serving_helpers import API_KEY, make_dependencies

from campaign_response.serving.app import MissingApiKeyError, create_app
from campaign_response.serving.security import API_KEY_HEADER, INVALID_API_KEY, api_key_checker

WRONG_KEY = "w" * 40


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()

    @app.get("/protected", dependencies=[Depends(api_key_checker(API_KEY))])
    def protected() -> dict[str, str]:
        return {"status": "ok"}

    return TestClient(app)


def test_api_key_right_key_200(client: TestClient) -> None:
    response = client.get("/protected", headers={API_KEY_HEADER: API_KEY.get_secret_value()})

    assert response.status_code == 200


@pytest.mark.parametrize("headers", [
    {},
    {API_KEY_HEADER: ""},
    {API_KEY_HEADER: WRONG_KEY},
    {API_KEY_HEADER: API_KEY.get_secret_value()[:-1]},
    {API_KEY_HEADER: API_KEY.get_secret_value() + "x"},
    {"Authorization": f"Bearer {API_KEY.get_secret_value()}"},
], ids=["missing", "empty", "wrong", "prefix", "longer", "other-header"])
def test_api_key_missing_or_wrong_401(client: TestClient, headers: dict[str, str]) -> None:
    response = client.get("/protected", headers=headers)

    assert response.status_code == 401
    assert response.json() == {"detail": INVALID_API_KEY}
    assert response.headers["WWW-Authenticate"] == API_KEY_HEADER


def test_api_key_never_logged(client: TestClient, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.DEBUG):
        client.get("/protected", headers={API_KEY_HEADER: WRONG_KEY})
        client.get("/protected", headers={API_KEY_HEADER: API_KEY.get_secret_value()})

    assert [record.__dict__["key_sent"] for record in caplog.records if "API key" in record.message] == [True]
    assert WRONG_KEY not in caplog.text
    assert API_KEY.get_secret_value() not in caplog.text


def test_api_key_in_openapi_schema(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()

    assert schema["components"]["securitySchemes"]["APIKeyHeader"] == {
        "type": "apiKey", "in": "header", "name": API_KEY_HEADER, "description": "Required for /train and /predict.",
    }


def test_create_app_without_key_refuses() -> None:
    dependencies = make_dependencies(api_key=None)

    with pytest.raises(MissingApiKeyError, match="CRP_API_KEY"):
        create_app(dependencies)

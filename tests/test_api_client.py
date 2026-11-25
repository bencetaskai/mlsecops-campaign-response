"""Tests for the API client against a scripted stand-in API."""

import httpx2
import pandas as pd
import pytest
from api_stub import StubApi, ok, overview, status, trained
from pydantic import SecretStr
from serving_helpers import payload_records

from campaign_response.api_client import TRAIN_TIMEOUT_SECONDS, ApiClient
from campaign_response.serving.schemas import CustomerRecord, PredictRequest, PredictResponse, TrainRequest
from campaign_response.serving.security import API_KEY_HEADER

KEY = SecretStr("k" * 40)
URL = "http://api:8000"


def test_client_model_overview_parsed() -> None:
    stub = StubApi({"GET /model": [ok(overview("champion"))]})

    result = ApiClient(URL, None, stub.transport).model_overview()

    assert [entry.alias for entry in result.versions] == ["champion"]
    assert API_KEY_HEADER not in stub.requests[0].headers


def test_client_train_sends_key_and_body() -> None:
    stub = StubApi({"POST /train": [ok(trained())]})

    result = ApiClient(URL, KEY, stub.transport).train(TrainRequest(candidates=["logistic_regression"], n_iter=1))

    assert result.promoted
    request = stub.requests[0]
    assert str(request.url) == f"{URL}/train"
    assert request.headers[API_KEY_HEADER] == KEY.get_secret_value()
    assert stub.body(0)["candidates"] == ["logistic_regression"] and stub.body(0)["n_iter"] == 1
    assert request.extensions["timeout"]["read"] == TRAIN_TIMEOUT_SECONDS


def test_client_predict_round_trip(raw_frame: pd.DataFrame) -> None:
    record = payload_records(raw_frame)[0]
    response = PredictResponse(request_id="r1", model_name="m", model_version=2, model_alias="champion",
                               threshold=0.3, predictions=[])
    stub = StubApi({"POST /predict": [ok(response)]})

    result = ApiClient(URL, KEY, stub.transport).predict(
        PredictRequest(records=[CustomerRecord.model_validate(record)], source="test")
    )

    assert result.model_version == 2
    assert stub.body(0)["source"] == "test"
    assert stub.body(0)["records"][0]["Dt_Customer"] == record["Dt_Customer"]


@pytest.mark.parametrize("code", [401, 422, 503])
def test_client_error_status_raises(code: int) -> None:
    stub = StubApi({"GET /model": [status(code)]})

    with pytest.raises(httpx2.HTTPStatusError) as error:
        ApiClient(URL, KEY, stub.transport).model_overview()

    assert error.value.response.status_code == code


def test_client_close_closes_connection() -> None:
    client = ApiClient(URL, KEY, StubApi({}).transport)

    client.close()

    assert client._http.is_closed

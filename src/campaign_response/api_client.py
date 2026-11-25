"""A typed client for the REST API, used by the bootstrap job and the smoke script.

Responses are parsed with the server's own response models, so a change on one side breaks the tests on the other.
"""

from typing import Final

import httpx2
from pydantic import SecretStr

from campaign_response.serving.schemas import (
    ModelOverview,
    PredictRequest,
    PredictResponse,
    TrainRequest,
    TrainResponse,
)
from campaign_response.serving.security import API_KEY_HEADER

DEFAULT_TIMEOUT_SECONDS: Final = 30.0
# A training run takes about 2 minutes with the defaults and at most about 4 within the API's limits.
TRAIN_TIMEOUT_SECONDS: Final = 900.0


class ApiClient:
    def __init__(
        self, base_url: str, api_key: SecretStr | None, transport: httpx2.BaseTransport | None = None
    ) -> None:
        headers = {API_KEY_HEADER: api_key.get_secret_value()} if api_key is not None else {}
        self._http = httpx2.Client(
            base_url=base_url, headers=headers, transport=transport, timeout=DEFAULT_TIMEOUT_SECONDS,
        )

    def close(self) -> None:
        self._http.close()

    def model_overview(self) -> ModelOverview:
        response = self._http.get("/model")
        response.raise_for_status()
        return ModelOverview.model_validate_json(response.content)

    def train(self, request: TrainRequest) -> TrainResponse:
        response = self._http.post("/train", json=request.model_dump(mode="json"), timeout=TRAIN_TIMEOUT_SECONDS)
        response.raise_for_status()
        return TrainResponse.model_validate_json(response.content)

    def predict(self, request: PredictRequest) -> PredictResponse:
        response = self._http.post("/predict", json=request.model_dump(mode="json"))
        response.raise_for_status()
        return PredictResponse.model_validate_json(response.content)

"""A stand-in API for client tests: scripted answers per path, served through httpx2's MockTransport."""

import json
from typing import Any

import httpx2
from pydantic import BaseModel

from campaign_response.serving.schemas import (
    AliasedVersion,
    HoldoutMetrics,
    ModelOverview,
    TrainResponse,
)

Answer = httpx2.Response | Exception
METRICS = HoldoutMetrics(pr_auc=0.69, roc_auc=0.9, f1=0.6, precision=0.55, recall=0.66)


def overview(*aliases: str) -> ModelOverview:
    return ModelOverview(model_name="campaign-response-classifier", versions=[
        AliasedVersion(alias=alias, version=1, run_id="run-1", best_candidate="logistic_regression", threshold=0.3,
                       holdout_metrics=METRICS, promoted=True, gate_reason="first model") for alias in aliases
    ])


def trained(promoted: bool = True) -> TrainResponse:
    return TrainResponse(
        run_id="run-1", model_name="campaign-response-classifier", model_version=1,
        best_candidate="logistic_regression", threshold=0.3, holdout_metrics=METRICS, champion_before=None,
        champion_after=1 if promoted else None, promoted=promoted,
        gate_reason="first model" if promoted else "floor: pr_auc 0.2 vs floor 0.3", duration_s=95.0,
    )


def ok(model: BaseModel) -> httpx2.Response:
    return httpx2.Response(200, content=model.model_dump_json())


def status(code: int) -> httpx2.Response:
    return httpx2.Response(code, json={"detail": "scripted"})


class StubApi:
    """Answers each "METHOD /path" from its queue, last answer repeating; records every request."""

    def __init__(self, answers: dict[str, list[Answer]]) -> None:
        self.answers = answers
        self.requests: list[httpx2.Request] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        queue = self.answers[f"{request.method} {request.url.path}"]
        answer = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(answer, Exception):
            raise answer
        return answer

    @property
    def transport(self) -> httpx2.MockTransport:
        return httpx2.MockTransport(self)

    def calls(self) -> list[str]:
        return [f"{request.method} {request.url.path}" for request in self.requests]

    def body(self, index: int) -> Any:
        return json.loads(self.requests[index].content)

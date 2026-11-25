"""FastAPI app factory: every dependency comes in as an argument, so tests build the app with fakes.

Endpoints are plain ``def``, not ``async def``: the MLflow client blocks, and FastAPI runs sync endpoints in a
thread pool instead of stalling the event loop.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request, Response, status
from pydantic import SecretStr

from campaign_response import __version__
from campaign_response.data.cleaning import CleaningParams
from campaign_response.registry.ports import CHAMPION, ExperimentTracker, ModelRegistry, ModelVersion
from campaign_response.serving.middleware import request_context
from campaign_response.serving.model_cache import ChampionCache, ModelUnavailableError
from campaign_response.serving.predict_service import PredictService
from campaign_response.serving.prediction_logging import log_inference_run, prediction_records, save_predictions
from campaign_response.serving.schemas import (
    HealthStatus,
    Prediction,
    PredictRequest,
    PredictResponse,
    VersionInfo,
)
from campaign_response.serving.security import api_key_checker
from campaign_response.storage.prediction_store import PredictionStore

logger = logging.getLogger(__name__)

NO_CHAMPION = "no champion model yet; call POST /train"
REGISTRY_UNREACHABLE = "model registry unreachable"


class MissingApiKeyError(RuntimeError):
    """The API was started without an API key; it fails closed instead of running unprotected."""


@dataclass(frozen=True)
class ApiDependencies:
    registry: ModelRegistry
    tracker: ExperimentTracker
    cache: ChampionCache
    store: PredictionStore
    cleaning: CleaningParams
    model_name: str
    git_sha: str
    api_key: SecretStr | None
    clock: Callable[[], datetime] = lambda: datetime.now(UTC)


def _champion(registry: ModelRegistry) -> tuple[ModelVersion | None, str | None]:
    """The champion and None, or None and why there is none."""
    try:
        champion = registry.get_by_alias(CHAMPION)
    # Broad on purpose: an unreachable MLflow raises MlflowException or requests errors depending on where it fails,
    # and a health check must answer 503 for all of them, never 500. Only the type is logged: the message can
    # contain the tracking URI.
    except Exception as error:
        logger.warning(REGISTRY_UNREACHABLE, extra={"error_type": type(error).__name__})
        return None, REGISTRY_UNREACHABLE
    return champion, None if champion is not None else NO_CHAMPION


def create_app(dependencies: ApiDependencies) -> FastAPI:
    if dependencies.api_key is None:
        raise MissingApiKeyError("no API key configured: set CRP_API_KEY or the crp_api_key Docker secret")
    app = FastAPI(
        title="Campaign response API",
        version=__version__,
        description="Trains campaign response models with a promotion gate and serves the champion.",
    )
    app.middleware("http")(request_context)
    require_api_key = Depends(api_key_checker(dependencies.api_key))
    predict_service = PredictService(dependencies.cache, dependencies.cleaning)

    @app.get("/health/live", tags=["health"])
    def live() -> HealthStatus:
        """The process is up; says nothing about the model."""
        return HealthStatus(status="ok")

    @app.get("/health/ready", tags=["health"], responses={503: {"model": HealthStatus}})
    def ready(response: Response) -> HealthStatus:
        """Ready to predict: the registry answers and a champion exists."""
        champion, problem = _champion(dependencies.registry)
        if champion is None:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            return HealthStatus(status="unavailable", detail=problem)
        return HealthStatus(status="ok", detail=f"champion version {champion.version}")

    @app.get("/version", tags=["health"])
    def version() -> VersionInfo:
        """Code and model version; the champion is null while there is none or the registry is unreachable."""
        champion, _ = _champion(dependencies.registry)
        return VersionInfo(
            app_version=__version__,
            git_sha=dependencies.git_sha,
            model_name=dependencies.model_name,
            champion_version=champion.version if champion is not None else None,
        )

    @app.post(
        "/predict",
        tags=["model"],
        dependencies=[require_api_key],
        responses={401: {"description": "Missing or invalid API key"}, 503: {"description": "No model to serve"}},
    )
    def predict(body: PredictRequest, request: Request, background: BackgroundTasks) -> PredictResponse:
        """Score 1-1000 customers with the champion; every record and result goes to the prediction log."""
        request_id: str = request.state.request_id
        try:
            scored = predict_service.predict(body.records)
        except ModelUnavailableError as error:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error)) from error
        save_predictions(dependencies.store, prediction_records(request_id, dependencies.clock(), body, scored))
        # About 0.4 s against the compose MLflow, so it runs after the response instead of delaying it.
        background.add_task(log_inference_run, dependencies.tracker, request_id, body, scored)
        logger.info("scored request", extra={
            "n_records": len(body.records), "model_version": scored.model_version.version, "source": body.source,
        })
        return PredictResponse(
            request_id=request_id,
            model_name=scored.model_version.name,
            model_version=scored.model_version.version,
            model_alias=CHAMPION,
            threshold=scored.threshold,
            predictions=[
                Prediction(ID=record.ID, probability=probability, label=label)
                for record, probability, label in zip(body.records, scored.probabilities, scored.labels, strict=True)
            ],
        )

    return app

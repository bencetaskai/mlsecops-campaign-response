"""FastAPI app factory: every dependency comes in as an argument, so tests build the app with fakes.

Endpoints are plain ``def``, not ``async def``: the MLflow client blocks, and FastAPI runs sync endpoints in a
thread pool instead of stalling the event loop.
"""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request, Response, params, status
from pydantic import SecretStr

from campaign_response import __version__
from campaign_response.data.cleaning import CleaningParams
from campaign_response.data.io import RawDataError
from campaign_response.pipeline.train_pipeline import DataValidationError, TrainingOutcome, TrainingRequest
from campaign_response.registry.ports import CHAMPION, ExperimentTracker, ModelRegistry, ModelVersion
from campaign_response.serving.middleware import request_context
from campaign_response.serving.model_cache import ChampionCache, ModelUnavailableError
from campaign_response.serving.predict_service import PredictService
from campaign_response.serving.prediction_logging import log_inference_run, prediction_records, save_predictions
from campaign_response.serving.schemas import (
    HealthStatus,
    HoldoutMetrics,
    Prediction,
    PredictRequest,
    PredictResponse,
    TrainRequest,
    TrainResponse,
    VersionInfo,
)
from campaign_response.serving.security import api_key_checker
from campaign_response.serving.training_service import TrainingInProgressError, TrainingService
from campaign_response.storage.prediction_store import PredictionStore

logger = logging.getLogger(__name__)

NO_CHAMPION = "no champion model yet; call POST /train"
TRAINING_DATA_MISSING = "training data not found on the server; fetch it with `dvc pull`"
TRAINING_DATA_INVALID = "training data failed validation; see validation_report.json in the training run"
REGISTRY_UNREACHABLE = "model registry unreachable"


class MissingApiKeyError(RuntimeError):
    """The API was started without an API key; it fails closed instead of running unprotected."""


@dataclass(frozen=True)
class ApiDependencies:
    registry: ModelRegistry
    tracker: ExperimentTracker
    cache: ChampionCache
    training: TrainingService
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
    _add_health_routes(app, dependencies)
    _add_predict_route(app, dependencies, require_api_key)
    _add_train_route(app, dependencies, require_api_key)
    return app


def _add_health_routes(app: FastAPI, dependencies: ApiDependencies) -> None:
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


def _add_predict_route(app: FastAPI, dependencies: ApiDependencies, require_api_key: params.Depends) -> None:
    predict_service = PredictService(dependencies.cache, dependencies.cleaning)

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


def _add_train_route(app: FastAPI, dependencies: ApiDependencies, require_api_key: params.Depends) -> None:
    @app.post(
        "/train",
        tags=["model"],
        dependencies=[require_api_key],
        responses={
            401: {"description": "Missing or invalid API key"},
            409: {"description": "Another training run is in progress"},
            503: {"description": "Training data missing or invalid"},
        },
    )
    def train(body: TrainRequest) -> TrainResponse:
        """Train the candidates, register the best as challenger and promote it only if the gate passes.

        Takes about 2 minutes with the defaults. A rejected challenger is a normal outcome (200, promoted false).
        """
        started = time.perf_counter()
        request = TrainingRequest(
            candidates=tuple(body.candidates), n_iter=body.n_iter, cv_folds=body.cv_folds, seed=body.seed,
            trained_by=body.trained_by,
        )
        unavailable = status.HTTP_503_SERVICE_UNAVAILABLE
        try:
            outcome = dependencies.training.train(request)
        except TrainingInProgressError as error:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error
        # Server-side data problems, not request errors; the message names no path or file content.
        except FileNotFoundError as error:
            raise HTTPException(status_code=unavailable, detail=TRAINING_DATA_MISSING) from error
        except (DataValidationError, RawDataError) as error:
            raise HTTPException(status_code=unavailable, detail=TRAINING_DATA_INVALID) from error
        return _train_response(outcome, time.perf_counter() - started)


def _train_response(outcome: TrainingOutcome, seconds: float) -> TrainResponse:
    before = outcome.champion.version if outcome.champion is not None else None
    holdout = outcome.holdout
    return TrainResponse(
        run_id=outcome.run_id,
        model_name=outcome.version.name,
        model_version=outcome.version.version,
        best_candidate=outcome.best_candidate,
        threshold=outcome.threshold,
        holdout_metrics=HoldoutMetrics(
            pr_auc=holdout.pr_auc, roc_auc=holdout.roc_auc, f1=holdout.f1, precision=holdout.precision,
            recall=holdout.recall,
        ),
        champion_before=before,
        champion_after=outcome.version.version if outcome.promoted else before,
        promoted=outcome.promoted,
        gate_reason=outcome.decision.reason,
        duration_s=round(seconds, 1),
    )

"""FastAPI app factory: every dependency comes in as an argument, so tests build the app with fakes.

Endpoints are plain ``def``, not ``async def``: the MLflow client blocks, and FastAPI runs sync endpoints in a
thread pool instead of stalling the event loop.
"""

import logging
from dataclasses import dataclass

from fastapi import FastAPI, Response, status

from campaign_response import __version__
from campaign_response.registry.ports import CHAMPION, ModelRegistry, ModelVersion
from campaign_response.serving.middleware import request_context
from campaign_response.serving.schemas import HealthStatus, VersionInfo

logger = logging.getLogger(__name__)

NO_CHAMPION = "no champion model yet; call POST /train"
REGISTRY_UNREACHABLE = "model registry unreachable"


@dataclass(frozen=True)
class ApiDependencies:
    registry: ModelRegistry
    model_name: str
    git_sha: str


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
    app = FastAPI(
        title="Campaign response API",
        version=__version__,
        description="Trains campaign response models with a promotion gate and serves the champion.",
    )
    app.middleware("http")(request_context)

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

    return app

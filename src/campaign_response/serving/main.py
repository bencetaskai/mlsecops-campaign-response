"""Entry point of the API: builds the real adapters from the settings and runs uvicorn.

Run with ``python -m campaign_response.serving.main``. Everything testable lives in build_dependencies(); the app
itself is created in the worker by uvicorn's factory mode, so settings errors surface at start-up, not at import.
"""

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Final

import uvicorn
from fastapi import FastAPI
from sqlalchemy import URL, Engine, create_engine

from campaign_response.config import Settings
from campaign_response.logging_setup import configure_logging
from campaign_response.params import load_params
from campaign_response.pipeline.train_pipeline import TrainingContext
from campaign_response.registry.lineage import git_sha
from campaign_response.registry.mlflow_adapter import MlflowRegistry, MlflowTracker
from campaign_response.serving.app import ApiDependencies, create_app
from campaign_response.serving.model_cache import ChampionCache
from campaign_response.serving.training_service import TrainingService
from campaign_response.storage.prediction_store import SqlPredictionStore

RAW_FILE: Final = Path("raw") / "marketing_campaign.csv"
REQUIREMENTS_FILE: Final = Path("requirements.txt")
SQLITE_FILE: Final = "prediction_log.db"


class MissingDatabasePasswordError(RuntimeError):
    """A database host is configured without its password."""


def prediction_db_url(settings: Settings) -> URL:
    """Postgres from the settings' parts, or a SQLite file in data_dir when no host is set.

    URL.create escapes the password, so special characters cannot break the URL.
    """
    if settings.prediction_db_host is None:
        return URL.create("sqlite", database=str(settings.data_dir / SQLITE_FILE))
    if settings.prediction_db_password is None:
        raise MissingDatabasePasswordError(
            "prediction_db_host is set but no password: set the crp_prediction_db_password secret"
        )
    return URL.create(
        "postgresql+psycopg",
        username=settings.prediction_db_user,
        password=settings.prediction_db_password.get_secret_value(),
        host=settings.prediction_db_host,
        port=settings.prediction_db_port,
        database=settings.prediction_db_name,
    )


def prediction_db_engine(settings: Settings) -> Engine:
    # hide_parameters: SQLAlchemy errors would otherwise quote the inserted values, i.e. customer data.
    # pool_pre_ping: a connection dropped by a Postgres restart is replaced instead of failing the next request.
    return create_engine(prediction_db_url(settings), hide_parameters=True, pool_pre_ping=True)


def build_dependencies(settings: Settings, environ: Mapping[str, str] = os.environ) -> ApiDependencies:
    tracker = MlflowTracker(settings.mlflow_tracking_uri)
    registry = MlflowRegistry(settings.mlflow_tracking_uri, settings.model_name)
    cache = ChampionCache(registry, ttl_seconds=settings.model_cache_ttl_seconds)
    params = load_params(settings.params_path)
    store = SqlPredictionStore(prediction_db_engine(settings))
    store.create_schema()
    context = TrainingContext(
        tracker=tracker,
        registry=registry,
        params=params,
        raw_data_path=settings.data_dir / RAW_FILE,
        requirements_path=REQUIREMENTS_FILE,
        environ=environ,
    )
    return ApiDependencies(
        registry=registry,
        tracker=tracker,
        cache=cache,
        training=TrainingService(context, cache),
        store=store,
        cleaning=params.cleaning,
        model_name=settings.model_name,
        git_sha=git_sha(environ),
        api_key=settings.api_key,
    )


def create() -> FastAPI:
    """The app for uvicorn's factory mode."""
    settings = Settings()
    configure_logging(settings.log_level, settings.log_format)
    return create_app(build_dependencies(settings))


def main() -> None:
    settings = Settings()
    uvicorn.run(
        "campaign_response.serving.main:create",
        factory=True,
        host=settings.api_host,
        port=settings.api_port,
        # One worker: the training lock and the model cache are per process (see TrainingService).
        workers=1,
        # Our logging setup stays in charge, and the request middleware already logs every request once.
        log_config=None,
        access_log=False,
    )


if __name__ == "__main__":  # pragma: no cover  # module run as a script; main() itself is tested
    main()

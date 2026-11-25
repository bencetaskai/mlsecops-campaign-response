"""Tests for the API entry point: settings to adapters, the database URL and the uvicorn call."""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import mlflow
import pytest
from fastapi.testclient import TestClient
from serving_helpers import API_KEY
from sqlalchemy import inspect

from campaign_response.config import Settings
from campaign_response.registry.mlflow_adapter import MlflowRegistry, MlflowTracker
from campaign_response.serving import main
from campaign_response.storage.prediction_store import TABLE_NAME, SqlPredictionStore

REPO = Path(__file__).resolve().parents[2]
UNREACHABLE_MLFLOW = "http://localhost:1"
PASSWORD = "p@ss/w:rd#1"


@pytest.fixture(autouse=True)
def restore_tracking_uri() -> Iterator[None]:
    """MlflowTracker sets the process-wide tracking URI; other tests must not inherit it."""
    previous = mlflow.get_tracking_uri()
    yield
    mlflow.set_tracking_uri(previous)


def settings(tmp_path: Path, **changes: Any) -> Settings:
    values: dict[str, Any] = {
        "data_dir": tmp_path, "api_key": API_KEY, "mlflow_tracking_uri": UNREACHABLE_MLFLOW,
        "params_path": REPO / "params.yaml", **changes,
    }
    return Settings(_env_file=None, **values)


def test_db_url_without_host_sqlite_in_data_dir(tmp_path: Path) -> None:
    url = main.prediction_db_url(settings(tmp_path))

    assert (url.drivername, url.database) == ("sqlite", str(tmp_path / main.SQLITE_FILE))


def test_db_url_postgres_password_escaped_and_hidden(tmp_path: Path) -> None:
    url = main.prediction_db_url(settings(tmp_path, prediction_db_host="postgres", prediction_db_password=PASSWORD))

    assert (url.drivername, url.host, url.port, url.database, url.username) == (
        "postgresql+psycopg", "postgres", 5432, "monitoring", "monitoring",
    )
    assert url.password == PASSWORD
    assert "p%40ss%2Fw%3Ard%231@postgres" in url.render_as_string(hide_password=False)
    assert PASSWORD not in str(url)


def test_db_url_host_without_password_refused(tmp_path: Path) -> None:
    with pytest.raises(main.MissingDatabasePasswordError, match="crp_prediction_db_password"):
        main.prediction_db_url(settings(tmp_path, prediction_db_host="postgres"))


def test_db_engine_hides_parameters(tmp_path: Path) -> None:
    engine = main.prediction_db_engine(settings(tmp_path))

    assert engine.hide_parameters
    engine.dispose()


def test_build_dependencies_real_adapters(tmp_path: Path) -> None:
    dependencies = main.build_dependencies(settings(tmp_path, model_cache_ttl_seconds=5), environ={"GIT_SHA": "f00"})

    assert isinstance(dependencies.registry, MlflowRegistry)
    assert isinstance(dependencies.tracker, MlflowTracker)
    assert isinstance(dependencies.store, SqlPredictionStore)
    assert dependencies.git_sha == "f00"
    assert dependencies.api_key == API_KEY
    assert dependencies.cache._ttl == 5
    assert dependencies.training._context.raw_data_path == tmp_path / main.RAW_FILE
    assert dependencies.cleaning.reference_date.isoformat() == "2014-12-31"
    assert TABLE_NAME in inspect(dependencies.store._engine).get_table_names()


def test_create_app_from_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    for name, value in {
        "CRP_API_KEY": API_KEY.get_secret_value(), "CRP_DATA_DIR": str(tmp_path),
        "CRP_MLFLOW_TRACKING_URI": UNREACHABLE_MLFLOW, "CRP_PARAMS_PATH": str(REPO / "params.yaml"),
    }.items():
        monkeypatch.setenv(name, value)
    configured: list[tuple[str, str]] = []
    monkeypatch.setattr(main, "configure_logging", lambda level, log_format: configured.append((level, log_format)))

    app = main.create()

    assert configured == [("INFO", "text")]
    assert TestClient(app).get("/health/live").status_code == 200


def test_main_runs_uvicorn_single_worker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CRP_API_HOST", "0.0.0.0")
    monkeypatch.setenv("CRP_API_PORT", "8000")
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    monkeypatch.setattr(main.uvicorn, "run", lambda *args, **kwargs: calls.append((args, kwargs)))

    main.main()

    [(args, kwargs)] = calls
    assert args == ("campaign_response.serving.main:create",)
    assert kwargs == {
        "factory": True, "host": "0.0.0.0", "port": 8000, "workers": 1, "log_config": None, "access_log": False,
    }

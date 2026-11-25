"""Values and builders shared by the API tests."""

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fakes import FakePredictionStore, FakeRegistry, FakeTracker
from pydantic import SecretStr

from campaign_response.data.cleaning import CleaningParams
from campaign_response.data.schema import DATE_COLUMN, DATE_FORMAT, FEATURE_COLUMNS
from campaign_response.data.splitting import SplitParams
from campaign_response.monitoring.retraining import MonitoringParams
from campaign_response.params import PipelineParams
from campaign_response.pipeline.train_pipeline import TrainingContext
from campaign_response.registry.ports import CHAMPION
from campaign_response.registry.promotion import GateParams
from campaign_response.serving.app import ApiDependencies
from campaign_response.serving.model_cache import ChampionCache
from campaign_response.serving.security import API_KEY_HEADER
from campaign_response.serving.training_service import TrainingService

# Repeated characters: obviously fake and low in entropy, so gitleaks has nothing to flag and needs no exception.
API_KEY = SecretStr("k" * 40)
AUTH = {API_KEY_HEADER: API_KEY.get_secret_value()}
NOW = datetime(2026, 11, 25, 12, 0, tzinfo=UTC)
MODEL_NAME = "campaign-response-classifier"
REQUIREMENTS = Path(__file__).resolve().parents[1] / "requirements.txt"


def make_dependencies(
    registry: FakeRegistry | None = None, raw_data_path: Path = Path("missing/marketing_campaign.csv"),
    **changes: object,
) -> ApiDependencies:
    """Fakes for every port, the registry's own tracker, a fixed clock; keyword arguments replace single fields.

    Training reads raw_data_path, by default a file that does not exist.
    """
    registry = registry if registry is not None else FakeRegistry(FakeTracker())
    cache = ChampionCache(registry)
    context = TrainingContext(
        tracker=registry.tracker,
        registry=registry,
        params=PipelineParams(CleaningParams(), SplitParams(), GateParams(), MonitoringParams()),
        raw_data_path=raw_data_path,
        requirements_path=REQUIREMENTS,
        environ={"GIT_SHA": "abc123"},
    )
    dependencies = ApiDependencies(
        registry=registry,
        tracker=registry.tracker,
        cache=cache,
        training=TrainingService(context, cache),
        store=FakePredictionStore(),
        cleaning=CleaningParams(),
        model_name=MODEL_NAME,
        git_sha="abc123",
        api_key=API_KEY,
        clock=lambda: NOW,
    )
    return replace(dependencies, **changes)  # type: ignore[arg-type]  # field names and types checked at runtime


def payload_records(raw: pd.DataFrame) -> list[dict[str, Any]]:
    """Raw rows as JSON records for /predict: ISO dates, null for missing values, plain Python numbers."""
    records = []
    for row in raw[list(FEATURE_COLUMNS)].to_dict("records"):
        values: dict[str, Any] = {
            str(column): None if pd.isna(value) else value.item() if isinstance(value, np.generic) else value
            for column, value in row.items()
        }
        values[DATE_COLUMN] = datetime.strptime(values[DATE_COLUMN], DATE_FORMAT).date().isoformat()
        records.append(values)
    return records


def add_champion(registry: FakeRegistry, model: object) -> int:
    """Log the model in the registry's tracker, register it and make it champion; returns the version."""
    with registry.tracker.run("training", "train") as run_id:
        uri = registry.tracker.log_model(run_id, model, pd.DataFrame({"x": [1]}))
    version = registry.register(uri, run_id).version
    registry.set_alias(CHAMPION, version)
    return version

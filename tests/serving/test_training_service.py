"""Tests for the training service: real training runs on the fakes, and the one-run-at-a-time lock."""

import threading
from pathlib import Path

import pytest
from fakes import FakeRegistry, FakeTracker

from campaign_response.data.cleaning import CleaningParams
from campaign_response.data.splitting import SplitParams
from campaign_response.monitoring.retraining import MonitoringParams
from campaign_response.params import PipelineParams
from campaign_response.pipeline import train_pipeline
from campaign_response.pipeline.train_pipeline import TrainingContext, TrainingOutcome, TrainingRequest
from campaign_response.registry.ports import CHAMPION
from campaign_response.registry.promotion import GateParams
from campaign_response.serving.model_cache import ChampionCache
from campaign_response.serving.training_service import TrainingInProgressError, TrainingService

REQUIREMENTS = Path(__file__).resolve().parents[2] / "requirements.txt"
REQUEST = TrainingRequest(candidates=("logistic_regression",), n_iter=1, cv_folds=3, seed=1, trained_by="api")


@pytest.fixture
def registry() -> FakeRegistry:
    return FakeRegistry(FakeTracker())


@pytest.fixture
def cache(registry: FakeRegistry) -> ChampionCache:
    return ChampionCache(registry)


@pytest.fixture
def service(registry: FakeRegistry, cache: ChampionCache, raw_customers_file: Path) -> TrainingService:
    context = TrainingContext(
        tracker=registry.tracker,
        registry=registry,
        params=PipelineParams(CleaningParams(), SplitParams(), GateParams(), MonitoringParams()),
        raw_data_path=raw_customers_file,
        requirements_path=REQUIREMENTS,
        environ={"GIT_SHA": "abc123"},
    )
    return TrainingService(context, cache)


def test_training_service_first_run_champion(service: TrainingService, registry: FakeRegistry) -> None:
    outcome = service.train(REQUEST)

    assert outcome.promoted
    assert registry.get_by_alias(CHAMPION) == outcome.version
    assert registry.tracker.runs[outcome.run_id].tags["trained_by"] == "api"
    assert not service.running


def test_training_service_promotion_served_at_once(
    service: TrainingService, cache: ChampionCache
) -> None:
    service.train(REQUEST)
    assert cache.get().version.version == 1

    second = service.train(REQUEST)

    assert second.promoted
    assert cache.get().version.version == 2


def test_training_service_second_run_409_while_busy(
    service: TrainingService, monkeypatch: pytest.MonkeyPatch
) -> None:
    started, release = threading.Event(), threading.Event()
    real_run = train_pipeline.run_training

    def slow_run(request: TrainingRequest, context: TrainingContext) -> TrainingOutcome:
        started.set()
        assert release.wait(timeout=10)
        return real_run(request, context)

    monkeypatch.setattr(train_pipeline, "run_training", slow_run)
    outcomes: list[TrainingOutcome] = []
    first = threading.Thread(target=lambda: outcomes.append(service.train(REQUEST)))
    first.start()
    assert started.wait(timeout=10)

    assert service.running
    with pytest.raises(TrainingInProgressError, match="already in progress"):
        service.train(REQUEST)
    release.set()
    first.join(timeout=60)

    assert [outcome.version.version for outcome in outcomes] == [1]
    assert not service.running


def test_training_service_failure_releases_lock(
    service: TrainingService, cache: ChampionCache, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(request: TrainingRequest, context: TrainingContext) -> TrainingOutcome:
        raise FileNotFoundError("raw data file not found")

    monkeypatch.setattr(train_pipeline, "run_training", broken)
    invalidated: list[bool] = []
    monkeypatch.setattr(cache, "invalidate", lambda: invalidated.append(True))

    with pytest.raises(FileNotFoundError):
        service.train(REQUEST)

    assert not service.running
    assert invalidated == [True]

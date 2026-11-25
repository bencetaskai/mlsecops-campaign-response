"""Tests for loading the champion's reference sample, after a real training run on the in-memory fakes."""

from pathlib import Path

import pandas as pd
import pytest
from fakes import FakeRegistry, FakeTracker

from campaign_response.data.cleaning import CleaningParams, clean
from campaign_response.data.splitting import SplitParams, split_train_holdout
from campaign_response.monitoring.reference import ReferenceUnavailableError, load_reference
from campaign_response.monitoring.retraining import MonitoringParams
from campaign_response.params import PipelineParams
from campaign_response.pipeline.train_pipeline import TrainingContext, TrainingRequest, run_training
from campaign_response.registry.ports import CHAMPION
from campaign_response.registry.promotion import GateParams

REQUIREMENTS = Path(__file__).resolve().parents[2] / "requirements.txt"
REQUEST = TrainingRequest(candidates=("logistic_regression",), n_iter=1, cv_folds=3, seed=1, trained_by="cli")


@pytest.fixture
def registry() -> FakeRegistry:
    return FakeRegistry(FakeTracker())


def train(registry: FakeRegistry, raw_file: Path) -> int:
    context = TrainingContext(
        tracker=registry.tracker, registry=registry,
        params=PipelineParams(CleaningParams(), SplitParams(), GateParams(), MonitoringParams()),
        raw_data_path=raw_file, requirements_path=REQUIREMENTS, environ={"GIT_SHA": "abc123"},
    )
    return run_training(REQUEST, context).version.version


def test_reference_is_champion_train_split(
    registry: FakeRegistry, raw_customers_file: Path, raw_customers: pd.DataFrame
) -> None:
    version = train(registry, raw_customers_file)

    reference = load_reference(registry)

    expected, _ = split_train_holdout(clean(raw_customers)[0], SplitParams())
    assert reference.version.version == version
    pd.testing.assert_frame_equal(reference.data, expected.reset_index(drop=True))


def test_reference_follows_champion(registry: FakeRegistry, raw_customers_file: Path) -> None:
    train(registry, raw_customers_file)
    second = train(registry, raw_customers_file)

    assert load_reference(registry).version.version == second


def test_reference_no_champion_unavailable(registry: FakeRegistry) -> None:
    with pytest.raises(ReferenceUnavailableError, match="no champion"):
        load_reference(registry)


def test_reference_missing_sample_unavailable(registry: FakeRegistry) -> None:
    with registry.tracker.run("manual", "upload") as run_id:
        uri = registry.tracker.log_model(run_id, object(), pd.DataFrame({"x": [1]}))
    registry.set_alias(CHAMPION, registry.register(uri, run_id).version)

    with pytest.raises(ReferenceUnavailableError, match="version 1 has no reference sample"):
        load_reference(registry)

"""Registry behaviour every implementation must show, run against the in-memory fake and a real local MLflow.

If the fake drifts away from MLflow's behaviour, the same test passes for one backend and fails for the other.
"""

from collections.abc import Iterator
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest
from fakes import FakeRegistry, FakeTracker
from sklearn.dummy import DummyClassifier

from campaign_response.registry.mlflow_adapter import MlflowRegistry, MlflowTracker
from campaign_response.registry.ports import CHAMPION, ExperimentTracker, ModelRegistry

FEATURES = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
TARGET = [0, 1, 0, 1]


@pytest.fixture(params=["fake", "mlflow"])
def backend(request: pytest.FixtureRequest) -> Iterator[tuple[ExperimentTracker, ModelRegistry]]:
    if request.param == "fake":
        tracker = FakeTracker()
        yield tracker, FakeRegistry(tracker)
    else:
        # One store per module; a fresh model name per test gives each test its own version numbering.
        uri = request.getfixturevalue("mlflow_tracking_uri")
        yield MlflowTracker(uri), MlflowRegistry(uri, f"contract-{uuid4().hex[:8]}")


def register_model(backend: tuple[ExperimentTracker, ModelRegistry], constant: int = 0) -> int:
    tracker, registry = backend
    model = DummyClassifier(strategy="constant", constant=constant).fit(FEATURES, TARGET)
    with tracker.run("contract", "train") as run_id:
        uri = tracker.log_model(run_id, model, FEATURES)
    return registry.register(uri, run_id).version


def test_registry_unset_alias_None(backend: tuple[ExperimentTracker, ModelRegistry]) -> None:
    assert backend[1].get_by_alias(CHAMPION) is None


def test_registry_versions_count_from_one(backend: tuple[ExperimentTracker, ModelRegistry]) -> None:
    assert [register_model(backend), register_model(backend)] == [1, 2]


def test_registry_alias_points_at_version(backend: tuple[ExperimentTracker, ModelRegistry]) -> None:
    version = register_model(backend)

    backend[1].set_alias(CHAMPION, version)

    champion = backend[1].get_by_alias(CHAMPION)
    assert champion is not None and champion.version == version


def test_registry_alias_moves_to_new_version(backend: tuple[ExperimentTracker, ModelRegistry]) -> None:
    first, second = register_model(backend), register_model(backend)
    backend[1].set_alias(CHAMPION, first)

    backend[1].set_alias(CHAMPION, second)

    champion = backend[1].get_by_alias(CHAMPION)
    assert champion is not None and champion.version == second


def test_registry_alias_unknown_version_LookupError(backend: tuple[ExperimentTracker, ModelRegistry]) -> None:
    register_model(backend)

    with pytest.raises(LookupError):
        backend[1].set_alias(CHAMPION, 99)


def test_registry_load_model_predicts_like_logged(backend: tuple[ExperimentTracker, ModelRegistry]) -> None:
    version = register_model(backend, constant=1)

    loaded = backend[1].load_model(version)

    np.testing.assert_array_equal(loaded.predict(FEATURES), [1, 1, 1, 1])


def test_registry_version_tags_accepted(backend: tuple[ExperimentTracker, ModelRegistry]) -> None:
    version = register_model(backend)

    backend[1].set_version_tags(version, {"gate": "passed"})

    with pytest.raises(LookupError):
        backend[1].set_version_tags(version + 1, {"gate": "passed"})


def test_registry_version_details_tags_and_metrics(backend: tuple[ExperimentTracker, ModelRegistry]) -> None:
    tracker, registry = backend
    model = DummyClassifier().fit(FEATURES, TARGET)
    with tracker.run("contract", "train") as run_id:
        tracker.log_metrics(run_id, {"holdout_pr_auc": 0.62})
        tracker.set_tags(run_id, {"best_candidate": "dummy"})
        version = registry.register(tracker.log_model(run_id, model, FEATURES), run_id).version
    registry.set_version_tags(version, {"promoted": "true"})

    details = registry.version_details(version)

    assert (details.version.version, details.version.run_id) == (version, run_id)
    assert details.version_tags == {"promoted": "true"}
    assert details.run_tags == {"best_candidate": "dummy"}
    assert details.run_metrics == {"holdout_pr_auc": 0.62}


def test_registry_version_details_unknown_LookupError(backend: tuple[ExperimentTracker, ModelRegistry]) -> None:
    register_model(backend)

    with pytest.raises(LookupError):
        backend[1].version_details(99)


def test_registry_version_artifact_read_back(backend: tuple[ExperimentTracker, ModelRegistry], tmp_path: Path) -> None:
    tracker, registry = backend
    sample = tmp_path / "reference_sample.csv"
    sample.write_bytes(b"ID,Income\n1,50000\n")
    with tracker.run("contract", "train") as run_id:
        tracker.log_file(run_id, sample, "reference")
        version = registry.register(tracker.log_model(run_id, DummyClassifier().fit(FEATURES, TARGET), FEATURES),
                                    run_id).version

    assert registry.version_artifact(version, "reference/reference_sample.csv") == b"ID,Income\n1,50000\n"


@pytest.mark.parametrize(("version", "path"), [(1, "reference/missing.csv"), (99, "reference/reference_sample.csv")],
                         ids=["unknown-file", "unknown-version"])
def test_registry_version_artifact_unknown_LookupError(
    backend: tuple[ExperimentTracker, ModelRegistry], version: int, path: str
) -> None:
    register_model(backend)

    with pytest.raises(LookupError):
        backend[1].version_artifact(version, path)

"""Registry behaviour every implementation must show; parametrised so the MLflow adapter runs the same checks."""

from collections.abc import Iterator

import numpy as np
import pandas as pd
import pytest
from fakes import FakeRegistry, FakeTracker
from sklearn.dummy import DummyClassifier

from campaign_response.registry.ports import CHAMPION, ExperimentTracker, ModelRegistry

FEATURES = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
TARGET = [0, 1, 0, 1]


@pytest.fixture(params=["fake"])
def backend(request: pytest.FixtureRequest) -> Iterator[tuple[ExperimentTracker, ModelRegistry]]:
    tracker = FakeTracker()
    yield tracker, FakeRegistry(tracker)


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

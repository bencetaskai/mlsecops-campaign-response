"""MLflow registry adapter details beyond the contract: run linkage and the refusals of the safe loader."""

from collections.abc import Callable
from pathlib import Path
from typing import Self
from uuid import uuid4

import mlflow.sklearn
import numpy as np
import pandas as pd
import pytest
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR
from skops.io.exceptions import UntrustedTypesFoundException
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier

from campaign_response.registry.mlflow_adapter import MlflowRegistry, MlflowTracker
from campaign_response.registry.ports import CHAMPION, ModelRegistry

FEATURES = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
TARGET = [0, 1, 0, 1]


class Sneaky(ClassifierMixin, BaseEstimator):
    """Stands for any class outside our allow-list, as an attacker's model file would contain."""

    def fit(self, X: pd.DataFrame, y: list[int]) -> Self:
        self.classes_ = np.array([0, 1])
        return self


@pytest.fixture
def adapters(mlflow_tracking_uri: str) -> tuple[MlflowTracker, MlflowRegistry]:
    return MlflowTracker(mlflow_tracking_uri), MlflowRegistry(mlflow_tracking_uri, f"test-{uuid4().hex[:8]}")


def test_registry_conforms_to_port(adapters: tuple[MlflowTracker, MlflowRegistry]) -> None:
    assert isinstance(adapters[1], ModelRegistry)


def test_registry_version_linked_to_run(
    adapters: tuple[MlflowTracker, MlflowRegistry], mlflow_tracking_uri: str
) -> None:
    tracker, registry = adapters
    with tracker.run("registry", "train") as run_id:
        uri = tracker.log_model(run_id, DummyClassifier().fit(FEATURES, TARGET), FEATURES)

    version = registry.register(uri, run_id)
    registry.set_alias(CHAMPION, version.version)

    assert registry.get_by_alias(CHAMPION) == version
    assert version.run_id == run_id
    stored = MlflowClient(mlflow_tracking_uri).get_model_version(version.name, str(version.version))
    assert stored.run_id == run_id


def test_load_model_refuses_pickle_format(adapters: tuple[MlflowTracker, MlflowRegistry]) -> None:
    tracker, registry = adapters
    with tracker.run("registry", "pickled") as run_id:
        info = mlflow.sklearn.log_model(DummyClassifier().fit(FEATURES, TARGET), name="model",
                                        serialization_format="cloudpickle", pip_requirements=[], run_id=run_id)
    version = registry.register(info.model_uri, run_id)

    with pytest.raises(ValueError, match="only skops models are accepted"):
        registry.load_model(version.version)


def test_load_model_ignores_trust_list_in_model_file(adapters: tuple[MlflowTracker, MlflowRegistry]) -> None:
    tracker, registry = adapters
    sneaky_type = f"{Sneaky.__module__}.{Sneaky.__qualname__}"
    with tracker.run("registry", "sneaky") as run_id:
        info = mlflow.sklearn.log_model(Sneaky().fit(FEATURES, TARGET), name="model", pip_requirements=[],
                                        skops_trusted_types=[sneaky_type], run_id=run_id)
    version = registry.register(info.model_uri, run_id)

    assert isinstance(mlflow.sklearn.load_model(info.model_uri), Sneaky)
    with pytest.raises(UntrustedTypesFoundException, match="Sneaky"):
        registry.load_model(version.version)


def test_load_model_ignores_global_tracking_uri(
    adapters: tuple[MlflowTracker, MlflowRegistry], mlflow_tracking_uri: str, tmp_path: Path
) -> None:
    """The API process has no tracker setting the global URI; loading must use the registry's own URI."""
    tracker, registry = adapters
    with tracker.run("registry", "elsewhere") as run_id:
        uri = tracker.log_model(run_id, DummyClassifier().fit(FEATURES, TARGET), FEATURES)
    version = registry.register(uri, run_id)

    mlflow.set_tracking_uri(f"sqlite:///{(tmp_path / 'other.db').as_posix()}")
    try:
        model = registry.load_model(version.version)
    finally:
        mlflow.set_tracking_uri(mlflow_tracking_uri)

    assert isinstance(model, DummyClassifier)


def test_load_model_unknown_version_LookupError(adapters: tuple[MlflowTracker, MlflowRegistry]) -> None:
    tracker, registry = adapters
    with tracker.run("registry", "one") as run_id:
        registry.register(tracker.log_model(run_id, DummyClassifier().fit(FEATURES, TARGET), FEATURES), run_id)

    with pytest.raises(LookupError, match="unknown version 7"):
        registry.load_model(7)


@pytest.mark.parametrize(
    ("client_method", "call"),
    [
        ("get_registered_model", lambda registry: registry.get_by_alias(CHAMPION)),
        ("create_registered_model", lambda registry: registry.register("models:/m-1", "run-1")),
        ("set_registered_model_alias", lambda registry: registry.set_alias(CHAMPION, 1)),
    ],
)
def test_registry_other_mlflow_errors_propagate(
    adapters: tuple[MlflowTracker, MlflowRegistry],
    monkeypatch: pytest.MonkeyPatch,
    client_method: str,
    call: Callable[[MlflowRegistry], object],
) -> None:
    """A server failure must surface, not be mistaken for "no champion" or "unknown version"."""
    registry = adapters[1]

    def fail(*args: object, **kwargs: object) -> None:
        raise MlflowException("database unavailable", INTERNAL_ERROR)

    monkeypatch.setattr(registry._client, client_method, fail)

    with pytest.raises(MlflowException, match="database unavailable"):
        call(registry)

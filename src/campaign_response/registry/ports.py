"""The two interfaces the training pipeline and the API use for MLflow, as structural Protocols.

Business code depends on these, never on mlflow directly: tests pass in-memory fakes, the application passes the
MLflow adapters, and the gate and pipeline logic can be tested without a server. The registry is bound to one
registered model name when it is created, so callers deal in versions and aliases only.
"""

from collections.abc import Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import pandas as pd
from matplotlib.figure import Figure
from numpy.typing import NDArray

CHAMPION = "champion"
CHALLENGER = "challenger"
PREVIOUS = "previous"


class Classifier(Protocol):
    """What the gate and the API need from a loaded model; the tuned sklearn pipeline satisfies it."""

    def predict_proba(self, X: pd.DataFrame) -> NDArray[Any]: ...

    def predict(self, X: pd.DataFrame) -> NDArray[Any]: ...


@dataclass(frozen=True)
class ModelVersion:
    name: str
    version: int
    run_id: str

    def __post_init__(self) -> None:
        if self.version < 1:
            raise ValueError(f"model versions start at 1, got {self.version}")


@runtime_checkable
class ExperimentTracker(Protocol):
    def run(
        self, experiment: str, run_name: str, parent_run_id: str | None = None
    ) -> AbstractContextManager[str]:
        """Open a run and yield its id; the run is marked failed if the block raises, finished otherwise."""
        ...

    def log_params(self, run_id: str, params: Mapping[str, Any]) -> None: ...

    def log_metrics(self, run_id: str, metrics: Mapping[str, float]) -> None: ...

    def set_tags(self, run_id: str, tags: Mapping[str, str]) -> None: ...

    def log_dict(self, run_id: str, data: Mapping[str, Any], artifact_file: str) -> None: ...

    def log_figure(self, run_id: str, figure: Figure, artifact_file: str) -> None: ...

    def log_file(self, run_id: str, local_path: Path, artifact_path: str | None = None) -> None: ...

    def log_model(self, run_id: str, model: Any, input_example: pd.DataFrame) -> str:
        """Store the fitted model in the run and return its URI for registration.

        Any, not Classifier: sklearn exposes predict_proba through a descriptor that exists only when the wrapped
        estimator supports it, which a static Protocol cannot match.
        """
        ...


@runtime_checkable
class ModelRegistry(Protocol):
    def register(self, model_uri: str, run_id: str) -> ModelVersion: ...

    def set_alias(self, alias: str, version: int) -> None:
        """Point the alias at the version, moving it away from any other version."""
        ...

    def get_by_alias(self, alias: str) -> ModelVersion | None:
        """The version the alias points at, or None if the alias is not set."""
        ...

    def set_version_tags(self, version: int, tags: Mapping[str, str]) -> None: ...

    def load_model(self, version: int) -> Classifier: ...

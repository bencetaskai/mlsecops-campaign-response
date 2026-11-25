"""Tests for the tracker and registry interfaces: what counts as an implementation and the value object rules.

The stubs only need the method names: runtime_checkable Protocols check that the methods exist, not their signatures;
Pyright checks the signatures of the MLflow adapters in src/ wherever they are passed as these types.
"""

from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

import pytest

from campaign_response.registry.ports import (
    CHALLENGER,
    CHAMPION,
    PREVIOUS,
    ExperimentTracker,
    ModelRegistry,
    ModelVersion,
)


class CompleteTracker:
    def run(self, experiment: str, run_name: str, parent_run_id: str | None = None) -> AbstractContextManager[str]: ...

    def log_params(self, run_id: str, params: Any) -> None: ...
    def log_metrics(self, run_id: str, metrics: Any) -> None: ...
    def set_tags(self, run_id: str, tags: Any) -> None: ...
    def log_dict(self, run_id: str, data: Any, artifact_file: str) -> None: ...
    def log_figure(self, run_id: str, figure: Any, artifact_file: str) -> None: ...
    def log_file(self, run_id: str, local_path: Path, artifact_path: str | None = None) -> None: ...
    def log_model(self, run_id: str, model: Any, input_example: Any) -> str: ...


class CompleteRegistry:
    def register(self, model_uri: str, run_id: str) -> ModelVersion: ...
    def set_alias(self, alias: str, version: int) -> None: ...
    def get_by_alias(self, alias: str) -> ModelVersion | None: ...
    def set_version_tags(self, version: int, tags: Any) -> None: ...
    def version_details(self, version: int) -> Any: ...
    def version_artifact(self, version: int, artifact_path: str) -> bytes: ...
    def load_model(self, version: int) -> Any: ...


def test_ports_complete_classes_conform() -> None:
    assert isinstance(CompleteTracker(), ExperimentTracker)
    assert isinstance(CompleteRegistry(), ModelRegistry)


def test_ports_missing_method_not_conform() -> None:
    assert not isinstance(object(), ExperimentTracker)
    assert not isinstance(CompleteTracker(), ModelRegistry)


def test_model_version_valid_fields() -> None:
    version = ModelVersion("campaign-response-classifier", 3, "abc")

    assert (version.name, version.version, version.run_id) == ("campaign-response-classifier", 3, "abc")


@pytest.mark.parametrize("number", [0, -1])
def test_model_version_below_one_ValueError(number: int) -> None:
    with pytest.raises(ValueError, match="start at 1"):
        ModelVersion("m", number, "abc")


def test_aliases_distinct_lowercase() -> None:
    aliases = (CHAMPION, CHALLENGER, PREVIOUS)

    assert len(set(aliases)) == 3
    assert all(alias.islower() for alias in aliases)

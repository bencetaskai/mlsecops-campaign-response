"""In-memory test doubles for the tracker and registry ports, used wherever a test needs MLflow behaviour.

They follow MLflow's rules that matter to our code: params cannot change once logged, a failed block marks the run
failed, versions count up from 1, and an alias points at exactly one existing version. The contract tests in
tests/registry run the same checks against these fakes and against a real local MLflow.
"""

import json
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from matplotlib.figure import Figure

from campaign_response.registry.ports import Classifier, ModelVersion

RUNNING, FINISHED, FAILED = "RUNNING", "FINISHED", "FAILED"


@dataclass
class FakeRun:
    run_id: str
    experiment: str
    name: str
    parent_run_id: str | None
    status: str = RUNNING
    params: dict[str, str] = field(default_factory=dict[str, str])
    metrics: dict[str, float] = field(default_factory=dict[str, float])
    tags: dict[str, str] = field(default_factory=dict[str, str])
    artifacts: dict[str, bytes] = field(default_factory=dict[str, bytes])
    model: Classifier | None = None


class FakeTracker:
    def __init__(self) -> None:
        self.runs: dict[str, FakeRun] = {}

    @contextmanager
    def run(self, experiment: str, run_name: str, parent_run_id: str | None = None) -> Iterator[str]:
        if parent_run_id is not None:
            self._get(parent_run_id)
        run = FakeRun(f"run-{len(self.runs) + 1}", experiment, run_name, parent_run_id)
        self.runs[run.run_id] = run
        try:
            yield run.run_id
        except BaseException:
            run.status = FAILED
            raise
        run.status = FINISHED

    def log_params(self, run_id: str, params: Mapping[str, Any]) -> None:
        run = self._get(run_id)
        for key, value in params.items():
            # MLflow stores params as strings and refuses to change a logged value.
            if key in run.params and run.params[key] != str(value):
                raise ValueError(f"param {key!r} already logged as {run.params[key]!r}")
            run.params[key] = str(value)

    def log_metrics(self, run_id: str, metrics: Mapping[str, float]) -> None:
        self._get(run_id).metrics.update({key: float(value) for key, value in metrics.items()})

    def set_tags(self, run_id: str, tags: Mapping[str, str]) -> None:
        self._get(run_id).tags.update({key: str(value) for key, value in tags.items()})

    def log_dict(self, run_id: str, data: Mapping[str, Any], artifact_file: str) -> None:
        self._get(run_id).artifacts[artifact_file] = json.dumps(data, indent=2).encode()

    def log_figure(self, run_id: str, figure: Figure, artifact_file: str) -> None:
        self._get(run_id).artifacts[artifact_file] = b"figure"

    def log_file(self, run_id: str, local_path: Path, artifact_path: str | None = None) -> None:
        name = f"{artifact_path}/{local_path.name}" if artifact_path else local_path.name
        self._get(run_id).artifacts[name] = local_path.read_bytes()

    def log_model(self, run_id: str, model: Classifier, input_example: pd.DataFrame) -> str:
        self._get(run_id).model = model
        return f"runs:/{run_id}/model"

    def json_artifact(self, run_id: str, artifact_file: str) -> Any:
        return json.loads(self.runs[run_id].artifacts[artifact_file])

    def children(self, parent_run_id: str) -> list[FakeRun]:
        return [run for run in self.runs.values() if run.parent_run_id == parent_run_id]

    def _get(self, run_id: str) -> FakeRun:
        if run_id not in self.runs:
            raise KeyError(f"unknown run {run_id!r}")
        return self.runs[run_id]


class FakeRegistry:
    """Bound to one model name like the real adapter; models are looked up in the tracker that logged them."""

    def __init__(self, tracker: FakeTracker, name: str = "campaign-response-classifier") -> None:
        self.tracker = tracker
        self.name = name
        self.versions: list[ModelVersion] = []
        self.aliases: dict[str, int] = {}
        self.version_tags: dict[int, dict[str, str]] = {}

    def register(self, model_uri: str, run_id: str) -> ModelVersion:
        if self.tracker.runs[run_id].model is None:
            raise ValueError(f"run {run_id!r} has no logged model")
        version = ModelVersion(self.name, len(self.versions) + 1, run_id)
        self.versions.append(version)
        self.version_tags[version.version] = {}
        return version

    def set_alias(self, alias: str, version: int) -> None:
        self._get(version)
        self.aliases[alias] = version

    def get_by_alias(self, alias: str) -> ModelVersion | None:
        return self._get(self.aliases[alias]) if alias in self.aliases else None

    def set_version_tags(self, version: int, tags: Mapping[str, str]) -> None:
        self._get(version)
        self.version_tags[version].update({key: str(value) for key, value in tags.items()})

    def load_model(self, version: int) -> Classifier:
        model = self.tracker.runs[self._get(version).run_id].model
        assert model is not None
        return model

    def _get(self, version: int) -> ModelVersion:
        if not 1 <= version <= len(self.versions):
            raise KeyError(f"unknown version {version} of {self.name!r}")
        return self.versions[version - 1]

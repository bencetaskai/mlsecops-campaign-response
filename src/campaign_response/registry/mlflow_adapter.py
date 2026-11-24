"""MLflow implementations of the tracker and registry ports.

Runs are addressed by id through MlflowClient instead of mlflow.start_run(): the fluent API keeps the active run in
global state, which two concurrent API requests would share.
"""

from collections.abc import Generator, Mapping
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from matplotlib.figure import Figure
from mlflow import MlflowClient
from mlflow.entities import Metric, Param, RunStatus, RunTag
from mlflow.sklearn import log_model

from campaign_response.registry.ports import Classifier

MODEL_ARTIFACT_NAME = "model"

# Declared instead of inferred: inference loads the model in a subprocess (8-9 s per model) and records whatever the
# logging process had imported, such as pytest. The feature code itself ships in our image, not as a pip package.
MODEL_REQUIREMENTS = ("numpy", "pandas", "scikit-learn", "scipy", "skops")

# skops rebuilds only these types when a model is loaded, so a tampered model file cannot run code the way a pickle
# can. A fixed list, never derived from the file being loaded; a test checks that every candidate fits within it.
SKOPS_TRUSTED_TYPES = (
    "campaign_response.features.transformer.FeatureEngineer",
    "datetime.date",
    "numpy.dtype",
    "sklearn.compose._column_transformer.make_column_selector",
    "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
    "sklearn.metrics._classification.f1_score",
    "sklearn.metrics._scorer._CurveScorer",
    "sklearn.metrics._scorer._Scorer",
    "sklearn.model_selection._split.StratifiedKFold",
    "sklearn.tree._tree.Tree",
    "sklearn.utils._metadata_requests.MetadataRequest",
    "sklearn.utils._metadata_requests.MethodMetadataRequest",
)


class MlflowTracker:
    def __init__(self, tracking_uri: str) -> None:
        # log_model has no client parameter and always uses the process-wide URI; one tracking server per process.
        mlflow.set_tracking_uri(tracking_uri)
        self._client = MlflowClient(tracking_uri)

    @contextmanager
    def run(self, experiment: str, run_name: str, parent_run_id: str | None = None) -> Generator[str]:
        tags = {"mlflow.parentRunId": parent_run_id} if parent_run_id else None
        run = self._client.create_run(self._experiment_id(experiment), run_name=run_name, tags=tags)
        run_id = run.info.run_id
        try:
            yield run_id
        except BaseException:
            self._client.set_terminated(run_id, RunStatus.to_string(RunStatus.FAILED))
            raise
        self._client.set_terminated(run_id, RunStatus.to_string(RunStatus.FINISHED))

    def log_params(self, run_id: str, params: Mapping[str, Any]) -> None:
        self._client.log_batch(run_id, params=[Param(key, str(value)) for key, value in params.items()])

    def log_metrics(self, run_id: str, metrics: Mapping[str, float]) -> None:
        timestamp = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
        batch = [Metric(key, float(value), timestamp, 0) for key, value in metrics.items()]
        self._client.log_batch(run_id, metrics=batch)

    def set_tags(self, run_id: str, tags: Mapping[str, str]) -> None:
        self._client.log_batch(run_id, tags=[RunTag(key, str(value)) for key, value in tags.items()])

    def log_dict(self, run_id: str, data: Mapping[str, Any], artifact_file: str) -> None:
        self._client.log_dict(run_id, dict(data), artifact_file)

    def log_figure(self, run_id: str, figure: Figure, artifact_file: str) -> None:
        self._client.log_figure(run_id, figure, artifact_file)

    def log_file(self, run_id: str, local_path: Path, artifact_path: str | None = None) -> None:
        self._client.log_artifact(run_id, str(local_path), artifact_path)

    def log_model(self, run_id: str, model: Classifier, input_example: pd.DataFrame) -> str:
        """Log as skops with the signature inferred from the example; returns the models:/ URI to register."""
        info = log_model(
            model,
            name=MODEL_ARTIFACT_NAME,
            input_example=input_example,
            skops_trusted_types=list(SKOPS_TRUSTED_TYPES),
            pip_requirements=[f"{package}=={version(package)}" for package in MODEL_REQUIREMENTS],
            run_id=run_id,
        )
        return info.model_uri

    def _experiment_id(self, name: str) -> str:
        experiment = self._client.get_experiment_by_name(name)
        return experiment.experiment_id if experiment is not None else self._client.create_experiment(name)

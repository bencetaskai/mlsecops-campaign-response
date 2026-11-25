"""MLflow implementations of the tracker and registry ports.

Runs are addressed by id through MlflowClient instead of mlflow.start_run(): the fluent API keeps the active run in
global state, which two concurrent API requests would share.
"""

import tempfile
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path
from typing import Any

import mlflow
import mlflow.artifacts
import pandas as pd
import skops.io
from matplotlib.figure import Figure
from mlflow import MlflowClient
from mlflow.entities import Metric, Param, RunStatus, RunTag
from mlflow.exceptions import MlflowException
from mlflow.models import Model
from mlflow.sklearn import log_model

from campaign_response.registry.ports import Classifier, ModelVersion, VersionDetails

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

    def log_model(self, run_id: str, model: Any, input_example: pd.DataFrame) -> str:
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
        if experiment is not None:
            return experiment.experiment_id
        try:
            return self._client.create_experiment(name)
        except MlflowException as error:
            # Lost a race: parallel /predict background tasks both found no experiment and both tried to create it.
            if error.error_code != "RESOURCE_ALREADY_EXISTS":
                raise
        created = self._client.get_experiment_by_name(name)
        if created is None:
            raise LookupError(f"experiment {name!r} reported as existing but not found") from None
        return created.experiment_id


class MlflowRegistry:
    """The ModelRegistry port for one registered model name."""

    def __init__(self, tracking_uri: str, model_name: str) -> None:
        self._tracking_uri = tracking_uri
        self._client = MlflowClient(tracking_uri)
        self._name = model_name

    def register(self, model_uri: str, run_id: str) -> ModelVersion:
        self._ensure_registered_model()
        created = self._client.create_model_version(
            self._name, source=model_uri, run_id=run_id, model_id=model_uri.removeprefix("models:/")
        )
        return ModelVersion(self._name, int(created.version), run_id)

    def set_alias(self, alias: str, version: int) -> None:
        with _version_must_exist(self._name, version):
            self._client.set_registered_model_alias(self._name, alias, str(version))

    def get_by_alias(self, alias: str) -> ModelVersion | None:
        # Read from the alias map: an unset alias is a normal state, and MLflow reports it only through a generic
        # INVALID_PARAMETER_VALUE error that would have to be told apart by its message.
        try:
            aliases = self._client.get_registered_model(self._name).aliases
        except MlflowException as error:
            if error.error_code == "RESOURCE_DOES_NOT_EXIST":
                return None
            raise
        return self._version(int(aliases[alias])) if alias in aliases else None

    def set_version_tags(self, version: int, tags: Mapping[str, str]) -> None:
        with _version_must_exist(self._name, version):
            for key, value in tags.items():
                self._client.set_model_version_tag(self._name, str(version), key, str(value))

    def version_details(self, version: int) -> VersionDetails:
        # Every version is registered from a run (register() requires one), so the run always exists.
        with _version_must_exist(self._name, version):
            found = self._client.get_model_version(self._name, str(version))
        run_id = found.run_id or ""
        run = self._client.get_run(run_id)
        return VersionDetails(
            version=ModelVersion(self._name, int(found.version), run_id),
            version_tags=dict(found.tags),
            # MLflow's own run tags (mlflow.user, mlflow.source.name, ...) describe the client, not the model.
            run_tags={key: value for key, value in run.data.tags.items() if not key.startswith("mlflow.")},
            run_metrics=dict(run.data.metrics),
        )

    def version_artifact(self, version: int, artifact_path: str) -> bytes:
        run_id = self._version(version).run_id
        # Listed first: through the server's artifact proxy a missing file is an HTTP 500, which the MLflow client
        # retries with backoff for about 4.5 minutes before giving up (measured against the compose stack).
        parent = artifact_path.rpartition("/")[0] or None
        if artifact_path not in {item.path for item in self._client.list_artifacts(run_id, parent)}:
            raise LookupError(f"version {version} has no artifact {artifact_path!r}")
        with tempfile.TemporaryDirectory() as folder:
            return Path(self._client.download_artifacts(run_id, artifact_path, folder)).read_bytes()

    def load_model(self, version: int) -> Classifier:
        """Load the skops file with our own trusted types, never the list stored in the model's MLmodel file.

        mlflow.sklearn.load_model trusts the list saved next to the model, so whoever can replace the model can also
        widen that list. Pickle-based models are refused outright.
        """
        self._version(version)
        # Resolved with our own client, not passed as models:/name/version: MLflow 3.18 resolves that form with a
        # default client on the process-wide registry URI, ignoring the URI given here. It worked only where
        # MlflowTracker had set the global URI, and failed in the API, which has no tracker.
        storage_uri = self._client.get_model_version_download_uri(self._name, str(version))
        with tempfile.TemporaryDirectory() as folder:
            local = Path(mlflow.artifacts.download_artifacts(
                artifact_uri=storage_uri, dst_path=folder, tracking_uri=self._tracking_uri
            ))
            flavor = Model.load(str(local / "MLmodel")).flavors["sklearn"]
            if flavor.get("serialization_format") != "skops":
                raise ValueError(
                    f"refusing to load version {version}: serialization {flavor.get('serialization_format')!r} "
                    "can run arbitrary code; only skops models are accepted"
                )
            return skops.io.load(local / flavor["pickled_model"], trusted=list(SKOPS_TRUSTED_TYPES))

    def _version(self, version: int) -> ModelVersion:
        with _version_must_exist(self._name, version):
            found = self._client.get_model_version(self._name, str(version))
        return ModelVersion(self._name, int(found.version), found.run_id or "")

    def _ensure_registered_model(self) -> None:
        try:
            self._client.create_registered_model(self._name)
        except MlflowException as error:
            if error.error_code != "RESOURCE_ALREADY_EXISTS":
                raise


@contextmanager
def _version_must_exist(name: str, version: int) -> Generator[None]:
    """Translate MLflow's not-found error into the LookupError the port promises."""
    try:
        yield
    except MlflowException as error:
        if error.error_code == "RESOURCE_DOES_NOT_EXIST":
            raise LookupError(f"unknown version {version} of {name!r}") from error
        raise

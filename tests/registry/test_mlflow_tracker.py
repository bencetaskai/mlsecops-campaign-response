"""The MLflow tracker adapter against a real local MLflow (store from tests/registry/conftest.py)."""

import json
from pathlib import Path

import mlflow
import mlflow.artifacts
import mlflow.models
import mlflow.sklearn
import numpy as np
import pandas as pd
import pytest
import skops.io
from matplotlib.figure import Figure
from mlflow import MlflowClient

from campaign_response.modeling.candidates import CANDIDATE_NAMES, get_candidate
from campaign_response.modeling.training import fit_with_tuned_threshold
from campaign_response.registry.mlflow_adapter import MODEL_REQUIREMENTS, SKOPS_TRUSTED_TYPES, MlflowTracker
from campaign_response.registry.ports import ExperimentTracker


@pytest.fixture(scope="module")
def tracker(mlflow_tracking_uri: str) -> MlflowTracker:
    return MlflowTracker(mlflow_tracking_uri)


@pytest.fixture(scope="module")
def client(mlflow_tracking_uri: str) -> MlflowClient:
    return MlflowClient(mlflow_tracking_uri)


@pytest.fixture(scope="module")
def data(customers: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return customers.drop(columns=["ID", "Response"]), customers["Response"].astype(int)


def test_tracker_conforms_to_port(tracker: MlflowTracker) -> None:
    assert isinstance(tracker, ExperimentTracker)


def test_tracker_run_finished_in_named_experiment(tracker: MlflowTracker, client: MlflowClient) -> None:
    with tracker.run("training", "first") as first, tracker.run("training", "second") as second:
        pass

    runs = [client.get_run(run_id) for run_id in (first, second)]
    assert [run.info.status for run in runs] == ["FINISHED", "FINISHED"]
    assert runs[0].info.experiment_id == runs[1].info.experiment_id
    assert client.get_experiment(runs[0].info.experiment_id).name == "training"
    assert runs[0].info.run_name == "first"


def test_tracker_run_failed_and_error_reraised(tracker: MlflowTracker, client: MlflowClient) -> None:
    started: list[str] = []

    with pytest.raises(RuntimeError, match="boom"), tracker.run("training", "crash") as run_id:
        started.append(run_id)
        raise RuntimeError("boom")

    assert client.get_run(started[0]).info.status == "FAILED"


def test_tracker_nested_run_has_parent(tracker: MlflowTracker, client: MlflowClient) -> None:
    with tracker.run("training", "parent") as parent, tracker.run("training", "child", parent) as child:
        pass

    assert client.get_run(child).data.tags["mlflow.parentRunId"] == parent


def test_tracker_params_metrics_tags_logged(tracker: MlflowTracker, client: MlflowClient) -> None:
    with tracker.run("training", "values") as run_id:
        tracker.log_params(run_id, {"seed": 42, "candidates": ["logistic_regression"]})
        tracker.log_metrics(run_id, {"pr_auc": 0.61, "n_rows": 1550})
        tracker.set_tags(run_id, {"promoted": "false"})

    data = client.get_run(run_id).data
    assert data.params == {"seed": "42", "candidates": "['logistic_regression']"}
    assert data.metrics == {"pr_auc": 0.61, "n_rows": 1550.0}
    assert data.tags["promoted"] == "false"


def test_tracker_artifacts_logged(tracker: MlflowTracker, client: MlflowClient, tmp_path: Path) -> None:
    local = tmp_path / "reference.csv"
    local.write_text("a\n1\n", encoding="utf-8")

    with tracker.run("training", "artifacts") as run_id:
        tracker.log_dict(run_id, {"promoted": True}, "gate_decision.json")
        tracker.log_figure(run_id, Figure(), "plots/pr_curve.png")
        tracker.log_file(run_id, local, artifact_path="reference")

    names = {item.path for folder in ("", "plots", "reference") for item in client.list_artifacts(run_id, folder)}
    assert {"gate_decision.json", "plots/pr_curve.png", "reference/reference.csv"} <= names
    decision = client.download_artifacts(run_id, "gate_decision.json", str(tmp_path))
    assert json.loads(Path(decision).read_text(encoding="utf-8")) == {"promoted": True}


def test_tracker_log_model_loads_with_same_predictions(
    tracker: MlflowTracker, data: tuple[pd.DataFrame, pd.Series]
) -> None:
    features, target = data
    model = fit_with_tuned_threshold(get_candidate("logistic_regression"), {}, features, target, cv_folds=3, seed=1)

    with tracker.run("training", "model") as run_id:
        uri = tracker.log_model(run_id, model, features.head(5))

    flavor = mlflow.models.get_model_info(uri).flavors["sklearn"]
    assert uri.startswith("models:/")
    assert flavor["serialization_format"] == "skops"
    assert flavor["skops_trusted_types"] == list(SKOPS_TRUSTED_TYPES)
    requirements = (Path(mlflow.artifacts.download_artifacts(uri)) / "requirements.txt").read_text(encoding="utf-8")
    assert {line.split("==")[0] for line in requirements.split()} == {"mlflow", *MODEL_REQUIREMENTS}
    loaded = mlflow.sklearn.load_model(uri)
    np.testing.assert_array_equal(loaded.predict_proba(features), model.predict_proba(features))


@pytest.mark.parametrize("name", [name for name in CANDIDATE_NAMES if name != "dummy"])
def test_skops_trusted_types_cover_candidate(name: str, data: tuple[pd.DataFrame, pd.Series]) -> None:
    model = fit_with_tuned_threshold(get_candidate(name), {}, *data, cv_folds=3, seed=1)

    untrusted = skops.io.get_untrusted_types(data=skops.io.dumps(model))

    assert set(untrusted) <= set(SKOPS_TRUSTED_TYPES)

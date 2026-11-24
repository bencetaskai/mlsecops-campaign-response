"""The training pipeline against a real local MLflow: three runs through champion, rejection and promotion.

The fakes cover every branch quickly; this checks that the real adapters accept everything the pipeline sends and
that the aliases end up right in an actual registry.
"""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from mlflow import MlflowClient

from campaign_response.params import load_params
from campaign_response.pipeline.train_pipeline import TrainingContext, TrainingOutcome, TrainingRequest, run_training
from campaign_response.registry.mlflow_adapter import MlflowRegistry, MlflowTracker
from campaign_response.registry.ports import CHALLENGER, CHAMPION, PREVIOUS
from campaign_response.registry.promotion import GateParams

pytestmark = pytest.mark.integration

REPO = Path(__file__).resolve().parents[2]
MODEL_NAME = "campaign-response-classifier"
REQUEST = TrainingRequest(candidates=("logistic_regression",), n_iter=1, cv_folds=3, seed=1, trained_by="cli")


@pytest.fixture(scope="module")
def runs(mlflow_tracking_uri: str, raw_customers_file: Path) -> tuple[TrainingContext, list[TrainingOutcome]]:
    """First champion; a challenger the gate rejects (recall minimum 1.0); an equal retrain that is promoted."""
    context = TrainingContext(
        tracker=MlflowTracker(mlflow_tracking_uri),
        registry=MlflowRegistry(mlflow_tracking_uri, MODEL_NAME),
        params=load_params(REPO / "params.yaml"),
        raw_data_path=raw_customers_file,
        requirements_path=REPO / "requirements.txt",
        environ={"GIT_SHA": "integration"},
    )
    strict = replace(context, params=replace(context.params, gate=GateParams(min_recall=1.0)))
    outcomes = [run_training(REQUEST, context), run_training(REQUEST, strict), run_training(REQUEST, context)]
    return context, outcomes


@pytest.fixture(scope="module")
def client(mlflow_tracking_uri: str) -> MlflowClient:
    return MlflowClient(mlflow_tracking_uri)


def test_mlflow_gate_outcomes_in_order(runs: tuple[TrainingContext, list[TrainingOutcome]]) -> None:
    assert [outcome.promoted for outcome in runs[1]] == [True, False, True]
    assert [outcome.version.version for outcome in runs[1]] == [1, 2, 3]


def test_mlflow_aliases_after_three_runs(
    client: MlflowClient, runs: tuple[TrainingContext, list[TrainingOutcome]]
) -> None:
    aliases = client.get_registered_model(MODEL_NAME).aliases

    assert {alias: int(version) for alias, version in aliases.items()} == {CHAMPION: 3, PREVIOUS: 1, CHALLENGER: 3}


def test_mlflow_version_tags_record_gate(
    client: MlflowClient, runs: tuple[TrainingContext, list[TrainingOutcome]]
) -> None:
    tags = [client.get_model_version(MODEL_NAME, str(number)).tags for number in (1, 2, 3)]

    assert [version["promoted"] for version in tags] == ["true", "false", "true"]
    assert tags[1]["gate_reason"].startswith("recall:")


def test_mlflow_parent_run_complete(client: MlflowClient, runs: tuple[TrainingContext, list[TrainingOutcome]]) -> None:
    outcome = runs[1][1]

    run = client.get_run(outcome.run_id)
    folders = ("", "plots", "reference")
    artifacts = {item.path for folder in folders for item in client.list_artifacts(outcome.run_id, folder)}
    children = client.search_runs([run.info.experiment_id], f"tags.mlflow.parentRunId = '{outcome.run_id}'")

    assert run.info.status == "FINISHED"
    assert run.data.tags["git_sha"] == "integration"
    assert run.data.tags["promoted"] == "false" and run.data.tags["champion_version"] == "1"
    assert {"gate_decision.json", "reference/reference_sample.csv", "plots/pr_curve.png"} <= artifacts
    assert sorted(child.info.run_name for child in children) == ["dummy", "logistic_regression"]
    assert run.data.metrics["holdout_pr_auc"] == pytest.approx(outcome.holdout.pr_auc)


def test_mlflow_gate_decision_readable(
    client: MlflowClient, runs: tuple[TrainingContext, list[TrainingOutcome]], tmp_path: Path
) -> None:
    outcome = runs[1][1]

    path = client.download_artifacts(outcome.run_id, "gate_decision.json", str(tmp_path))

    assert json.loads(Path(path).read_text(encoding="utf-8")) == outcome.decision.as_dict()


def test_mlflow_champion_loads_and_predicts(
    runs: tuple[TrainingContext, list[TrainingOutcome]], customers: pd.DataFrame
) -> None:
    context, outcomes = runs
    champion = context.registry.get_by_alias(CHAMPION)
    assert champion is not None

    model = context.registry.load_model(champion.version)

    assert model.best_threshold_ == pytest.approx(outcomes[2].threshold)
    assert np.isin(model.predict(customers.drop(columns=["ID", "Response"])), [0, 1]).all()

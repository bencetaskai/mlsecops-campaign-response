"""The training run end to end on the synthetic raw file, recorded by the in-memory tracker and registry."""

from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
from fakes import FAILED, FINISHED, FakeRegistry, FakeTracker

from campaign_response.data.cleaning import CleaningParams, clean
from campaign_response.data.io import load_processed
from campaign_response.data.splitting import SplitParams, split_train_holdout
from campaign_response.params import PipelineParams
from campaign_response.pipeline.train_pipeline import (
    EXPERIMENT,
    DataValidationError,
    TrainingContext,
    TrainingOutcome,
    TrainingRequest,
    flatten,
    run_training,
)
from campaign_response.registry.ports import CHALLENGER, CHAMPION, PREVIOUS
from campaign_response.registry.promotion import GateParams

REQUIREMENTS = Path(__file__).resolve().parents[2] / "requirements.txt"
REQUEST = TrainingRequest(candidates=("logistic_regression",), n_iter=1, cv_folds=3, seed=1, trained_by="cli")


def make_context(raw_file: Path) -> TrainingContext:
    tracker = FakeTracker()
    return TrainingContext(
        tracker=tracker,
        registry=FakeRegistry(tracker),
        params=PipelineParams(CleaningParams(), SplitParams(), GateParams()),
        raw_data_path=raw_file,
        requirements_path=REQUIREMENTS,
        environ={"GIT_SHA": "abc123"},
    )


@pytest.fixture(scope="module")
def trained(raw_customers_file: Path) -> tuple[TrainingContext, TrainingOutcome]:
    context = make_context(raw_customers_file)
    return context, run_training(REQUEST, context)


def tracker_of(context: TrainingContext) -> FakeTracker:
    assert isinstance(context.tracker, FakeTracker)
    return context.tracker


def test_training_registers_challenger(trained: tuple[TrainingContext, TrainingOutcome]) -> None:
    context, outcome = trained

    assert outcome.version.version == 1
    assert context.registry.get_by_alias(CHALLENGER) == outcome.version
    assert outcome.version.run_id == outcome.run_id
    assert outcome.best_candidate == "logistic_regression"
    assert 0 < outcome.threshold < 1


def test_training_parent_run_finished_with_lineage(trained: tuple[TrainingContext, TrainingOutcome]) -> None:
    context, outcome = trained

    run = tracker_of(context).runs[outcome.run_id]

    assert (run.status, run.experiment, run.name) == (FINISHED, EXPERIMENT, "train-cli")
    assert run.tags["git_sha"] == "abc123"
    assert run.tags["trained_by"] == "cli"
    assert {"data_sha256", "dvc_md5", "params_sha256", "requirements_sha256"} <= set(run.tags)
    assert (run.tags["best_candidate"], run.tags["model_version"]) == ("logistic_regression", "1")


def test_training_params_flattened(trained: tuple[TrainingContext, TrainingOutcome]) -> None:
    context, outcome = trained

    params = tracker_of(context).runs[outcome.run_id].params

    assert params["request.n_iter"] == "1"
    assert params["gate.min_recall"] == "0.3"
    assert params["cleaning.reference_date"] == "2014-12-31"
    assert params["split.holdout_size"] == "0.25"


def test_training_metrics_cover_all_rows(trained: tuple[TrainingContext, TrainingOutcome]) -> None:
    context, outcome = trained
    tracker = tracker_of(context)

    metrics = tracker.runs[outcome.run_id].metrics
    cleaned_rows = tracker.json_artifact(outcome.run_id, "cleaning_report.json")["rows_out"]

    assert metrics["train_rows"] + metrics["holdout_rows"] == cleaned_rows
    assert metrics["holdout_n_samples"] == metrics["holdout_rows"] == outcome.holdout.n_samples
    assert metrics["holdout_pr_auc"] == outcome.holdout.pr_auc
    assert metrics["cv_pr_auc"] > metrics["baseline_cv_pr_auc"]


def test_training_artifacts_logged(trained: tuple[TrainingContext, TrainingOutcome]) -> None:
    context, outcome = trained

    artifacts = set(tracker_of(context).runs[outcome.run_id].artifacts)

    assert artifacts == {
        "validation_report.json", "cleaning_report.json", "cv_results.json", "gate_decision.json",
        "reference/reference_sample.csv",
        "plots/pr_curve.png", "plots/roc_curve.png", "plots/confusion_matrix.png", "plots/calibration.png",
    }
    assert tracker_of(context).runs[outcome.run_id].model is not None


def test_training_reference_sample_is_train_split(
    trained: tuple[TrainingContext, TrainingOutcome], raw_customers: pd.DataFrame, tmp_path: Path
) -> None:
    context, outcome = trained
    stored = tmp_path / "reference_sample.csv"
    stored.write_bytes(tracker_of(context).runs[outcome.run_id].artifacts["reference/reference_sample.csv"])

    train, _ = split_train_holdout(clean(raw_customers)[0], context.params.split)

    pd.testing.assert_frame_equal(load_processed(stored), train.reset_index(drop=True))


def test_training_gate_decision_logged(trained: tuple[TrainingContext, TrainingOutcome]) -> None:
    context, outcome = trained

    logged = tracker_of(context).json_artifact(outcome.run_id, "gate_decision.json")

    assert logged == outcome.decision.as_dict()
    assert logged["promoted"] is True and logged["champion"] is None


def test_training_child_run_per_candidate(trained: tuple[TrainingContext, TrainingOutcome]) -> None:
    context, outcome = trained
    tracker = tracker_of(context)

    children = {run.name: run for run in tracker.children(outcome.run_id)}

    assert set(children) == {"dummy", "logistic_regression"}
    assert children["logistic_regression"].tags["selected"] == "true"
    assert children["dummy"].tags["selected"] == "false"
    assert children["logistic_regression"].params == {"candidate": "logistic_regression", "C": "1.0"}
    assert tracker.json_artifact(outcome.run_id, "cv_results.json")["selected"] == "logistic_regression"


def test_training_first_model_becomes_champion(trained: tuple[TrainingContext, TrainingOutcome]) -> None:
    context, outcome = trained
    run = tracker_of(context).runs[outcome.run_id]

    assert outcome.promoted and outcome.champion is None
    assert context.registry.get_by_alias(CHAMPION) == outcome.version
    assert context.registry.get_by_alias(PREVIOUS) is None
    assert (run.tags["promoted"], run.tags["champion_version"]) == ("true", "none")
    assert run.tags["gate_reason"] == outcome.decision.reason


def test_training_equal_retrain_promoted_previous_kept(raw_customers_file: Path) -> None:
    context = make_context(raw_customers_file)
    first = run_training(REQUEST, context)

    second = run_training(REQUEST, context)

    assert second.version.version == 2 and second.promoted
    assert second.champion == first.version
    assert context.registry.get_by_alias(CHAMPION) == second.version
    assert context.registry.get_by_alias(PREVIOUS) == first.version
    assert context.registry.get_by_alias(CHALLENGER) == second.version


def test_training_rejected_challenger_leaves_champion(raw_customers_file: Path) -> None:
    context = make_context(raw_customers_file)
    first = run_training(REQUEST, context)
    strict = replace(context, params=replace(context.params, gate=GateParams(min_recall=1.0)))

    second = run_training(REQUEST, strict)

    assert not second.promoted
    assert second.decision.reason.startswith("recall:")
    assert context.registry.get_by_alias(CHAMPION) == first.version
    assert context.registry.get_by_alias(PREVIOUS) is None
    assert context.registry.get_by_alias(CHALLENGER) == second.version
    assert isinstance(context.registry, FakeRegistry)
    assert context.registry.version_tags[2]["promoted"] == "false"
    assert tracker_of(context).runs[second.run_id].tags["champion_version"] == "1"
    logged = tracker_of(context).json_artifact(second.run_id, "gate_decision.json")
    assert logged["promoted"] is False and logged["champion"]["pr_auc"] == second.decision.champion.pr_auc


def test_training_rejected_first_model_no_champion(raw_customers_file: Path) -> None:
    context = make_context(raw_customers_file)
    strict = replace(context, params=replace(context.params, gate=GateParams(min_pr_auc=1.0)))

    outcome = run_training(REQUEST, strict)

    assert not outcome.promoted
    assert context.registry.get_by_alias(CHAMPION) is None
    assert context.registry.get_by_alias(CHALLENGER) == outcome.version


def test_training_unloadable_champion_fails_run(raw_customers_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    context = make_context(raw_customers_file)
    first = run_training(REQUEST, context)

    def broken(version: int) -> None:
        raise ValueError("refusing to load")

    monkeypatch.setattr(context.registry, "load_model", broken)

    with pytest.raises(ValueError, match="refusing to load"):
        run_training(REQUEST, context)

    assert context.registry.get_by_alias(CHAMPION) == first.version
    parents = [run for run in tracker_of(context).runs.values() if run.parent_run_id is None]
    assert [run.status for run in parents] == [FINISHED, FAILED]


def test_training_invalid_data_fails_run(raw_customers: pd.DataFrame, tmp_path: Path) -> None:
    raw_file = tmp_path / "marketing_campaign.csv"
    raw_customers.head(50).to_csv(raw_file, sep="\t", index=False, lineterminator="\r\n")
    context = make_context(raw_file)

    with pytest.raises(DataValidationError, match="validation check"):
        run_training(REQUEST, context)

    run = next(iter(tracker_of(context).runs.values()))
    assert run.status == FAILED
    assert tracker_of(context).json_artifact(run.run_id, "validation_report.json")["passed"] is False
    assert context.registry.get_by_alias(CHALLENGER) is None


def test_flatten_nested_dotted_keys() -> None:
    assert flatten({"a": 1, "b": {"c": 2, "d": {"e": 3}}}) == {"a": 1, "b.c": 2, "b.d.e": 3}

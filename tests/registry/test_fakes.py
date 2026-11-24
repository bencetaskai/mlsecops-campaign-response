"""Behaviour of the in-memory tracker fake; the registry fake is covered by the registry contract tests."""

from pathlib import Path

import pytest
from fakes import FAILED, FINISHED, FakeRegistry, FakeTracker
from matplotlib.figure import Figure

from campaign_response.registry.ports import ExperimentTracker, ModelRegistry


def test_fakes_conform_to_ports() -> None:
    tracker = FakeTracker()

    assert isinstance(tracker, ExperimentTracker)
    assert isinstance(FakeRegistry(tracker), ModelRegistry)


def test_fake_run_finished_after_block() -> None:
    tracker = FakeTracker()

    with tracker.run("exp", "train") as run_id:
        pass

    assert tracker.runs[run_id].status == FINISHED
    assert (tracker.runs[run_id].experiment, tracker.runs[run_id].name) == ("exp", "train")


def test_fake_run_failed_and_error_reraised() -> None:
    tracker = FakeTracker()

    with pytest.raises(RuntimeError, match="boom"), tracker.run("exp", "train"):
        raise RuntimeError("boom")

    assert tracker.runs["run-1"].status == FAILED


def test_fake_nested_runs_listed_as_children() -> None:
    tracker = FakeTracker()

    with tracker.run("exp", "parent") as parent, tracker.run("exp", "child", parent_run_id=parent) as child:
        pass

    assert [run.run_id for run in tracker.children(parent)] == [child]


def test_fake_unknown_parent_KeyError() -> None:
    with pytest.raises(KeyError, match="unknown run"):
        FakeTracker().run("exp", "child", parent_run_id="nope").__enter__()


def test_fake_params_strings_immutable() -> None:
    tracker = FakeTracker()
    with tracker.run("exp", "train") as run_id:
        tracker.log_params(run_id, {"seed": 42})
        tracker.log_params(run_id, {"seed": "42"})

        with pytest.raises(ValueError, match="already logged"):
            tracker.log_params(run_id, {"seed": 7})

    assert tracker.runs[run_id].params == {"seed": "42"}


def test_fake_records_metrics_tags_artifacts(tmp_path: Path) -> None:
    tracker = FakeTracker()
    local = tmp_path / "sample.csv"
    local.write_text("a\n1\n", encoding="utf-8")

    with tracker.run("exp", "train") as run_id:
        tracker.log_metrics(run_id, {"pr_auc": 0.6})
        tracker.set_tags(run_id, {"promoted": "true"})
        tracker.log_dict(run_id, {"promoted": True}, "gate_decision.json")
        tracker.log_figure(run_id, Figure(), "pr_curve.png")
        tracker.log_file(run_id, local, artifact_path="reference")

    run = tracker.runs[run_id]
    assert (run.metrics, run.tags) == ({"pr_auc": 0.6}, {"promoted": "true"})
    assert tracker.json_artifact(run_id, "gate_decision.json") == {"promoted": True}
    assert set(run.artifacts) == {"gate_decision.json", "pr_curve.png", "reference/sample.csv"}


def test_fake_log_unknown_run_KeyError() -> None:
    with pytest.raises(KeyError, match="unknown run"):
        FakeTracker().log_metrics("run-9", {"pr_auc": 0.5})


def test_fake_register_without_model_ValueError() -> None:
    tracker = FakeTracker()
    with tracker.run("exp", "train") as run_id:
        pass

    with pytest.raises(ValueError, match="no logged model"):
        FakeRegistry(tracker).register(f"runs:/{run_id}/model", run_id)

"""End-to-end tests of the prepare and split commands on small files in a temporary working directory."""

import json
import logging
import shutil
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest
from fakes import FakeRegistry, FakeTracker

from campaign_response.cli import build_parser, main
from campaign_response.data import schema
from campaign_response.data.io import load_processed
from campaign_response.registry import mlflow_adapter

RAW = Path("data/raw/marketing_campaign.csv")
CLEANED = Path("data/processed/customers_clean.csv")
REPO_PARAMS = Path(__file__).resolve().parents[1] / "params.yaml"


@pytest.fixture(autouse=True)
def isolated_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Run in an empty directory with the repository's params.yaml, default settings, and restore the root logger."""
    shutil.copy(REPO_PARAMS, tmp_path / "params.yaml")
    monkeypatch.chdir(tmp_path)
    for variable in ("CRP_LOG_LEVEL", "CRP_LOG_FORMAT", "CRP_DATA_DIR", "CRP_MLFLOW_TRACKING_URI", "CRP_MODEL_NAME"):
        monkeypatch.delenv(variable, raising=False)
    root = logging.getLogger()
    saved_handlers, saved_level = root.handlers[:], root.level
    yield
    root.handlers[:] = saved_handlers
    root.setLevel(saved_level)


def write_raw(rows: int) -> None:
    """Write a valid raw file in the Kaggle format: tab-separated, CRLF, day-first dates."""
    values: dict[str, list[object]] = {column: [0] * rows for column in schema.RAW_COLUMNS}
    values.update({
        "ID": list(range(rows)),
        "Year_Birth": [1950 + index % 40 for index in range(rows)],
        "Education": ["Graduation"] * rows,
        "Marital_Status": ["Married"] * rows,
        "Income": [30000 + index for index in range(rows)],
        "Dt_Customer": ["04-09-2012"] * rows,
        "Z_CostContact": [3] * rows,
        "Z_Revenue": [11] * rows,
        "Response": [index % 5 == 0 for index in range(rows)],
    })
    RAW.parent.mkdir(parents=True)
    pd.DataFrame(values).astype({"Response": int}).to_csv(RAW, sep="\t", index=False, lineterminator="\r\n")


def read_json(path: str) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def test_no_command_prints_version(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 0
    assert capsys.readouterr().out == "campaign-response 0.1.0\n"


def test_prepare_writes_cleaned_and_metrics() -> None:
    write_raw(120)

    assert main(["prepare"]) == 0

    cleaned = load_processed(CLEANED)
    assert len(cleaned) == 120
    assert set(schema.CONSTANT_COLUMNS).isdisjoint(cleaned.columns)
    metrics = read_json("metrics/prepare.json")
    assert metrics["rows_in"] == 120
    assert metrics["validation_checks_passed"] == 9


def test_prepare_failed_gate_exit_1_no_output() -> None:
    write_raw(10)

    assert main(["prepare"]) == 1

    assert not CLEANED.exists()
    assert not Path("metrics/prepare.json").exists()


def test_split_writes_disjoint_parts() -> None:
    write_raw(120)
    main(["prepare"])

    assert main(["split"]) == 0

    train = load_processed(Path("data/processed/train.csv"))
    holdout = load_processed(Path("data/processed/holdout.csv"))
    assert len(train) + len(holdout) == 120
    assert set(train["ID"]).isdisjoint(holdout["ID"])
    assert read_json("metrics/split.json")["holdout_rows"] == len(holdout)


def test_outputs_lf_line_endings() -> None:
    write_raw(120)
    main(["prepare"])
    main(["split"])

    for path in (CLEANED, Path("metrics/prepare.json"), Path("metrics/split.json")):
        assert b"\r\n" not in path.read_bytes(), path


def test_prepare_twice_identical_files() -> None:
    write_raw(120)
    main(["prepare"])
    first = CLEANED.read_bytes(), Path("metrics/prepare.json").read_bytes()

    main(["prepare"])

    assert (CLEANED.read_bytes(), Path("metrics/prepare.json").read_bytes()) == first


def test_split_uses_params_file() -> None:
    write_raw(120)
    main(["prepare"])
    Path("half.yaml").write_text(
        REPO_PARAMS.read_text(encoding="utf-8").replace("holdout_size: 0.25", "holdout_size: 0.5"), encoding="utf-8"
    )

    main(["split"])
    quarter = read_json("metrics/split.json")["holdout_rows"]

    main(["--params", "half.yaml", "split"])

    assert read_json("metrics/split.json")["holdout_rows"] > quarter


def test_parser_defaults_follow_data_dir() -> None:
    args = build_parser(Path("elsewhere")).parse_args(["split"])

    assert args.train == Path("elsewhere/processed/train.csv")


@pytest.fixture
def fake_mlflow(monkeypatch: pytest.MonkeyPatch) -> tuple[FakeTracker, list[tuple[str, str]]]:
    """Swap the MLflow adapters for the fakes and record the URI and model name the command passed."""
    tracker, created = FakeTracker(), []
    registry = FakeRegistry(tracker)

    def make_registry(uri: str, name: str) -> FakeRegistry:
        created.append((uri, name))
        return registry

    monkeypatch.setattr(mlflow_adapter, "MlflowTracker", lambda uri: tracker)
    monkeypatch.setattr(mlflow_adapter, "MlflowRegistry", make_registry)
    return tracker, created


TRAIN = ["train", "--candidates", "logistic_regression", "--n-iter", "1", "--cv-folds", "3"]


def test_train_prints_outcome_and_registers(
    raw_customers_file: Path, fake_mlflow: tuple[FakeTracker, list[tuple[str, str]]],
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CRP_MODEL_NAME", "test-model")

    assert main([*TRAIN, "--raw", str(raw_customers_file)]) == 0

    outcome = json.loads(capsys.readouterr().out)
    assert (outcome["model_version"], outcome["best_candidate"]) == (1, "logistic_regression")
    assert outcome["promoted"] is True
    assert fake_mlflow[1] == [("http://localhost:5500", "test-model")]
    parent = next(run for run in fake_mlflow[0].runs.values() if run.parent_run_id is None)
    assert parent.tags["trained_by"] == "cli"
    assert parent.params["request.cv_folds"] == "3"


def test_train_invalid_data_exit_1(fake_mlflow: tuple[FakeTracker, list[tuple[str, str]]]) -> None:
    write_raw(10)

    assert main([*TRAIN, "--raw", str(RAW)]) == 1


def test_train_unknown_candidate_rejected_by_parser(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main(["train", "--candidates", "svm"])

    assert "invalid choice: 'svm'" in capsys.readouterr().err


def test_parser_train_defaults_match_request() -> None:
    args = build_parser(Path("data")).parse_args(["train"])

    assert args.candidates == ["logistic_regression", "random_forest", "hist_gradient_boosting"]
    assert (args.n_iter, args.cv_folds, args.seed) == (10, 5, 42)
    assert args.raw == Path("data/raw/marketing_campaign.csv")

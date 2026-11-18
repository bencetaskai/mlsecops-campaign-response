"""End-to-end tests of the prepare and split commands on small files in a temporary working directory."""

import json
import logging
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest

from campaign_response.cli import build_parser, main
from campaign_response.data import schema
from campaign_response.data.io import load_processed

RAW = Path("data/raw/marketing_campaign.csv")
CLEANED = Path("data/processed/customers_clean.csv")


@pytest.fixture(autouse=True)
def isolated_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Run in an empty directory with default settings and restore the root logger afterwards."""
    monkeypatch.chdir(tmp_path)
    for variable in ("CRP_LOG_LEVEL", "CRP_LOG_FORMAT", "CRP_DATA_DIR"):
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
    assert (len(train), len(holdout)) == (90, 30)
    assert set(train["ID"]).isdisjoint(holdout["ID"])
    assert read_json("metrics/split.json")["holdout_positive_rate"] == 0.2


def test_prepare_twice_identical_files() -> None:
    write_raw(120)
    main(["prepare"])
    first = CLEANED.read_bytes(), Path("metrics/prepare.json").read_bytes()

    main(["prepare"])

    assert (CLEANED.read_bytes(), Path("metrics/prepare.json").read_bytes()) == first


def test_parser_defaults_follow_data_dir() -> None:
    args = build_parser(Path("elsewhere")).parse_args(["split"])

    assert args.train == Path("elsewhere/processed/train.csv")

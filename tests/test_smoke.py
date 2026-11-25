"""Tests for the smoke check and its CLI command, against the scripted stand-in API."""

import io
from pathlib import Path

import pandas as pd
import pytest
from api_stub import StubApi, ok, overview, status
from pydantic import SecretStr

from campaign_response import cli
from campaign_response.api_client import ApiClient
from campaign_response.serving.schemas import AliasedVersion, ModelOverview, Prediction, PredictResponse
from campaign_response.smoke import records_from_raw, run_smoke, sample_customers

KEY = SecretStr("k" * 40)


def answer_for(customers: pd.DataFrame) -> PredictResponse:
    return PredictResponse(
        request_id="req-1", model_name="campaign-response-classifier", model_version=1, model_alias="champion",
        threshold=0.3, predictions=[Prediction(ID=int(i), probability=0.8, label=1) for i in customers["ID"]],
    )


def stub_for(customers: pd.DataFrame, model: ModelOverview | None = None) -> StubApi:
    return StubApi({"GET /model": [ok(model or overview("champion"))], "POST /predict": [ok(answer_for(customers))]})


def test_records_from_raw_keep_id_and_iso_date(raw_frame: pd.DataFrame) -> None:
    records = records_from_raw(raw_frame)

    assert [record.ID for record in records] == raw_frame["ID"].tolist()
    assert records[0].Dt_Customer.isoformat() == "2012-09-04"
    assert records[1].Income is None
    assert records[1].Marital_Status == "Alone"


def test_sample_customers_seeded_and_capped(raw_frame: pd.DataFrame) -> None:
    first, again = sample_customers(raw_frame, 2, seed=1), sample_customers(raw_frame, 2, seed=1)

    assert first.index.tolist() == again.index.tolist()
    assert len(sample_customers(raw_frame, 10, seed=1)) == len(raw_frame)


def test_smoke_report_and_request(raw_frame: pd.DataFrame) -> None:
    stub, out = stub_for(raw_frame), io.StringIO()

    code = run_smoke(ApiClient("http://api:8000", KEY, stub.transport), raw_frame, out)

    assert code == 0
    assert stub.calls() == ["GET /model", "POST /predict"]
    assert stub.body(1)["source"] == "test" and len(stub.body(1)["records"]) == 3
    report = out.getvalue()
    assert "Champion   version 1 (logistic_regression), holdout PR-AUC 0.690, threshold 0.300" in report
    assert "request id req-1" in report
    assert "  5524        0.800      1                 1" in report


def test_smoke_no_champion_exit_1(raw_frame: pd.DataFrame) -> None:
    stub, out = stub_for(raw_frame, overview("challenger")), io.StringIO()

    assert run_smoke(ApiClient("http://api:8000", KEY, stub.transport), raw_frame, out) == 1
    assert stub.calls() == ["GET /model"]
    assert "No champion yet" in out.getvalue()


def test_smoke_api_error_exit_1(raw_frame: pd.DataFrame) -> None:
    stub, out = StubApi({"GET /model": [ok(overview("champion"))], "POST /predict": [status(503)]}), io.StringIO()

    assert run_smoke(ApiClient("http://api:8000", KEY, stub.transport), raw_frame, out) == 1
    assert out.getvalue().startswith("API call failed: HTTPStatusError")


def test_smoke_champion_without_metrics_na(raw_frame: pd.DataFrame) -> None:
    bare = ModelOverview(model_name="m", versions=[AliasedVersion(
        alias="champion", version=3, run_id="r", best_candidate=None, threshold=None, holdout_metrics=None,
        promoted=None, gate_reason=None,
    )])
    out = io.StringIO()

    run_smoke(ApiClient("http://api:8000", KEY, stub_for(raw_frame, bare).transport), raw_frame, out)

    assert "holdout PR-AUC n/a" in out.getvalue()


@pytest.fixture
def raw_file(raw_frame: pd.DataFrame, tmp_path: Path) -> Path:
    path = tmp_path / "marketing_campaign.csv"
    raw_frame.to_csv(path, sep="\t", index=False, lineterminator="\r\n")
    return path


@pytest.fixture
def cli_stubs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, raw_frame: pd.DataFrame) -> StubApi:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CRP_API_KEY", raising=False)
    monkeypatch.setattr(cli, "configure_logging", lambda level, log_format: None)
    stub = stub_for(raw_frame)
    monkeypatch.setattr(cli, "ApiClient", lambda url, key: ApiClient(url, key, stub.transport))
    return stub


def test_cli_smoke_key_from_file(
    cli_stubs: StubApi, raw_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    key_file = tmp_path / "api_key.txt"
    key_file.write_text(KEY.get_secret_value() + "\n", encoding="utf-8")

    code = cli.main(["--params", str(Path(cli.__file__).parents[2] / "params.yaml"), "smoke", "--raw", str(raw_file),
                     "--records", "3", "--api-key-file", str(key_file)])

    assert code == 0
    assert cli_stubs.requests[0].headers["X-API-Key"] == KEY.get_secret_value()
    assert capsys.readouterr().out.startswith("API        http://localhost:8000")


def test_cli_smoke_key_from_environment(
    cli_stubs: StubApi, raw_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CRP_API_KEY", "e" * 40)

    cli.main(["--params", str(Path(cli.__file__).parents[2] / "params.yaml"), "smoke", "--raw", str(raw_file),
              "--api-key-file", "missing.txt"])

    assert cli_stubs.requests[0].headers["X-API-Key"] == "e" * 40


@pytest.mark.parametrize("records", ["0", "1001"])
def test_cli_smoke_records_out_of_range_rejected(records: str) -> None:
    with pytest.raises(SystemExit):
        cli.build_parser(Path("data")).parse_args(["smoke", "--records", records])

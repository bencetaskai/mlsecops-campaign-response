"""Command-line interface: ``python -m campaign_response <command>``; the DVC pipeline stages call these commands.

prepare  raw file -> validation gate -> cleaning -> cleaned data + metrics
split    cleaned data -> stratified train / holdout + metrics
train    raw file -> training run in MLflow -> challenger, promoted to champion if the gate passes
smoke    running API -> champion and a prediction for a few raw customers, printed as a report
"""

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from pydantic import SecretStr

from campaign_response import __version__
from campaign_response.api_client import ApiClient
from campaign_response.config import Settings
from campaign_response.data.cleaning import CleaningParams, clean
from campaign_response.data.io import load_processed, load_raw, write_processed
from campaign_response.data.splitting import SplitParams, split_train_holdout
from campaign_response.data.validation import validate_raw
from campaign_response.logging_setup import configure_logging, new_correlation_id
from campaign_response.modeling.candidates import BASELINE, CANDIDATE_NAMES
from campaign_response.params import PipelineParams, load_params
from campaign_response.pipeline.train_pipeline import (
    DataValidationError,
    TrainingContext,
    TrainingRequest,
    run_training,
)
from campaign_response.serving.schemas import MAX_RECORDS
from campaign_response.smoke import DEFAULT_RECORDS, run_smoke, sample_customers

logger = logging.getLogger(__name__)


def write_metrics(payload: dict[str, object], path: Path) -> None:
    """Write metrics as sorted, indented JSON with LF endings, so identical runs give identical files on any OS.

    newline="\\n" matters: text mode on Windows would write CRLF, and dvc.lock would then record a different hash
    than a Linux run of the same stage.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def prepare(raw: Path, cleaned_path: Path, metrics_path: Path, params: CleaningParams) -> int:
    """Validate and clean the raw file; on a failed gate write nothing and return 1."""
    frame = load_raw(raw)
    validation = validate_raw(frame)
    if not validation.passed:
        failures = [asdict(check) for check in validation.failures]
        logger.error("raw data failed the validation gate", extra={"failures": failures})
        return 1
    cleaned, report = clean(frame, params)
    write_processed(cleaned, cleaned_path)
    write_metrics({"validation_checks_passed": len(validation.checks), **asdict(report)}, metrics_path)
    return 0


def split(cleaned_path: Path, train_path: Path, holdout_path: Path, metrics_path: Path, params: SplitParams) -> int:
    """Split the cleaned data into train and holdout files."""
    train, holdout = split_train_holdout(load_processed(cleaned_path), params)
    write_processed(train, train_path)
    write_processed(holdout, holdout_path)
    write_metrics(
        {
            "train_rows": len(train),
            "train_positive_rate": round(float(train["Response"].mean()), 4),
            "holdout_rows": len(holdout),
            "holdout_positive_rate": round(float(holdout["Response"].mean()), 4),
        },
        metrics_path,
    )
    return 0


def train(raw: Path, request: TrainingRequest, params: PipelineParams, settings: Settings) -> int:
    """One training run against the MLflow server in settings; prints the outcome as JSON.

    Returns 0 whether or not the gate promotes the model (a rejection is a valid result), 1 if the data is invalid.
    """
    # Imported here: importing mlflow takes seconds, which prepare and split, run by DVC, should not pay.
    from campaign_response.registry.mlflow_adapter import MlflowRegistry, MlflowTracker

    context = TrainingContext(
        tracker=MlflowTracker(settings.mlflow_tracking_uri),
        registry=MlflowRegistry(settings.mlflow_tracking_uri, settings.model_name),
        params=params,
        raw_data_path=raw,
        requirements_path=Path("requirements.txt"),
    )
    try:
        outcome = run_training(request, context)
    except DataValidationError as error:
        logger.error("training refused", extra={"reason": str(error)})
        return 1
    print(json.dumps({
        "run_id": outcome.run_id,
        "model_version": outcome.version.version,
        "best_candidate": outcome.best_candidate,
        "threshold": round(outcome.threshold, 4),
        "holdout_pr_auc": round(outcome.holdout.pr_auc, 4),
        "promoted": outcome.promoted,
        "gate_reason": outcome.decision.reason,
    }, indent=2))
    return 0


def record_count(text: str) -> int:
    """argparse type for --records: the API accepts 1 to MAX_RECORDS records per request."""
    value = int(text)
    if not 1 <= value <= MAX_RECORDS:
        raise argparse.ArgumentTypeError(f"must be between 1 and {MAX_RECORDS}, got {value}")
    return value


def build_parser(data_dir: Path) -> argparse.ArgumentParser:
    raw, processed = data_dir / "raw", data_dir / "processed"
    parser = argparse.ArgumentParser(prog="campaign-response", description="Campaign response MLOps pipeline.")
    parser.add_argument("--params", type=Path, default=Path("params.yaml"), help="pipeline parameters file")
    commands = parser.add_subparsers(dest="command")

    prepare_parser = commands.add_parser("prepare", help="validate and clean the raw data")
    prepare_parser.add_argument("--raw", type=Path, default=raw / "marketing_campaign.csv")
    prepare_parser.add_argument("--out", type=Path, default=processed / "customers_clean.csv")
    prepare_parser.add_argument("--metrics", type=Path, default=Path("metrics/prepare.json"))

    split_parser = commands.add_parser("split", help="split the cleaned data into train and holdout")
    split_parser.add_argument("--cleaned", type=Path, default=processed / "customers_clean.csv")
    split_parser.add_argument("--train", type=Path, default=processed / "train.csv")
    split_parser.add_argument("--holdout", type=Path, default=processed / "holdout.csv")
    split_parser.add_argument("--metrics", type=Path, default=Path("metrics/split.json"))

    defaults = TrainingRequest()
    train_parser = commands.add_parser("train", help="train candidates, register a challenger, apply the gate")
    train_parser.add_argument("--raw", type=Path, default=raw / "marketing_campaign.csv")
    train_parser.add_argument("--candidates", nargs="+", default=list(defaults.candidates),
                              choices=[name for name in CANDIDATE_NAMES if name != BASELINE])
    train_parser.add_argument("--n-iter", type=int, default=defaults.n_iter)
    train_parser.add_argument("--cv-folds", type=int, default=defaults.cv_folds)
    train_parser.add_argument("--seed", type=int, default=defaults.seed)

    smoke_parser = commands.add_parser("smoke", help="check the running API with a prediction for a few customers")
    smoke_parser.add_argument("--raw", type=Path, default=raw / "marketing_campaign.csv")
    smoke_parser.add_argument("--records", type=record_count, default=DEFAULT_RECORDS, metavar=f"1..{MAX_RECORDS}")
    smoke_parser.add_argument("--seed", type=int, default=7)
    smoke_parser.add_argument("--api-key-file", type=Path, default=Path("secrets/api_key.txt"),
                              help="used when CRP_API_KEY is not set")
    return parser


def smoke(raw: Path, count: int, seed: int, api_key_file: Path, settings: Settings) -> int:
    """Run the smoke check against settings.api_url; the key comes from the settings or the key file."""
    key = settings.api_key or SecretStr(api_key_file.read_text(encoding="utf-8").strip())
    client = ApiClient(settings.api_url, key)
    try:
        print(f"API        {settings.api_url}")
        return run_smoke(client, sample_customers(load_raw(raw), count, seed), sys.stdout)
    finally:
        client.close()


def main(argv: Sequence[str] | None = None) -> int:
    """Run a command and return its exit code; without a command print the version (used as a smoke test)."""
    settings = Settings()
    args = build_parser(settings.data_dir).parse_args(argv)
    if args.command is None:
        print(f"campaign-response {__version__}")
        return 0
    configure_logging(settings.log_level, settings.log_format)
    logger.info("command started", extra={"command": args.command, "run_id": new_correlation_id()})
    params = load_params(args.params)
    if args.command == "prepare":
        return prepare(args.raw, args.out, args.metrics, params.cleaning)
    if args.command == "split":
        return split(args.cleaned, args.train, args.holdout, args.metrics, params.split)
    if args.command == "smoke":
        return smoke(args.raw, args.records, args.seed, args.api_key_file, settings)
    request = TrainingRequest(tuple(args.candidates), args.n_iter, args.cv_folds, args.seed, trained_by="cli")
    return train(args.raw, request, params, settings)

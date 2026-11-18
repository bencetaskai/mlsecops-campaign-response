"""Command-line interface: ``python -m campaign_response <command>``; the DVC pipeline stages call these commands.

prepare  raw file -> validation gate -> cleaning -> cleaned data + metrics
split    cleaned data -> stratified train / holdout + metrics
"""

import argparse
import json
import logging
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from campaign_response import __version__
from campaign_response.config import Settings
from campaign_response.data.cleaning import CleaningParams, clean
from campaign_response.data.io import load_processed, load_raw, write_processed
from campaign_response.data.splitting import SplitParams, split_train_holdout
from campaign_response.data.validation import validate_raw
from campaign_response.logging_setup import configure_logging, new_correlation_id
from campaign_response.params import load_params

logger = logging.getLogger(__name__)


def write_metrics(payload: dict[str, object], path: Path) -> None:
    """Write metrics as sorted, indented JSON, so identical runs give identical files and readable git diffs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
    return parser


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
    return split(args.cleaned, args.train, args.holdout, args.metrics, params.split)

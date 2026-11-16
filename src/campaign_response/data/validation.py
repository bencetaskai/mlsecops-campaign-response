"""Validation gate for raw data: decides whether a raw file is usable before anything is trained on it.

Each check returns a CheckResult; the report passes only if every check passes. Checks that need a column skip it
when the column is missing, so one schema problem produces one clear failure instead of a crash.
"""

import json
import logging
from dataclasses import asdict, dataclass

import pandas as pd

from campaign_response.data import schema

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Thresholds:
    min_rows: int = 100
    max_missing_share: float = 0.05
    max_out_of_range_share: float = 0.01
    max_unknown_category_share: float = 0.01


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class ValidationReport:
    rows: int
    checks: tuple[CheckResult, ...]

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    @property
    def failures(self) -> tuple[CheckResult, ...]:
        return tuple(check for check in self.checks if not check.passed)

    def to_json(self) -> str:
        return json.dumps({"passed": self.passed, **asdict(self)}, indent=2)


def _share(mask: "pd.Series[bool]") -> float:
    return float(mask.mean()) if len(mask) else 0.0


def check_columns(frame: pd.DataFrame) -> CheckResult:
    missing = [column for column in schema.RAW_COLUMNS if column not in frame.columns]
    unexpected = [str(column) for column in frame.columns if column not in schema.RAW_DTYPES]
    passed = not missing and not unexpected
    return CheckResult("columns", passed, f"missing={missing} unexpected={unexpected}")


def check_dtypes(frame: pd.DataFrame) -> CheckResult:
    wrong = {
        column: str(frame[column].dtype)
        for column, expected in schema.RAW_DTYPES.items()
        if column in frame.columns and str(frame[column].dtype) != expected
    }
    return CheckResult("dtypes", not wrong, f"wrong={wrong}")


def check_row_count(frame: pd.DataFrame, thresholds: Thresholds) -> CheckResult:
    passed = len(frame) >= thresholds.min_rows
    return CheckResult("row_count", passed, f"rows={len(frame)} min={thresholds.min_rows}")


def check_missing_values(frame: pd.DataFrame, thresholds: Thresholds) -> CheckResult:
    shares = {str(column): _share(frame[column].isna()) for column in frame.columns}
    too_high = {column: round(share, 4) for column, share in shares.items() if share > thresholds.max_missing_share}
    return CheckResult("missing_values", not too_high, f"above {thresholds.max_missing_share}: {too_high}")


def check_value_ranges(frame: pd.DataFrame, thresholds: Thresholds) -> CheckResult:
    too_high: dict[str, float] = {}
    for column, (low, high) in schema.VALUE_RANGES.items():
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        share = _share((values < low) | (values > high))
        if share > thresholds.max_out_of_range_share:
            too_high[column] = round(share, 4)
    return CheckResult("value_ranges", not too_high, f"out-of-range above {thresholds.max_out_of_range_share}: "
                                                     f"{too_high}")


def check_categories(frame: pd.DataFrame, thresholds: Thresholds) -> CheckResult:
    allowed = {"Education": schema.EDUCATION_LEVELS, "Marital_Status": schema.MARITAL_STATUSES}
    too_high: dict[str, float] = {}
    for column, levels in allowed.items():
        if column in frame.columns:
            share = _share(~frame[column].dropna().isin(levels))
            if share > thresholds.max_unknown_category_share:
                too_high[column] = round(share, 4)
    return CheckResult("categories", not too_high, f"unknown above {thresholds.max_unknown_category_share}: "
                                                   f"{too_high}")


def check_dates(frame: pd.DataFrame) -> CheckResult:
    if schema.DATE_COLUMN not in frame.columns:
        return CheckResult("dates", False, f"{schema.DATE_COLUMN} missing")
    values = frame[schema.DATE_COLUMN].dropna()
    unparseable = int(pd.to_datetime(values, format=schema.DATE_FORMAT, errors="coerce").isna().sum())
    return CheckResult("dates", unparseable == 0, f"unparseable={unparseable} format={schema.DATE_FORMAT}")


def check_ids_unique(frame: pd.DataFrame) -> CheckResult:
    if schema.ID_COLUMN not in frame.columns:
        return CheckResult("ids_unique", False, f"{schema.ID_COLUMN} missing")
    duplicates = int(frame[schema.ID_COLUMN].duplicated().sum())
    return CheckResult("ids_unique", duplicates == 0, f"duplicate ids={duplicates}")


def check_target(frame: pd.DataFrame) -> CheckResult:
    if schema.TARGET_COLUMN not in frame.columns:
        return CheckResult("target", False, f"{schema.TARGET_COLUMN} missing")
    classes = sorted(int(value) for value in frame[schema.TARGET_COLUMN].dropna().unique())
    return CheckResult("target", classes == [0, 1], f"classes={classes}")


def validate_raw(frame: pd.DataFrame, thresholds: Thresholds = Thresholds()) -> ValidationReport:
    """Run every check on a raw frame and log the outcome; the caller decides what a failure stops."""
    report = ValidationReport(
        rows=len(frame),
        checks=(
            check_columns(frame),
            check_dtypes(frame),
            check_row_count(frame, thresholds),
            check_missing_values(frame, thresholds),
            check_value_ranges(frame, thresholds),
            check_categories(frame, thresholds),
            check_dates(frame),
            check_ids_unique(frame),
            check_target(frame),
        ),
    )
    failed = [check.name for check in report.failures]
    logger.info("raw data validation", extra={"passed": report.passed, "rows": report.rows, "failed": failed})
    return report

"""Deterministic cleaning rules, one pure function per rule.

No rule uses statistics computed from the data (only fixed thresholds and mappings), so cleaning before the
train/holdout split causes no leakage. Every function returns a new frame and leaves its input unchanged.
"""

import logging
from dataclasses import dataclass
from datetime import date

import pandas as pd

from campaign_response.data.schema import CONSTANT_COLUMNS, DATE_COLUMN, DATE_FORMAT, ID_COLUMN, TARGET_COLUMN

logger = logging.getLogger(__name__)

MARITAL_STATUS_SYNONYMS = {"Alone": "Single"}
# Junk answers: the true status is unknown, so they become missing instead of being guessed.
UNKNOWN_MARITAL_STATUSES = frozenset({"Absurd", "YOLO"})


@dataclass(frozen=True)
class CleaningParams:
    """Fixed thresholds of the cleaning rules (moved to params.yaml with the DVC pipeline)."""

    # Fixed, never today() or the data maximum: the same customer must always get the same result.
    reference_date: date = date(2014, 12, 31)
    min_age: int = 18
    max_age: int = 100
    # Catches placeholders such as 666666, not rich customers: the data has no income between 162,398 and 666,665.
    max_income: float = 200_000


def birth_year_bounds(params: CleaningParams) -> tuple[int, int]:
    """Plausible birth years: customers aged min_age to max_age at the reference date (only the year is known)."""
    return params.reference_date.year - params.max_age, params.reference_date.year - params.min_age


def drop_unused_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop the customer ID (an identifier, not a feature) and the two constant columns."""
    return frame.drop(columns=[ID_COLUMN, *CONSTANT_COLUMNS])


def drop_duplicate_customers(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop rows identical in every column except ID, keeping the first; return the frame and the removed count.

    A duplicate in both train and holdout would let the model be scored on a customer it has already seen.
    The original index is kept, so removed rows can be traced back to the raw file.
    """
    subset = [column for column in frame.columns if column != ID_COLUMN]
    duplicated = frame.duplicated(subset=subset, keep="first")
    removed = int(duplicated.sum())
    logger.info("dropped duplicate customers", extra={"removed": removed, "kept": len(frame) - removed})
    return frame.loc[~duplicated].copy(), removed


def drop_conflicting_labels(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop every row of a customer recorded more than once with different target values; return the removed count.

    Such rows are identical in every column except ID and the target, so the true label cannot be known. Keeping
    one copy would pick a label arbitrarily, keeping all would train on contradictions.
    """
    features = [column for column in frame.columns if column not in {ID_COLUMN, TARGET_COLUMN}]
    labels_per_customer = frame.groupby(features, dropna=False)[TARGET_COLUMN].transform("nunique")
    conflicting = labels_per_customer > 1
    removed = int(conflicting.sum())
    logger.info("dropped conflicting labels", extra={"removed": removed, "kept": len(frame) - removed})
    return frame.loc[~conflicting].copy(), removed


def mask_implausible_birth_year(frame: pd.DataFrame, params: CleaningParams = CleaningParams()) -> pd.DataFrame:
    """Set birth years outside the plausible age range to missing; the imputer in the model pipeline fills them.

    Masking instead of dropping keeps the rest of the customer's data.
    """
    low, high = birth_year_bounds(params)
    years = frame["Year_Birth"]
    implausible = ((years < low) | (years > high)).fillna(False)
    result = frame.copy()
    result.loc[implausible, "Year_Birth"] = pd.NA
    logger.info("masked implausible birth years", extra={"masked": int(implausible.sum()), "bounds": [low, high]})
    return result


def mask_implausible_income(frame: pd.DataFrame, params: CleaningParams = CleaningParams()) -> pd.DataFrame:
    """Set incomes above max_income to missing; unusual but possible values, including low ones, are kept."""
    implausible = (frame["Income"] > params.max_income).fillna(False)
    result = frame.copy()
    result.loc[implausible, "Income"] = pd.NA
    logger.info("masked implausible incomes", extra={"masked": int(implausible.sum()), "max": params.max_income})
    return result


def normalise_marital_status(frame: pd.DataFrame) -> pd.DataFrame:
    """Map synonyms (Alone -> Single) and set junk answers to missing; any other value passes through unchanged."""
    status = frame["Marital_Status"]
    unknown = status.isin(UNKNOWN_MARITAL_STATUSES).fillna(False)
    synonym = status.isin(MARITAL_STATUS_SYNONYMS.keys()).fillna(False)
    result = frame.copy()
    result["Marital_Status"] = status.replace(MARITAL_STATUS_SYNONYMS).mask(unknown)
    logger.info("normalised marital status", extra={"mapped": int(synonym.sum()), "masked": int(unknown.sum())})
    return result


def parse_enrollment_date(frame: pd.DataFrame) -> pd.DataFrame:
    """Parse the enrollment date with the strict day-first format; raise if any value does not match.

    No format guessing: 38% of the dates (e.g. 04-09-2012) would parse silently but wrongly as month-first.
    A non-matching value means corrupt data that the validation gate should have rejected, so it is an error,
    not a missing value. Values that are already missing stay missing.
    """
    raw = frame[DATE_COLUMN]
    parsed = pd.to_datetime(raw, format=DATE_FORMAT, errors="coerce")
    invalid = parsed.isna() & raw.notna()
    if invalid.any():
        examples = raw[invalid].head(5).tolist()
        raise ValueError(f"{int(invalid.sum())} {DATE_COLUMN} values do not match {DATE_FORMAT}, e.g. {examples}")
    result = frame.copy()
    result[DATE_COLUMN] = parsed
    return result

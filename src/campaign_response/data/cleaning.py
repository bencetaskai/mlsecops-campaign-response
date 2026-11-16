"""Deterministic cleaning rules, one pure function per rule.

No rule uses statistics computed from the data (only fixed thresholds and mappings), so cleaning before the
train/holdout split causes no leakage. Every function returns a new frame and leaves its input unchanged.
"""

import logging

import pandas as pd

from campaign_response.data.schema import CONSTANT_COLUMNS, ID_COLUMN

logger = logging.getLogger(__name__)


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

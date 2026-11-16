"""Deterministic cleaning rules, one pure function per rule.

No rule uses statistics computed from the data (only fixed thresholds and mappings), so cleaning before the
train/holdout split causes no leakage. Every function returns a new frame and leaves its input unchanged.
"""

import pandas as pd

from campaign_response.data.schema import CONSTANT_COLUMNS, ID_COLUMN


def drop_unused_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop the customer ID (an identifier, not a feature) and the two constant columns."""
    return frame.drop(columns=[ID_COLUMN, *CONSTANT_COLUMNS])

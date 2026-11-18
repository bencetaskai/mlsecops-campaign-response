"""Train / holdout split of the cleaned data."""

import logging
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from campaign_response.data.schema import ID_COLUMN, TARGET_COLUMN

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SplitParams:
    """25% holdout: a split analysis showed holdout PR-AUC noise sd 0.054 at 20% and 0.042 at 25%."""

    holdout_size: float = 0.25
    seed: int = 42


def split_train_holdout(
    frame: pd.DataFrame, params: SplitParams = SplitParams()
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split randomly, stratified on the target, so both parts keep the positive rate; IDs must be unique.

    Model choices (candidates, hyperparameters, threshold) use only the train part; the holdout is scored once.
    """
    duplicate_ids = int(frame[ID_COLUMN].duplicated().sum())
    if duplicate_ids:
        raise ValueError(f"{duplicate_ids} duplicate {ID_COLUMN} values: a customer could land in both parts")
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=params.holdout_size, random_state=params.seed)
    train_rows, holdout_rows = next(splitter.split(frame, frame[TARGET_COLUMN]))
    train, holdout = frame.iloc[train_rows], frame.iloc[holdout_rows]
    logger.info(
        "split train / holdout",
        extra={
            "train_rows": len(train),
            "holdout_rows": len(holdout),
            "train_positive_rate": round(float(train[TARGET_COLUMN].mean()), 4),
            "holdout_positive_rate": round(float(holdout[TARGET_COLUMN].mean()), 4),
        },
    )
    return train, holdout

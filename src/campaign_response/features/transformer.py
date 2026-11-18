"""The feature engineering as an sklearn transformer, so the saved model pipeline carries its own feature code."""

from datetime import date
from typing import Self

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from campaign_response.features.engineering import DEFAULT_REFERENCE_DATE, MODEL_FEATURES, add_engineered_features


def to_sklearn_dtypes(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert pandas nullable dtypes to float64 / object with NaN for missing values.

    sklearn's imputers cannot handle pd.NA: SimpleImputer fails on a nullable string column with
    "boolean value of NA is ambiguous".
    """
    columns: dict[str, pd.Series] = {}
    for name, column in frame.items():
        if pd.api.types.is_numeric_dtype(column):
            columns[str(name)] = column.astype("float64")
        else:
            columns[str(name)] = column.astype(object).where(column.notna(), np.nan)
    return pd.DataFrame(columns, index=frame.index)


class FeatureEngineer(TransformerMixin, BaseEstimator):
    """Stateless step: fit learns nothing, transform adds the engineered features and returns MODEL_FEATURES.

    Stateless on purpose: every feature is a fixed formula, so nothing can leak from the training data.
    """

    def __init__(self, reference_date: date = DEFAULT_REFERENCE_DATE) -> None:
        self.reference_date = reference_date

    def fit(self, X: pd.DataFrame, y: object = None) -> Self:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return to_sklearn_dtypes(add_engineered_features(X, self.reference_date)[list(MODEL_FEATURES)])

    def get_feature_names_out(self, input_features: object = None) -> npt.NDArray[np.object_]:
        return np.asarray(MODEL_FEATURES, dtype=object)

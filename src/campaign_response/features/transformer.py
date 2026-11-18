"""The feature engineering as an sklearn transformer, so the saved model pipeline carries its own feature code."""

from datetime import date
from typing import Self

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from campaign_response.features.engineering import DEFAULT_REFERENCE_DATE, MODEL_FEATURES, add_engineered_features


class FeatureEngineer(TransformerMixin, BaseEstimator):
    """Stateless step: fit learns nothing, transform adds the engineered features and returns MODEL_FEATURES.

    Stateless on purpose: every feature is a fixed formula, so nothing can leak from the training data.
    """

    def __init__(self, reference_date: date = DEFAULT_REFERENCE_DATE) -> None:
        self.reference_date = reference_date

    def fit(self, X: pd.DataFrame, y: object = None) -> Self:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return add_engineered_features(X, self.reference_date)[list(MODEL_FEATURES)]

    def get_feature_names_out(self, input_features: object = None) -> npt.NDArray[np.object_]:
        return np.asarray(MODEL_FEATURES, dtype=object)

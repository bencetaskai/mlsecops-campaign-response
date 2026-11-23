"""Imputation, scaling and one-hot encoding of the model features, learned on the training data only."""

from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Categories rarer than this in the training data share one "infrequent" column; today the smallest category has 37
# training rows, so this only guards against rare values in later data versions.
DEFAULT_MIN_CATEGORY_COUNT = 10


def build_preprocessor(min_category_count: int = DEFAULT_MIN_CATEGORY_COUNT) -> ColumnTransformer:
    """Return an unfitted transformer for the FeatureEngineer output (float64 numbers, object categories).

    No missing-value indicators: they gave no CV gain, and SimpleImputer adds them only for columns with gaps in the
    training data, so the output columns would change with the data version instead of with the code.
    An unseen category at predict time becomes all zeros in its one-hot block instead of raising.
    """
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=min_category_count,
                                 sparse_output=False)),
    ])
    return ColumnTransformer(
        [
            ("numeric", numeric, make_column_selector(dtype_include="number")),
            ("categorical", categorical, make_column_selector(dtype_exclude="number")),
        ],
        verbose_feature_names_out=False,
    )

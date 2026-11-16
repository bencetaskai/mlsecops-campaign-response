"""Contract of the raw Kaggle file: column names and order, dtypes, allowed categories and plausible ranges.

Loader, validation and cleaning import these constants instead of repeating column names as string literals.
"""

import math
from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

ID_COLUMN: Final = "ID"
TARGET_COLUMN: Final = "Response"
DATE_COLUMN: Final = "Dt_Customer"
DATE_FORMAT: Final = "%d-%m-%Y"
CONSTANT_COLUMNS: Final = ("Z_CostContact", "Z_Revenue")

EDUCATION_LEVELS: Final = frozenset({"Basic", "Graduation", "2n Cycle", "Master", "PhD"})
MARITAL_STATUSES: Final = frozenset(
    {"Single", "Together", "Married", "Divorced", "Widow", "Alone", "Absurd", "YOLO"}
)

# Column order of the file (AcceptedCmp3-5 come before 1-2 in the source). Nullable "Int64" lets a missing value
# load as <NA>, so the validation gate reports it instead of the loader crashing.
RAW_DTYPES: Final[Mapping[str, str]] = MappingProxyType({
    "ID": "Int64",
    "Year_Birth": "Int64",
    "Education": "string",
    "Marital_Status": "string",
    "Income": "Float64",
    "Kidhome": "Int64",
    "Teenhome": "Int64",
    "Dt_Customer": "string",
    "Recency": "Int64",
    "MntWines": "Int64",
    "MntFruits": "Int64",
    "MntMeatProducts": "Int64",
    "MntFishProducts": "Int64",
    "MntSweetProducts": "Int64",
    "MntGoldProds": "Int64",
    "NumDealsPurchases": "Int64",
    "NumWebPurchases": "Int64",
    "NumCatalogPurchases": "Int64",
    "NumStorePurchases": "Int64",
    "NumWebVisitsMonth": "Int64",
    "AcceptedCmp3": "Int64",
    "AcceptedCmp4": "Int64",
    "AcceptedCmp5": "Int64",
    "AcceptedCmp1": "Int64",
    "AcceptedCmp2": "Int64",
    "Complain": "Int64",
    "Z_CostContact": "Int64",
    "Z_Revenue": "Int64",
    "Response": "Int64",
})
RAW_COLUMNS: Final = tuple(RAW_DTYPES)

BINARY_COLUMNS: Final = (
    "AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3", "AcceptedCmp4", "AcceptedCmp5", "Complain", TARGET_COLUMN,
)
NON_NEGATIVE_COLUMNS: Final = (
    "Income", "Kidhome", "Teenhome", "Recency",
    "MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", "MntGoldProds",
    "NumDealsPurchases", "NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases", "NumWebVisitsMonth",
)

# Plausible (min, max) per numeric column for the validation gate. Deliberately wide: they catch broken data
# (negative counts, a birth year of 2090), not outliers; masking outliers is a cleaning rule.
VALUE_RANGES: Final[Mapping[str, tuple[float, float]]] = MappingProxyType({
    "Year_Birth": (1890, 2010),
    **{column: (0, 1) for column in BINARY_COLUMNS},
    **{column: (0, math.inf) for column in NON_NEGATIVE_COLUMNS},
})

FEATURE_COLUMNS: Final = tuple(
    column for column in RAW_COLUMNS if column not in {ID_COLUMN, TARGET_COLUMN, *CONSTANT_COLUMNS}
)

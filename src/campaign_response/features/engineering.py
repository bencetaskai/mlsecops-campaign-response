"""Engineered customer features, computed with fixed formulas only.

No statistic is learned from the data and the reference date is fixed, so the features are identical in training
and at /predict time, for the same customer, whenever they are computed.
"""

from datetime import date

import pandas as pd

from campaign_response.data.cleaning import CleaningParams
from campaign_response.data.schema import DATE_COLUMN, FEATURE_COLUMNS

DEFAULT_REFERENCE_DATE = CleaningParams().reference_date

SPEND_COLUMNS = ("MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", "MntGoldProds")
CAMPAIGN_COLUMNS = ("AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3", "AcceptedCmp4", "AcceptedCmp5")
PARTNERED_STATUSES = frozenset({"Married", "Together"})

ENGINEERED_FEATURES = (
    "age", "tenure_days", "total_spend", "total_accepted_cmp", "household_size", "income_per_member",
)
REPLACED_COLUMNS = ("Year_Birth", DATE_COLUMN)
MODEL_FEATURES = tuple(column for column in FEATURE_COLUMNS if column not in REPLACED_COLUMNS) + ENGINEERED_FEATURES


def add_engineered_features(frame: pd.DataFrame, reference_date: date = DEFAULT_REFERENCE_DATE) -> pd.DataFrame:
    """Add the six engineered features and drop the two raw columns they replace (Year_Birth, Dt_Customer).

    household_size assumes one adult unless the customer is Married or Together; a missing marital status gives a
    missing household size and income per member, which the imputer fills like any other missing value.
    """
    result = frame.copy()
    partner = result["Marital_Status"].isin(PARTNERED_STATUSES).astype("Int64").mask(result["Marital_Status"].isna())
    household = (1 + partner + result["Kidhome"] + result["Teenhome"]).astype("Float64")

    result["age"] = (reference_date.year - result["Year_Birth"]).astype("Float64")
    result["tenure_days"] = (pd.Timestamp(reference_date) - result[DATE_COLUMN]).dt.days.astype("Int64")
    result["total_spend"] = result[list(SPEND_COLUMNS)].sum(axis=1).astype("Int64")
    result["total_accepted_cmp"] = result[list(CAMPAIGN_COLUMNS)].sum(axis=1).astype("Int64")
    result["household_size"] = household
    result["income_per_member"] = result["Income"] / household
    return result.drop(columns=list(REPLACED_COLUMNS))

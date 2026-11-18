"""Fixtures shared by all test folders."""

import pandas as pd
import pytest

from campaign_response.data import schema


@pytest.fixture
def raw_frame() -> pd.DataFrame:
    """Three raw customers with the schema dtypes; rows differ so each cleaning rule has something to act on."""
    rows = 3
    values: dict[str, list[object]] = {column: [0] * rows for column in schema.RAW_COLUMNS}
    values.update({
        "ID": [5524, 2174, 4141],
        "Year_Birth": [1957, 1954, 1965],
        "Education": ["Graduation", "2n Cycle", "PhD"],
        "Marital_Status": ["Single", "Alone", "Together"],
        "Income": [58138.0, None, 71613.0],
        "Dt_Customer": ["04-09-2012", "08-03-2014", "21-08-2013"],
        "MntWines": [635, 11, 426],
        "Z_CostContact": [3] * rows,
        "Z_Revenue": [11] * rows,
        "Response": [1, 0, 0],
    })
    return pd.DataFrame(values).astype(dict(schema.RAW_DTYPES))

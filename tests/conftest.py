"""Fixtures shared by all test folders."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from campaign_response.data import schema
from campaign_response.data.cleaning import clean


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
        "NumWebPurchases": [8, 1, 8],
        "NumCatalogPurchases": [10, 1, 2],
        "NumStorePurchases": [4, 2, 10],
        "Z_CostContact": [3] * rows,
        "Z_Revenue": [11] * rows,
        "Response": [1, 0, 0],
    })
    return pd.DataFrame(values).astype(dict(schema.RAW_DTYPES))


@pytest.fixture(scope="session")
def raw_customers() -> pd.DataFrame:
    """240 raw synthetic customers with the schema dtypes, about 20% responders driven by spend and acceptances."""
    rows = 240
    rng = np.random.default_rng(7)
    spend = rng.gamma(2.0, 150.0, rows).round()
    accepted = rng.binomial(1, 0.15, (rows, 5))
    signal = 0.004 * spend + 1.5 * accepted.sum(axis=1) + rng.normal(0, 1, rows)
    response = (signal > np.quantile(signal, 0.8)).astype(int)
    dates = pd.Timestamp("2012-08-01") + pd.to_timedelta(rng.integers(0, 680, rows), unit="D")
    values: dict[str, object] = {column: np.zeros(rows, dtype=int) for column in schema.RAW_COLUMNS}
    values.update({
        "ID": np.arange(1, rows + 1),
        "Year_Birth": rng.integers(1950, 1990, rows),
        "Education": rng.choice(["Graduation", "PhD", "Master", "2n Cycle", "Basic"], rows,
                                p=[0.5, 0.2, 0.15, 0.1, 0.05]),
        "Marital_Status": rng.choice(["Married", "Together", "Single", "Divorced", "Widow"], rows),
        "Income": rng.normal(52000, 15000, rows).clip(5000, 150000).round(),
        "Kidhome": rng.integers(0, 3, rows),
        "Teenhome": rng.integers(0, 3, rows),
        "Dt_Customer": dates.strftime("%d-%m-%Y"),
        "Recency": rng.integers(0, 100, rows),
        "MntWines": (spend * 0.5).round(),
        "MntMeatProducts": (spend * 0.3).round(),
        "MntGoldProds": (spend * 0.2).round(),
        "NumWebPurchases": rng.integers(0, 10, rows),
        "NumCatalogPurchases": rng.integers(0, 8, rows),
        "NumStorePurchases": rng.integers(1, 12, rows),
        "NumWebVisitsMonth": rng.integers(0, 10, rows),
        **{f"AcceptedCmp{index + 1}": accepted[:, index] for index in range(5)},
        "Z_CostContact": np.full(rows, 3),
        "Z_Revenue": np.full(rows, 11),
        "Response": response,
    })
    return pd.DataFrame(values).astype(dict(schema.RAW_DTYPES))


@pytest.fixture(scope="session")
def customers(raw_customers: pd.DataFrame) -> pd.DataFrame:
    """The synthetic customers after clean(), so dtypes match the real pipeline; enough rows for 3-fold CV."""
    cleaned, _ = clean(raw_customers)
    return cleaned


@pytest.fixture(scope="session")
def raw_customers_file(raw_customers: pd.DataFrame, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The raw customers stored like the Kaggle file: tab-separated with CRLF line endings."""
    path = tmp_path_factory.mktemp("raw") / "marketing_campaign.csv"
    raw_customers.to_csv(path, sep="\t", index=False, lineterminator="\r\n")
    return path

"""Smoke check of the running API: the champion, then a prediction for a few customers from the raw file.

``python -m campaign_response smoke`` prints a readable report (the REST API screenshot) and exits 0 when the API
served a prediction, 1 when it could not.
"""

import time
from collections.abc import Sequence
from datetime import datetime
from typing import Any, Final, TextIO

import httpx2
import numpy as np
import pandas as pd

from campaign_response.api_client import ApiClient
from campaign_response.data.schema import DATE_COLUMN, DATE_FORMAT, FEATURE_COLUMNS, ID_COLUMN, TARGET_COLUMN
from campaign_response.registry.ports import CHAMPION
from campaign_response.serving.schemas import CustomerRecord, PredictRequest

DEFAULT_RECORDS: Final = 5


def records_from_raw(raw: pd.DataFrame) -> list[CustomerRecord]:
    """Raw rows as the API receives them: ISO dates, null for missing values, the customer ID kept."""
    records = []
    for row in raw[[ID_COLUMN, *FEATURE_COLUMNS]].to_dict("records"):
        values: dict[str, Any] = {
            str(column): None if pd.isna(value) else value.item() if isinstance(value, np.generic) else value
            for column, value in row.items()
        }
        values[DATE_COLUMN] = datetime.strptime(values[DATE_COLUMN], DATE_FORMAT).date()
        records.append(CustomerRecord.model_validate(values))
    return records


def sample_customers(raw: pd.DataFrame, count: int, seed: int) -> pd.DataFrame:
    return raw.sample(n=min(count, len(raw)), random_state=seed)


def run_smoke(client: ApiClient, customers: pd.DataFrame, out: TextIO) -> int:
    try:
        overview = client.model_overview()
        champion = next((entry for entry in overview.versions if entry.alias == CHAMPION), None)
        if champion is None:
            print("No champion yet: run the model-bootstrap service or POST /train first.", file=out)
            return 1
        started = time.perf_counter()
        result = client.predict(PredictRequest(records=records_from_raw(customers), source="test"))
        seconds = time.perf_counter() - started
    except httpx2.HTTPError as error:
        print(f"API call failed: {type(error).__name__}: {error}", file=out)
        return 1

    metrics = champion.holdout_metrics
    pr_auc = f"{metrics.pr_auc:.3f}" if metrics is not None else "n/a"
    print(f"Champion   version {champion.version} ({champion.best_candidate}), holdout PR-AUC {pr_auc}, "
          f"threshold {result.threshold:.3f}", file=out)
    print(f"Request    {len(result.predictions)} customers, request id {result.request_id}, {seconds:.2f} s, "
          f"model version {result.model_version}", file=out)
    print(file=out)
    _print_table(customers[TARGET_COLUMN].tolist(), result.predictions, out)
    return 0


def _print_table(actual: Sequence[int], predictions: Sequence[Any], out: TextIO) -> None:
    print(f"{'ID':>6}  {'probability':>11}  {'label':>5}  {'Response in file':>16}", file=out)
    for prediction, response in zip(predictions, actual, strict=True):
        print(f"{prediction.ID:>6}  {prediction.probability:>11.3f}  {prediction.label:>5}  {response:>16}", file=out)

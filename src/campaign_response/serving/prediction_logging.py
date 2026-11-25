"""Recording each /predict request: rows in the prediction log, and one MLflow run with the request's artifacts."""

import tempfile
from datetime import datetime
from pathlib import Path
from typing import Final

import pandas as pd

from campaign_response.data.schema import FEATURE_COLUMNS, ID_COLUMN
from campaign_response.logging_setup import correlation_id_var
from campaign_response.registry.ports import ExperimentTracker
from campaign_response.serving.predict_service import ScoredRecords
from campaign_response.serving.schemas import PredictRequest
from campaign_response.storage.prediction_store import PredictionRecord

INFERENCE_EXPERIMENT: Final = "campaign-response-inference"
REQUEST_ARTIFACT: Final = "request.json"
PREDICTIONS_FILE: Final = "predictions.csv"


def prediction_records(
    request_id: str, created_at: datetime, request: PredictRequest, scored: ScoredRecords
) -> list[PredictionRecord]:
    """One prediction-log row per record, with the raw features as sent."""
    return [
        PredictionRecord(
            request_id=request_id,
            created_at=created_at,
            model_version=scored.model_version.version,
            source=request.source,
            customer_id=record.ID,
            features=record.model_dump(include=set(FEATURE_COLUMNS)),
            probability=probability,
            label=label,
        )
        for record, probability, label in zip(request.records, scored.probabilities, scored.labels, strict=True)
    ]


def log_inference_run(
    tracker: ExperimentTracker, request_id: str, request: PredictRequest, scored: ScoredRecords
) -> None:
    """One MLflow run per request: tags for lineage, summary metrics, the request and the predictions.

    Runs after the response has been sent (a background task), so it sets the request id for its own log lines.
    """
    token = correlation_id_var.set(request_id)
    try:
        with tracker.run(INFERENCE_EXPERIMENT, f"predict-{request_id}") as run_id:
            tracker.set_tags(run_id, {
                "request_id": request_id,
                "model_name": scored.model_version.name,
                "model_version": str(scored.model_version.version),
                "source": request.source,
            })
            tracker.log_metrics(run_id, {
                "n_records": len(scored.probabilities),
                "mean_probability": sum(scored.probabilities) / len(scored.probabilities),
                "positive_rate": sum(scored.labels) / len(scored.labels),
                "threshold": scored.threshold,
            })
            tracker.log_dict(run_id, request.model_dump(mode="json"), REQUEST_ARTIFACT)
            predictions = pd.DataFrame({
                ID_COLUMN: pd.array([record.ID for record in request.records], dtype="Int64"),
                "probability": scored.probabilities,
                "label": scored.labels,
            })
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / PREDICTIONS_FILE
                predictions.to_csv(path, index=False, lineterminator="\n")
                tracker.log_file(run_id, path)
    finally:
        correlation_id_var.reset(token)

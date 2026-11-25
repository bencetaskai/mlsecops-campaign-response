"""One drift check: the champion's reference against the latest requests, the retraining rule, and an MLflow run.

The API endpoint and, through it, the Airflow DAG use this; it depends only on the ports, so tests run on fakes.
"""

import logging
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Final

from campaign_response.data.cleaning import CleaningParams
from campaign_response.monitoring.current import load_current
from campaign_response.monitoring.drift import DriftResult, build_drift_report
from campaign_response.monitoring.reference import load_reference
from campaign_response.monitoring.retraining import MonitoringParams, RetrainingAdvice, recommend_retraining
from campaign_response.registry.ports import ExperimentTracker, ModelRegistry
from campaign_response.storage.prediction_store import PredictionStore

logger = logging.getLogger(__name__)

MONITORING_EXPERIMENT: Final = "campaign-response-monitoring"
REPORT_FILE: Final = "drift_report.html"
SUMMARY_FILE: Final = "drift_summary.json"
NO_PREDICTIONS: Final = "no predictions logged yet"


@dataclass(frozen=True)
class DriftCheck:
    reference_version: int
    n_reference: int
    n_current: int
    threshold: float
    column_scores: dict[str, float]
    drifted_columns: list[str]
    share_drifted: float
    dataset_drift: bool
    retrain_recommended: bool
    reason: str
    report_run_id: str | None

    def summary(self) -> dict[str, object]:
        """Everything but the run id: the content of drift_summary.json, logged before the id exists."""
        return {key: value for key, value in asdict(self).items() if key != "report_run_id"}


class DriftService:
    def __init__(
        self, registry: ModelRegistry, tracker: ExperimentTracker, store: PredictionStore, cleaning: CleaningParams,
        params: MonitoringParams,
    ) -> None:
        self._registry = registry
        self._tracker = tracker
        self._store = store
        self._cleaning = cleaning
        self._params = params

    def check(self, window: int, model_version: int | None = None) -> DriftCheck:
        """Compare the last `window` logged requests with the champion's training data.

        ReferenceUnavailableError without a champion or its reference sample; registry errors propagate. An empty
        prediction log is a normal state on a fresh stack, so it gives a check without report, not an error.
        """
        reference = load_reference(self._registry)
        current = load_current(self._store, window, self._cleaning, model_version)
        if current.empty:
            return DriftCheck(
                reference_version=reference.version.version, n_reference=len(reference.data), n_current=0,
                threshold=self._params.drift_threshold, column_scores={}, drifted_columns=[], share_drifted=0.0,
                dataset_drift=False, retrain_recommended=False, reason=NO_PREDICTIONS, report_run_id=None,
            )
        result = build_drift_report(reference.data, current, self._params.drift_threshold)
        advice = recommend_retraining(result.n_current, len(result.drifted_columns), self._params)
        check = _drift_check(reference.version.version, result, advice)
        return replace(check, report_run_id=self._log_run(check, result.html, window, model_version))

    def _log_run(self, check: DriftCheck, html: str, window: int, model_version: int | None) -> str | None:
        """The report as an MLflow run; None if logging fails, because the drift answer is valid without it."""
        try:
            with self._tracker.run(MONITORING_EXPERIMENT, f"drift-v{check.reference_version}") as run_id:
                self._tracker.set_tags(run_id, {
                    "reference_version": str(check.reference_version),
                    "window": str(window),
                    "model_version_filter": "all" if model_version is None else str(model_version),
                    "dataset_drift": str(check.dataset_drift).lower(),
                    "retrain_recommended": str(check.retrain_recommended).lower(),
                })
                self._tracker.log_metrics(run_id, {
                    "n_current": check.n_current,
                    "n_reference": check.n_reference,
                    "n_drifted": len(check.drifted_columns),
                    "share_drifted": check.share_drifted,
                    **{f"drift_{column}": score for column, score in check.column_scores.items()},
                })
                self._tracker.log_dict(run_id, check.summary(), SUMMARY_FILE)
                with tempfile.TemporaryDirectory() as folder:
                    path = Path(folder) / REPORT_FILE
                    path.write_text(html, encoding="utf-8")
                    self._tracker.log_file(run_id, path)
        # Broad on purpose, as for the inference runs; only the type is logged, an MLflow message can contain the URI.
        except Exception as error:
            logger.error("drift report logging failed; drift result returned without a run",
                         extra={"error_type": type(error).__name__})
            return None
        return run_id


def _drift_check(reference_version: int, result: DriftResult, advice: RetrainingAdvice) -> DriftCheck:
    return DriftCheck(
        reference_version=reference_version,
        n_reference=result.n_reference,
        n_current=result.n_current,
        threshold=result.threshold,
        column_scores=result.column_scores,
        drifted_columns=result.drifted_columns,
        share_drifted=result.share_drifted,
        dataset_drift=advice.dataset_drift,
        retrain_recommended=advice.retrain_recommended,
        reason=advice.reason,
        report_run_id=None,
    )

"""One training run: raw file -> validation -> cleaning -> split -> candidate search -> MLflow -> challenger.

Everything with side effects arrives through TrainingContext, so the same code runs against MLflow in production and
against the in-memory fakes in tests.
"""

import logging
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from campaign_response.data.cleaning import clean
from campaign_response.data.io import load_raw
from campaign_response.data.schema import ID_COLUMN, TARGET_COLUMN
from campaign_response.data.splitting import split_train_holdout
from campaign_response.data.validation import validate_raw
from campaign_response.modeling.evaluation import Metrics
from campaign_response.modeling.plots import evaluation_figures
from campaign_response.modeling.training import TrainingResult, evaluate_model, train_candidates
from campaign_response.params import PipelineParams
from campaign_response.registry.lineage import TrainedBy, lineage_tags
from campaign_response.registry.ports import CHALLENGER, ExperimentTracker, ModelRegistry, ModelVersion

logger = logging.getLogger(__name__)

EXPERIMENT = "campaign-response-training"


class DataValidationError(ValueError):
    """The raw data failed the validation gate; the run is recorded as failed and nothing is registered."""


@dataclass(frozen=True)
class TrainingRequest:
    candidates: tuple[str, ...] = ("logistic_regression", "random_forest", "hist_gradient_boosting")
    n_iter: int = 10
    cv_folds: int = 5
    seed: int = 42
    trained_by: TrainedBy = "cli"


@dataclass(frozen=True)
class TrainingContext:
    tracker: ExperimentTracker
    registry: ModelRegistry
    params: PipelineParams
    raw_data_path: Path
    requirements_path: Path
    environ: Mapping[str, str] = field(default_factory=lambda: dict(os.environ))


@dataclass(frozen=True)
class TrainingOutcome:
    run_id: str
    version: ModelVersion
    best_candidate: str
    threshold: float
    holdout: Metrics


def flatten(values: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """Nested settings as dotted keys (gate.min_recall), since MLflow params are a flat mapping."""
    flat: dict[str, Any] = {}
    for key, value in values.items():
        name = f"{prefix}{key}"
        if isinstance(value, Mapping):
            flat.update(flatten(value, f"{name}."))
        else:
            flat[name] = value
    return flat


def split_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Model input and target: ID stays in the data as the lineage key but never reaches the model."""
    return frame.drop(columns=[ID_COLUMN, TARGET_COLUMN]), frame[TARGET_COLUMN].astype(int)


def run_training(request: TrainingRequest, context: TrainingContext) -> TrainingOutcome:
    tracker = context.tracker
    settings = {"request": asdict(request), **asdict(context.params)}
    with tracker.run(EXPERIMENT, f"train-{request.trained_by}") as run_id:
        tracker.set_tags(run_id, lineage_tags(
            data_path=context.raw_data_path, params=settings, trained_by=request.trained_by,
            requirements_path=context.requirements_path, environ=context.environ,
        ))
        tracker.log_params(run_id, flatten(settings))

        raw = load_raw(context.raw_data_path)
        validation = validate_raw(raw)
        tracker.log_dict(run_id, {"passed": validation.passed, **asdict(validation)}, "validation_report.json")
        if not validation.passed:
            raise DataValidationError(f"raw data failed {len(validation.failures)} validation check(s)")

        cleaned, cleaning_report = clean(raw, context.params.cleaning)
        tracker.log_dict(run_id, asdict(cleaning_report), "cleaning_report.json")
        train, holdout = split_train_holdout(cleaned, context.params.split)

        result = train_candidates(
            *split_features(train), request.candidates, n_iter=request.n_iter, cv_folds=request.cv_folds,
            seed=request.seed,
        )
        _log_candidates(tracker, run_id, result)

        holdout_features, holdout_target = split_features(holdout)
        holdout_metrics = evaluate_model(result.model, holdout_features, holdout_target)
        tracker.log_metrics(run_id, {
            "train_rows": len(train),
            "holdout_rows": len(holdout),
            "cv_pr_auc": result.best.best.mean_pr_auc,
            "cv_pr_auc_std": result.best.best.std_pr_auc,
            "baseline_cv_pr_auc": result.baseline.best.mean_pr_auc,
            **{f"holdout_{name}": value for name, value in holdout_metrics.as_dict().items()},
        })
        scores = result.model.predict_proba(holdout_features)[:, 1]
        for name, figure in evaluation_figures(holdout_target, scores, result.threshold).items():
            tracker.log_figure(run_id, figure, f"plots/{name}")

        model_uri = tracker.log_model(run_id, result.model, holdout_features.head(5))
        version = context.registry.register(model_uri, run_id)
        context.registry.set_alias(CHALLENGER, version.version)
        tracker.set_tags(run_id, {"best_candidate": result.best.name, "model_version": str(version.version)})

    logger.info("training run finished", extra={"run_id": run_id, "model_version": version.version,
                                                **result.summary()})
    return TrainingOutcome(run_id, version, result.best.name, result.threshold, holdout_metrics)


def _log_candidates(tracker: ExperimentTracker, parent_run_id: str, result: TrainingResult) -> None:
    """One nested run per candidate, so the MLflow UI compares them side by side; all scores also as one JSON."""
    for candidate in result.candidates:
        with tracker.run(EXPERIMENT, candidate.name, parent_run_id) as child_id:
            tracker.log_params(child_id, {"candidate": candidate.name, **candidate.best.parameters})
            tracker.log_metrics(child_id, {
                "cv_pr_auc": candidate.best.mean_pr_auc,
                "cv_pr_auc_std": candidate.best.std_pr_auc,
                "parameter_sets": len(candidate.scores),
            })
            tracker.set_tags(child_id, {"selected": str(candidate.name == result.best.name).lower()})
    tracker.log_dict(parent_run_id, {
        "selected": result.best.name,
        "candidates": [
            {"name": candidate.name, "parameter_sets": [asdict(score) for score in candidate.scores]}
            for candidate in result.candidates
        ],
    }, "cv_results.json")

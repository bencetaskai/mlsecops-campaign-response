"""Cross-validated search over the allow-listed candidates, selected by PR-AUC."""

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold, TunedThresholdClassifierCV
from sklearn.pipeline import Pipeline

from campaign_response.features.engineering import DEFAULT_REFERENCE_DATE
from campaign_response.features.preprocessing import build_preprocessor
from campaign_response.features.transformer import FeatureEngineer
from campaign_response.modeling.candidates import BASELINE, Candidate, CandidateName, get_candidate, parameter_sets
from campaign_response.modeling.evaluation import Metrics, compute_metrics

logger = logging.getLogger(__name__)

MODEL_STEP = "model"
# sklearn has no annotations, so Pyright types error_score from its default (np.nan) as float; "raise" is valid
# and makes a failing fit stop the search instead of silently scoring NaN.
RAISE_ON_FIT_ERROR: Any = "raise"


def build_pipeline(
    candidate: Candidate,
    seed: int,
    parameters: Mapping[str, Any] | None = None,
    reference_date: date = DEFAULT_REFERENCE_DATE,
) -> Pipeline:
    """Feature engineering, preprocessing and the estimator as one object, so CV refits all of it per fold.

    parameters are the estimator's own names, as in the candidate search space.
    """
    return Pipeline([
        ("features", FeatureEngineer(reference_date=reference_date)),
        ("preprocess", build_preprocessor()),
        (MODEL_STEP, candidate.build(seed).set_params(**(parameters or {}))),
    ])


@dataclass(frozen=True)
class ParameterSetScore:
    parameters: dict[str, Any]
    fold_pr_aucs: tuple[float, ...]

    @property
    def mean_pr_auc(self) -> float:
        return float(np.mean(self.fold_pr_aucs))

    @property
    def std_pr_auc(self) -> float:
        return float(np.std(self.fold_pr_aucs))


@dataclass(frozen=True)
class CandidateResult:
    """Scores of every parameter set tried for one candidate, in the order they were tried (defaults first)."""

    name: CandidateName
    scores: tuple[ParameterSetScore, ...]

    @property
    def best(self) -> ParameterSetScore:
        # max() keeps the first of equal scores, so a tie goes to the defaults.
        return max(self.scores, key=lambda score: score.mean_pr_auc)


def search_candidate(
    candidate: Candidate,
    features: pd.DataFrame,
    target: pd.Series,
    *,
    n_iter: int,
    cv_folds: int,
    seed: int,
    n_jobs: int | None = None,
) -> CandidateResult:
    """Score each parameter set of the candidate with the same stratified folds; nothing is refitted on all data.

    The folds depend only on the seed, so every candidate and parameter set is compared on identical splits.
    """
    sets = parameter_sets(candidate, n_iter=n_iter, seed=seed)
    # One single-point grid per set: GridSearchCV then evaluates exactly the chosen sets, in this order.
    grid = [{f"{MODEL_STEP}__{key}": [value] for key, value in point.items()} for point in sets]
    search = GridSearchCV(
        build_pipeline(candidate, seed),
        grid,
        scoring="average_precision",
        cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed),
        refit=False,
        error_score=RAISE_ON_FIT_ERROR,
        n_jobs=n_jobs,
    )
    search.fit(features, target)
    fold_scores = np.column_stack([search.cv_results_[f"split{fold}_test_score"] for fold in range(cv_folds)])
    result = CandidateResult(
        name=candidate.name,
        scores=tuple(
            ParameterSetScore(parameters=point, fold_pr_aucs=tuple(float(score) for score in row))
            for point, row in zip(sets, fold_scores, strict=True)
        ),
    )
    logger.info(
        "candidate searched",
        extra={
            "candidate": result.name,
            "parameter_sets": len(sets),
            "best_cv_pr_auc": round(result.best.mean_pr_auc, 4),
        },
    )
    return result


def select_best(results: Sequence[CandidateResult]) -> CandidateResult:
    """The candidate with the highest mean CV PR-AUC; the baseline only serves as the reference and is never chosen.

    Ties go to the earlier result, so the outcome does not depend on floating-point noise between equal models.
    """
    contenders = [result for result in results if result.name != BASELINE]
    if not contenders:
        raise ValueError("no candidate besides the baseline was searched")
    return max(contenders, key=lambda result: result.best.mean_pr_auc)


def fit_with_tuned_threshold(
    candidate: Candidate,
    parameters: Mapping[str, Any],
    features: pd.DataFrame,
    target: pd.Series,
    *,
    cv_folds: int,
    seed: int,
    n_jobs: int | None = None,
) -> TunedThresholdClassifierCV:
    """Fit the chosen candidate on all training data with the F1-optimal decision threshold.

    The threshold is chosen on out-of-fold scores of the training data only (the holdout stays untouched), then the
    pipeline is refitted on all of it. The result is one estimator: predict() applies the threshold and
    predict_proba() still returns the raw scores.
    """
    model = TunedThresholdClassifierCV(
        build_pipeline(candidate, seed, parameters),
        scoring="f1",
        cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed),
        refit=True,
        n_jobs=n_jobs,
    )
    model.fit(features, target)
    threshold = round(decision_threshold(model), 4)
    logger.info("threshold tuned", extra={"candidate": candidate.name, "threshold": threshold})
    return model


def decision_threshold(model: TunedThresholdClassifierCV) -> float:
    return float(model.best_threshold_)


def evaluate_model(model: TunedThresholdClassifierCV, features: pd.DataFrame, target: pd.Series) -> Metrics:
    """Metrics of a fitted model on new data, at the model's own threshold."""
    return compute_metrics(target, model.predict_proba(features)[:, 1], decision_threshold(model))


@dataclass(frozen=True)
class TrainingResult:
    model: TunedThresholdClassifierCV
    best: CandidateResult
    candidates: tuple[CandidateResult, ...]

    @property
    def threshold(self) -> float:
        return decision_threshold(self.model)

    @property
    def baseline(self) -> CandidateResult:
        return next(result for result in self.candidates if result.name == BASELINE)

    def summary(self) -> dict[str, Any]:
        return {
            "best_candidate": self.best.name,
            "best_parameters": self.best.best.parameters,
            "cv_pr_auc": self.best.best.mean_pr_auc,
            "baseline_cv_pr_auc": self.baseline.best.mean_pr_auc,
            "threshold": self.threshold,
        }


def train_candidates(
    features: pd.DataFrame,
    target: pd.Series,
    candidates: Sequence[str],
    *,
    n_iter: int,
    cv_folds: int,
    seed: int,
    n_jobs: int | None = None,
) -> TrainingResult:
    """Search the requested candidates plus the baseline, select the best and fit it with a tuned threshold.

    Names are checked against the allow-list before any training, so a bad request fails in milliseconds.
    """
    names = list(dict.fromkeys([BASELINE, *candidates]))
    chosen = [get_candidate(name) for name in names]
    if len(chosen) < 2:
        raise ValueError("request at least one candidate besides the baseline")

    results = tuple(
        search_candidate(candidate, features, target, n_iter=n_iter, cv_folds=cv_folds, seed=seed, n_jobs=n_jobs)
        for candidate in chosen
    )
    best = select_best(results)
    model = fit_with_tuned_threshold(
        get_candidate(best.name), best.best.parameters, features, target, cv_folds=cv_folds, seed=seed, n_jobs=n_jobs
    )
    result = TrainingResult(model=model, best=best, candidates=results)
    logger.info("training finished", extra=result.summary())
    return result

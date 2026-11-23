"""Cross-validated search over the allow-listed candidates, selected by PR-AUC."""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from campaign_response.features.engineering import DEFAULT_REFERENCE_DATE
from campaign_response.features.preprocessing import build_preprocessor
from campaign_response.features.transformer import FeatureEngineer
from campaign_response.modeling.candidates import BASELINE, Candidate, CandidateName, parameter_sets

logger = logging.getLogger(__name__)

MODEL_STEP = "model"
# sklearn has no annotations, so Pyright types error_score from its default (np.nan) as float; "raise" is valid
# and makes a failing fit stop the search instead of silently scoring NaN.
RAISE_ON_FIT_ERROR: Any = "raise"


def build_pipeline(candidate: Candidate, seed: int, reference_date: date = DEFAULT_REFERENCE_DATE) -> Pipeline:
    """Feature engineering, preprocessing and the estimator as one object, so CV refits all of it per fold."""
    return Pipeline([
        ("features", FeatureEngineer(reference_date=reference_date)),
        ("preprocess", build_preprocessor()),
        (MODEL_STEP, candidate.build(seed)),
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

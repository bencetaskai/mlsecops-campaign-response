"""Allow-list of candidate estimators and their hyperparameter search spaces.

Only names listed here can be trained: /train accepts a name, never a class path or parameters, so a request cannot
make the server import or configure arbitrary code. Search-space keys are the estimator's own parameter names;
training adds the pipeline step prefix.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, get_args

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterGrid

CandidateName = Literal["dummy", "logistic_regression", "random_forest", "hist_gradient_boosting"]
CANDIDATE_NAMES: tuple[CandidateName, ...] = get_args(CandidateName)
BASELINE: CandidateName = "dummy"


@dataclass(frozen=True)
class Candidate:
    name: CandidateName
    build: Callable[[int], BaseEstimator]
    search_space: Mapping[str, Sequence[Any]]

    def default_parameters(self) -> dict[str, Any]:
        """The values the estimator is built with, restricted to the searched parameters."""
        built = self.build(0).get_params()
        return {key: built[key] for key in self.search_space}


# Every space contains the built defaults (checked by a test), because the analysis on the train split found the
# defaults as good as any tuned setting; the search may still find better ones on more data.
CANDIDATES: Mapping[str, Candidate] = MappingProxyType({
    "dummy": Candidate("dummy", lambda seed: DummyClassifier(strategy="prior"), {}),
    "logistic_regression": Candidate(
        "logistic_regression",
        lambda seed: LogisticRegression(class_weight="balanced", max_iter=2000),
        {"C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]},
    ),
    "random_forest": Candidate(
        "random_forest",
        lambda seed: RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample", random_state=seed),
        {"max_depth": [None, 4, 8, 12], "min_samples_leaf": [1, 5, 10, 20], "max_features": ["sqrt", 0.5]},
    ),
    "hist_gradient_boosting": Candidate(
        "hist_gradient_boosting",
        lambda seed: HistGradientBoostingClassifier(class_weight="balanced", random_state=seed),
        {
            "learning_rate": [0.03, 0.1, 0.3],
            "max_leaf_nodes": [15, 31, 63],
            "min_samples_leaf": [10, 20, 40],
            "l2_regularization": [0.0, 0.1, 1.0],
        },
    ),
})


class UnknownCandidateError(ValueError):
    """A candidate name outside the allow-list."""


def get_candidate(name: str) -> Candidate:
    try:
        return CANDIDATES[name]
    except KeyError:
        raise UnknownCandidateError(
            f"unknown candidate {name!r}; allowed: {', '.join(CANDIDATE_NAMES)}"
        ) from None


def parameter_sets(candidate: Candidate, n_iter: int, seed: int) -> list[dict[str, Any]]:
    """The defaults first, then up to n_iter - 1 other grid points drawn without replacement.

    The result is capped at the grid size, so asking for more combinations than exist costs nothing extra.
    """
    if n_iter < 1:
        raise ValueError(f"n_iter must be at least 1, got {n_iter}")
    defaults = candidate.default_parameters()
    others = [dict(point) for point in ParameterGrid(dict(candidate.search_space)) if point != defaults]
    count = min(n_iter - 1, len(others))
    chosen = np.random.default_rng(seed).choice(len(others), size=count, replace=False) if count else []
    return [defaults, *(others[index] for index in chosen)]

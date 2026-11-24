"""The quality gate: a pure function that decides whether a challenger replaces the champion.

Both models are scored on the same current holdout. The comparison (G2) uses a paired bootstrap instead of a fixed
margin: on 470 holdout rows two models of equal quality differ in PR-AUC with sd 0.027, so a fixed margin such as
+0.005 decides by noise. A simulation on the training split (2026-11-24) gave these promotion rates:

    rule                                equal models   better model   clearly worse model
    diff >= +0.005                           43%            77%              3%
    diff >= -0.01 (lecture tolerance)        73%            87%             13%
    bootstrap 10th percentile > -0.02        33%            73%              0%
"""

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import average_precision_score

from campaign_response.modeling.evaluation import Metrics, check_inputs, compute_metrics


@dataclass(frozen=True)
class GateParams:
    min_pr_auc: float = 0.30
    prevalence_margin: float = 0.10
    max_roc_auc_drop: float = 0.02
    min_recall: float = 0.30
    min_prediction_rate: float = 0.02
    max_prediction_rate: float = 0.60
    bootstrap_resamples: int = 1000
    bootstrap_quantile: float = 0.10
    max_pr_auc_loss: float = 0.02
    seed: int = 42


@dataclass(frozen=True)
class GateCheck:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class GateDecision:
    promoted: bool
    checks: tuple[GateCheck, ...]
    challenger: Metrics
    champion: Metrics | None

    @property
    def reason(self) -> str:
        """The failed checks, or all checks when the challenger is promoted; for the MLflow tag and the API."""
        shown = [check for check in self.checks if not check.passed] or list(self.checks)
        return "; ".join(f"{check.name}: {check.detail}" for check in shown)

    def as_dict(self) -> dict[str, Any]:
        """The content of gate_decision.json."""
        return {
            "promoted": self.promoted,
            "reason": self.reason,
            "checks": [asdict(check) for check in self.checks],
            "challenger": self.challenger.as_dict(),
            "champion": self.champion.as_dict() if self.champion else None,
        }


def paired_bootstrap_quantile(
    y_true: ArrayLike, challenger: ArrayLike, champion: ArrayLike, *, resamples: int, quantile: float, seed: int
) -> float:
    """Quantile of PR-AUC(challenger) - PR-AUC(champion) over bootstrap resamples of the same rows.

    Resampling rows for both models at once keeps the comparison paired: rows that are hard for one model are hard
    for the other, which cancels much of the noise that comparing two independent scores would carry.
    """
    truth, challenger_scores = check_inputs(y_true, challenger)
    _, champion_scores = check_inputs(y_true, champion)
    rng = np.random.default_rng(seed)
    differences: list[float] = []
    for rows in rng.integers(0, len(truth), size=(resamples, len(truth))):
        sample = truth[rows]
        if sample.min() == sample.max():
            continue  # PR-AUC is undefined without both classes; only possible on tiny holdouts
        differences.append(
            float(average_precision_score(sample, challenger_scores[rows])
                  - average_precision_score(sample, champion_scores[rows]))
        )
    if not differences:
        raise ValueError("no bootstrap resample contained both classes")
    return float(np.quantile(differences, quantile))


def decide_promotion(
    y_true: ArrayLike,
    challenger_scores: ArrayLike,
    challenger_threshold: float,
    champion_scores: ArrayLike | None = None,
    champion_threshold: float | None = None,
    params: GateParams = GateParams(),
) -> GateDecision:
    """Promote only if every check passes. Without a champion, the comparison checks (G2, ROC-AUC guard) are skipped.

    G1 floor: PR-AUC clearly above the always-no baseline. G2: no likely PR-AUC loss against the champion.
    G3 guards: no ROC-AUC collapse, recall not collapsed. G4 sanity: a plausible share of predicted responders.
    """
    challenger = compute_metrics(y_true, challenger_scores, challenger_threshold)
    floor = max(params.min_pr_auc, challenger.prevalence + params.prevalence_margin)
    prediction_rate = (challenger.true_positives + challenger.false_positives) / challenger.n_samples
    checks = [
        GateCheck("floor", challenger.pr_auc >= floor, f"pr_auc {challenger.pr_auc:.3f} vs floor {floor:.3f}"),
        GateCheck("recall", challenger.recall >= params.min_recall,
                  f"recall {challenger.recall:.3f} vs minimum {params.min_recall:.3f}"),
        GateCheck("prediction_rate",
                  params.min_prediction_rate <= prediction_rate <= params.max_prediction_rate,
                  f"predicted responders {prediction_rate:.3f} vs "
                  f"[{params.min_prediction_rate:.2f}, {params.max_prediction_rate:.2f}]"),
    ]

    champion: Metrics | None = None
    if champion_scores is not None:
        if champion_threshold is None:
            raise ValueError("champion_scores need a champion_threshold")
        champion = compute_metrics(y_true, champion_scores, champion_threshold)
        loss_bound = paired_bootstrap_quantile(
            y_true, challenger_scores, champion_scores,
            resamples=params.bootstrap_resamples, quantile=params.bootstrap_quantile, seed=params.seed,
        )
        roc_floor = champion.roc_auc - params.max_roc_auc_drop
        checks += [
            GateCheck("no_regression", loss_bound > -params.max_pr_auc_loss,
                      f"pr_auc {challenger.pr_auc:.3f} vs champion {champion.pr_auc:.3f}; "
                      f"{params.bootstrap_quantile:.0%} bootstrap bound of the difference {loss_bound:+.3f} "
                      f"vs {-params.max_pr_auc_loss:+.3f}"),
            GateCheck("roc_auc", challenger.roc_auc >= roc_floor,
                      f"roc_auc {challenger.roc_auc:.3f} vs champion {champion.roc_auc:.3f} - "
                      f"{params.max_roc_auc_drop:.2f}"),
        ]

    return GateDecision(all(check.passed for check in checks), tuple(checks), challenger, champion)

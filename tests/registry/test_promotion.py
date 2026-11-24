"""Every branch of the promotion gate, on synthetic holdout scores with a known quality order."""

import json

import numpy as np
import pytest
from numpy.typing import NDArray

from campaign_response.registry.promotion import GateParams, decide_promotion, paired_bootstrap_quantile

RNG = np.random.default_rng(0)
LABELS = np.array([1] * 60 + [0] * 340)
NOISE = RNG.normal(0, 0.15, len(LABELS))
FAST = GateParams(bootstrap_resamples=200)


def scores(separation: float, noise: NDArray[np.float64] = NOISE) -> NDArray[np.float64]:
    """Positive-class scores whose quality grows with separation; 0 gives a model without skill."""
    return np.clip(0.3 + separation * LABELS + noise, 0, 1)


GOOD, BETTER, WORSE = scores(0.35), scores(0.5), scores(0.15)


def test_gate_first_model_promoted() -> None:
    decision = decide_promotion(LABELS, GOOD, 0.5, params=FAST)

    assert decision.promoted
    assert [check.name for check in decision.checks] == ["floor", "recall", "prediction_rate"]
    assert decision.champion is None


def test_gate_no_skill_fails_floor() -> None:
    decision = decide_promotion(LABELS, scores(0.0, RNG.uniform(-0.3, 0.7, len(LABELS))), 0.5, params=FAST)

    assert not decision.promoted
    assert decision.reason.startswith("floor: pr_auc")


def test_gate_floor_rises_with_prevalence() -> None:
    decision = decide_promotion(LABELS, GOOD, 0.5, params=GateParams(min_pr_auc=0.0, prevalence_margin=0.2))

    assert "floor 0.350" in decision.checks[0].detail


def test_gate_recall_collapse_fails() -> None:
    decision = decide_promotion(LABELS, GOOD, 0.9, params=FAST)

    failed = [check.name for check in decision.checks if not check.passed]
    assert "recall" in failed
    assert not decision.promoted


def test_gate_everyone_predicted_fails_sanity() -> None:
    decision = decide_promotion(LABELS, GOOD, 0.0, params=FAST)

    assert [check.name for check in decision.checks if not check.passed] == ["prediction_rate"]


def test_gate_equal_challenger_promoted() -> None:
    decision = decide_promotion(LABELS, GOOD, 0.5, GOOD, 0.5, params=FAST)

    assert decision.promoted
    assert [check.name for check in decision.checks][-2:] == ["no_regression", "roc_auc"]


def test_gate_better_challenger_promoted() -> None:
    assert decide_promotion(LABELS, BETTER, 0.5, GOOD, 0.5, params=FAST).promoted


def test_gate_worse_challenger_rejected() -> None:
    decision = decide_promotion(LABELS, WORSE, 0.45, GOOD, 0.5, params=FAST)

    assert not decision.promoted
    assert "no_regression" in decision.reason
    assert decision.champion is not None and decision.champion.pr_auc > decision.challenger.pr_auc


def test_gate_roc_guard_fails_alone() -> None:
    params = GateParams(bootstrap_resamples=200, max_pr_auc_loss=1.0, max_roc_auc_drop=0.0)

    decision = decide_promotion(LABELS, WORSE, 0.45, GOOD, 0.5, params=params)

    assert [check.name for check in decision.checks if not check.passed] == ["roc_auc"]


def test_gate_champion_without_threshold_ValueError() -> None:
    with pytest.raises(ValueError, match="champion_threshold"):
        decide_promotion(LABELS, GOOD, 0.5, GOOD)


def test_gate_same_seed_same_decision() -> None:
    first = decide_promotion(LABELS, BETTER, 0.5, GOOD, 0.5, params=FAST)

    assert decide_promotion(LABELS, BETTER, 0.5, GOOD, 0.5, params=FAST) == first


def test_gate_decision_json_serialisable() -> None:
    decision = decide_promotion(LABELS, WORSE, 0.45, GOOD, 0.5, params=FAST)

    content = json.loads(json.dumps(decision.as_dict()))

    assert content["promoted"] is False
    assert content["reason"] == decision.reason
    assert decision.champion is not None
    assert content["champion"]["pr_auc"] == decision.champion.pr_auc
    assert len(content["checks"]) == 5


def test_gate_promoted_reason_lists_all_checks() -> None:
    decision = decide_promotion(LABELS, GOOD, 0.5, params=FAST)

    assert decision.reason.count(";") == len(decision.checks) - 1


def test_bootstrap_identical_scores_zero() -> None:
    assert paired_bootstrap_quantile(LABELS, GOOD, GOOD, resamples=50, quantile=0.1, seed=1) == 0


def test_bootstrap_better_challenger_positive_bound() -> None:
    assert paired_bootstrap_quantile(LABELS, BETTER, WORSE, resamples=200, quantile=0.1, seed=1) > 0


def test_bootstrap_single_class_resamples_ValueError() -> None:
    # Seed 0 draws the same row twice, so the only resample holds one class.
    with pytest.raises(ValueError, match="both classes"):
        paired_bootstrap_quantile([0, 1], [0.2, 0.8], [0.3, 0.7], resamples=1, quantile=0.1, seed=0)

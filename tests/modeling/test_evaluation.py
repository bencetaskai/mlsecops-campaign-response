"""Tests for the shared metric computation, checked against values computed by hand."""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score, precision_score, recall_score

from campaign_response.modeling.evaluation import compute_metrics

LABELS = [0, 0, 1, 1]
SCORES = [0.1, 0.4, 0.35, 0.8]


def test_metrics_hand_computed_values() -> None:
    metrics = compute_metrics(LABELS, SCORES, threshold=0.5)

    # Ranked: 0.8 (pos), 0.4 (neg), 0.35 (pos), 0.1 (neg) -> AP = 0.5 * 1 + 0.5 * 2/3; predictions [0, 0, 0, 1].
    assert metrics.pr_auc == pytest.approx(5 / 6)
    assert metrics.roc_auc == pytest.approx(0.75)
    assert (metrics.precision, metrics.recall) == (1.0, 0.5)
    assert metrics.f1 == pytest.approx(2 / 3)
    assert metrics.balanced_accuracy == pytest.approx(0.75)
    assert metrics.mcc == pytest.approx(2 / np.sqrt(12))
    assert metrics.brier == pytest.approx((0.01 + 0.16 + 0.4225 + 0.04) / 4)
    assert (metrics.prevalence, metrics.n_samples) == (0.5, 4)


def test_metrics_confusion_counts() -> None:
    metrics = compute_metrics(LABELS, SCORES, threshold=0.3)

    counts = (metrics.true_positives, metrics.false_positives, metrics.false_negatives, metrics.true_negatives)
    assert counts == (2, 1, 0, 1)


def test_metrics_threshold_inclusive() -> None:
    assert compute_metrics(LABELS, SCORES, threshold=0.35).true_positives == 2


def test_metrics_threshold_only_changes_decisions() -> None:
    low, high = compute_metrics(LABELS, SCORES, 0.2), compute_metrics(LABELS, SCORES, 0.9)

    assert (low.pr_auc, low.roc_auc, low.brier) == (high.pr_auc, high.roc_auc, high.brier)
    assert low.recall != high.recall


def test_metrics_no_positive_predictions_zero_no_warning() -> None:
    metrics = compute_metrics(LABELS, SCORES, threshold=1.0)

    assert (metrics.precision, metrics.recall, metrics.f1, metrics.mcc) == (0.0, 0.0, 0.0, 0.0)


def test_metrics_pandas_nullable_labels() -> None:
    labels = pd.Series(LABELS, dtype="Int64")

    assert compute_metrics(labels, np.array(SCORES), 0.5) == compute_metrics(LABELS, SCORES, 0.5)


def test_metrics_as_dict_json_serialisable() -> None:
    values = compute_metrics(LABELS, SCORES, 0.5).as_dict()

    assert json.loads(json.dumps(values)) == values
    assert values["threshold"] == 0.5


@pytest.mark.parametrize(
    ("labels", "scores", "message"),
    [
        ([0, 1, 1], [0.1, 0.2], "equally long"),
        ([[0, 1]], [[0.1, 0.2]], "1-D"),
        ([0, 2, 1], [0.1, 0.2, 0.3], "only 0 and 1"),
        (pd.Series([0, None, 1], dtype="Int64"), [0.1, 0.2, 0.3], "only 0 and 1"),
        ([1, 1, 1], [0.1, 0.2, 0.3], "both classes"),
        ([0, 1, 1], [0.1, 1.2, 0.3], r"within \[0, 1\]"),
        ([0, 1, 1], [0.1, np.nan, 0.3], "finite"),
    ],
)
def test_metrics_invalid_inputs_ValueError(labels: object, scores: list[float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        compute_metrics(labels, scores, 0.5)


@pytest.mark.parametrize("threshold", [-0.01, 1.01, float("nan")])
def test_metrics_threshold_out_of_range_ValueError(threshold: float) -> None:
    with pytest.raises(ValueError, match="threshold"):
        compute_metrics(LABELS, SCORES, threshold)


def test_metrics_match_sklearn_random_data() -> None:
    rng = np.random.default_rng(0)
    labels, scores = rng.integers(0, 2, 200), rng.random(200)
    predicted = (scores >= 0.3).astype(int)

    metrics = compute_metrics(labels, scores, 0.3)

    assert metrics.precision == pytest.approx(precision_score(labels, predicted))
    assert metrics.recall == pytest.approx(recall_score(labels, predicted))
    assert metrics.f1 == pytest.approx(f1_score(labels, predicted))

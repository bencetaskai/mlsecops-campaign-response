"""The one place where model quality is computed: CV, holdout, the promotion gate and the API all call this."""

from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    matthews_corrcoef,
    roc_auc_score,
)


@dataclass(frozen=True)
class Metrics:
    """Ranking metrics (pr_auc, roc_auc) ignore the threshold; the rest describe the decisions at the threshold.

    brier is computed on the raw scores, so with class weights it also shows how far they are from probabilities.
    """

    pr_auc: float
    roc_auc: float
    f1: float
    precision: float
    recall: float
    balanced_accuracy: float
    mcc: float
    brier: float
    threshold: float
    prevalence: float
    n_samples: int
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def check_inputs(y_true: ArrayLike, y_score: ArrayLike) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Return 0/1 labels and positive-class scores as arrays, or raise ValueError if the metrics are undefined.

    Raising instead of returning NaN matters for the promotion gate: any comparison with NaN is False, so a NaN score
    would silently pass or fail it.
    """
    labels = np.asarray(y_true, dtype=float)
    scores = np.asarray(y_score, dtype=float)
    if labels.ndim != 1 or labels.shape != scores.shape:
        raise ValueError(f"y_true and y_score must be 1-D and equally long, got {labels.shape} and {scores.shape}")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("y_true must contain only 0 and 1")
    if len(np.unique(labels)) < 2:
        raise ValueError("y_true must contain both classes; ranking metrics are undefined for a single class")
    if not (np.isfinite(scores).all() and scores.min() >= 0 and scores.max() <= 1):
        raise ValueError("y_score must be finite and within [0, 1]")
    return labels.astype(np.int64), scores


def check_threshold(threshold: float) -> None:
    if not 0 <= threshold <= 1:
        raise ValueError(f"threshold must be within [0, 1], got {threshold}")


def compute_metrics(y_true: ArrayLike, y_score: ArrayLike, threshold: float) -> Metrics:
    """Evaluate positive-class scores in [0, 1] against 0/1 labels; a score >= threshold predicts 1, as in sklearn."""
    truth, scores = check_inputs(y_true, y_score)
    check_threshold(threshold)

    predicted = (scores >= threshold).astype(int)
    tn, fp, fn, tp = (int(count) for count in confusion_matrix(truth, predicted, labels=[0, 1]).ravel())
    # Both classes are present, so tp + fn > 0; only precision needs a zero guard (no positive predictions).
    precision = tp / (tp + fp) if tp + fp else 0.0
    return Metrics(
        pr_auc=float(average_precision_score(truth, scores)),
        roc_auc=float(roc_auc_score(truth, scores)),
        f1=2 * tp / (2 * tp + fp + fn),
        precision=precision,
        recall=tp / (tp + fn),
        balanced_accuracy=float(balanced_accuracy_score(truth, predicted)),
        mcc=float(matthews_corrcoef(truth, predicted)),
        brier=float(brier_score_loss(truth, scores)),
        threshold=float(threshold),
        prevalence=float(truth.mean()),
        n_samples=len(truth),
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        true_negatives=tn,
    )

"""Evaluation plots as in-memory matplotlib figures, ready for mlflow.log_figure.

Figures are created with Figure() instead of pyplot: pyplot keeps every figure in a global registry until it is
closed, which leaks memory in a long-running API and is not thread-safe. A plain Figure needs no GUI backend and is
freed like any other object once the caller drops it.
"""

from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from numpy.typing import ArrayLike
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve

from campaign_response.modeling.evaluation import check_inputs, check_threshold

FIGURE_SIZE = (5.0, 4.0)


def pr_curve_figure(y_true: ArrayLike, y_score: ArrayLike) -> Figure:
    """Precision-recall curve with the prevalence line, the PR-AUC of a model without skill."""
    truth, scores = check_inputs(y_true, y_score)
    precision, recall, _ = precision_recall_curve(truth, scores)
    figure = Figure(figsize=FIGURE_SIZE)
    axes = figure.add_subplot()
    axes.step(recall, precision, where="post", label=f"model (PR-AUC {average_precision_score(truth, scores):.3f})")
    axes.axhline(truth.mean(), linestyle="--", color="grey", label=f"no skill ({truth.mean():.3f})")
    axes.set(xlabel="Recall", ylabel="Precision", title="Precision-recall curve", xlim=(0, 1), ylim=(0, 1.02))
    axes.legend(loc="lower left")
    return figure


def roc_curve_figure(y_true: ArrayLike, y_score: ArrayLike) -> Figure:
    truth, scores = check_inputs(y_true, y_score)
    false_positive_rate, true_positive_rate, _ = roc_curve(truth, scores)
    figure = Figure(figsize=FIGURE_SIZE)
    axes = figure.add_subplot()
    axes.plot(false_positive_rate, true_positive_rate, label=f"model (ROC-AUC {roc_auc_score(truth, scores):.3f})")
    axes.plot([0, 1], [0, 1], linestyle="--", color="grey", label="no skill")
    axes.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve", xlim=(0, 1), ylim=(0, 1))
    axes.legend(loc="lower right")
    return figure


def confusion_matrix_figure(y_true: ArrayLike, y_score: ArrayLike, threshold: float) -> Figure:
    truth, scores = check_inputs(y_true, y_score)
    check_threshold(threshold)
    counts = confusion_matrix(truth, (scores >= threshold).astype(int), labels=[0, 1])
    figure = Figure(figsize=FIGURE_SIZE)
    axes = figure.add_subplot()
    axes.imshow(counts, cmap="Blues")
    for (row, column), count in np.ndenumerate(counts):
        # White text on the dark cells, where the darkest cell holds the largest count.
        color = "white" if count > counts.max() / 2 else "black"
        axes.text(column, row, str(count), ha="center", va="center", color=color)
    axes.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["no response", "response"],
             yticklabels=["no response", "response"], xlabel="Predicted", ylabel="Actual",
             title=f"Confusion matrix at threshold {threshold:.2f}")
    return figure


def calibration_figure(y_true: ArrayLike, y_score: ArrayLike, n_bins: int = 10) -> Figure:
    """Observed response rate per score bin; scores inflated by class weights lie below the diagonal.

    Quantile bins hold equally many customers, so no bin is a handful of rare high scores.
    """
    truth, scores = check_inputs(y_true, y_score)
    observed, predicted = calibration_curve(truth, scores, n_bins=n_bins, strategy="quantile")
    figure = Figure(figsize=FIGURE_SIZE)
    axes = figure.add_subplot()
    axes.plot(predicted, observed, marker="o", label="model")
    axes.plot([0, 1], [0, 1], linestyle="--", color="grey", label="perfectly calibrated")
    axes.set(xlabel="Mean score in bin", ylabel="Observed response rate", title="Calibration", xlim=(0, 1),
             ylim=(0, 1))
    axes.legend(loc="upper left")
    return figure


def evaluation_figures(y_true: ArrayLike, y_score: ArrayLike, threshold: float) -> dict[str, Figure]:
    """All evaluation plots keyed by the file name they are logged under."""
    return {
        "pr_curve.png": pr_curve_figure(y_true, y_score),
        "roc_curve.png": roc_curve_figure(y_true, y_score),
        "confusion_matrix.png": confusion_matrix_figure(y_true, y_score, threshold),
        "calibration.png": calibration_figure(y_true, y_score),
    }


def save_figures(figures: dict[str, Figure], directory: Path) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for name, figure in figures.items():
        path = directory / name
        figure.savefig(path, dpi=100, bbox_inches="tight")
        paths.append(path)
    return paths

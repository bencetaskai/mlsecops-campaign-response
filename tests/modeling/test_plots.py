"""Tests for the evaluation figures."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure

from campaign_response.modeling.plots import (
    calibration_figure,
    confusion_matrix_figure,
    evaluation_figures,
    pr_curve_figure,
    save_figures,
)

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


@pytest.fixture
def scored() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, 200)
    return labels, np.clip(0.3 * labels + rng.random(200) * 0.7, 0, 1)


def test_evaluation_figures_four_named(scored: tuple[np.ndarray, np.ndarray]) -> None:
    figures = evaluation_figures(*scored, threshold=0.5)

    assert list(figures) == ["pr_curve.png", "roc_curve.png", "confusion_matrix.png", "calibration.png"]
    assert all(isinstance(figure, Figure) for figure in figures.values())


def test_evaluation_figures_not_in_pyplot_registry(scored: tuple[np.ndarray, np.ndarray]) -> None:
    before = plt.get_fignums()

    evaluation_figures(*scored, threshold=0.5)

    assert plt.get_fignums() == before


def test_pr_curve_shows_auc_and_prevalence(scored: tuple[np.ndarray, np.ndarray]) -> None:
    labels = scored[0]

    legend = [text.get_text() for text in pr_curve_figure(*scored).axes[0].get_legend().get_texts()]

    assert legend[0].startswith("model (PR-AUC 0.")
    assert legend[1] == f"no skill ({labels.mean():.3f})"


def test_confusion_matrix_counts_in_cells() -> None:
    figure = confusion_matrix_figure([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8], threshold=0.5)

    cells = [text.get_text() for text in figure.axes[0].texts]

    assert cells == ["2", "0", "1", "1"]
    assert figure.axes[0].get_title() == "Confusion matrix at threshold 0.50"


def test_calibration_one_point_per_bin(scored: tuple[np.ndarray, np.ndarray]) -> None:
    line = calibration_figure(*scored, n_bins=5).axes[0].get_lines()[0]

    assert len(line.get_xdata()) == 5


def test_save_figures_writes_png(scored: tuple[np.ndarray, np.ndarray], tmp_path: Path) -> None:
    paths = save_figures(evaluation_figures(*scored, threshold=0.5), tmp_path / "plots")

    assert [path.name for path in paths] == ["pr_curve.png", "roc_curve.png", "confusion_matrix.png", "calibration.png"]
    assert all(path.read_bytes().startswith(PNG_SIGNATURE) for path in paths)


def test_figures_invalid_inputs_ValueError() -> None:
    with pytest.raises(ValueError, match="both classes"):
        pr_curve_figure([1, 1], [0.2, 0.9])
    with pytest.raises(ValueError, match="threshold"):
        confusion_matrix_figure([0, 1], [0.2, 0.9], threshold=1.5)

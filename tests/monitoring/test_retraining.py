"""Tests for the retraining rule at the boundaries of both conditions."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

import campaign_response
from campaign_response.monitoring.drift import DEFAULT_THRESHOLD
from campaign_response.monitoring.retraining import MonitoringParams, recommend_retraining

PARAMS = MonitoringParams()


@pytest.mark.parametrize(
    ("n_current", "n_drifted", "drift", "retrain"),
    [
        (200, 3, True, True),
        (200, 2, False, False),
        (199, 3, True, False),
        (199, 2, False, False),
        (5000, 25, True, True),
        (0, 0, False, False),
    ],
)
def test_retraining_boundaries(n_current: int, n_drifted: int, drift: bool, retrain: bool) -> None:
    advice = recommend_retraining(n_current, n_drifted, PARAMS)

    assert (advice.dataset_drift, advice.retrain_recommended) == (drift, retrain)


def test_retraining_reason_columns() -> None:
    advice = recommend_retraining(400, 4, PARAMS)

    assert advice.reason == "4 of 25 columns drifted (alarm at 3)"


def test_retraining_small_window_reason() -> None:
    advice = recommend_retraining(150, 7, PARAMS)

    assert advice.reason.startswith("only 150 predictions in the window, at least 200 needed")
    assert "7 of 25 columns drifted" in advice.reason


def test_retraining_custom_params_used() -> None:
    params = MonitoringParams(min_rows=50, min_drifted_columns=1)

    assert recommend_retraining(50, 1, params).retrain_recommended


def test_monitoring_threshold_matches_drift_default() -> None:
    assert PARAMS.drift_threshold == DEFAULT_THRESHOLD


def test_retraining_import_without_evidently() -> None:
    """params.py loads this module in every pipeline stage; Evidently's import alone takes about 3 s."""
    code = "import sys, campaign_response.params; print('evidently' in sys.modules)"
    env = {**os.environ, "PYTHONPATH": str(Path(campaign_response.__file__).parents[1])}

    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, env=env)

    assert result.stdout.strip() == "False"

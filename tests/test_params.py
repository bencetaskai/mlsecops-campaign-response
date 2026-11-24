"""Tests for loading and validating params.yaml."""

from datetime import date
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from campaign_response.data.cleaning import CleaningParams
from campaign_response.data.splitting import SplitParams
from campaign_response.params import load_params
from campaign_response.registry.promotion import GateParams

REPO_PARAMS = Path(__file__).resolve().parents[1] / "params.yaml"
VALID = """
cleaning:
  reference_date: 2015-06-30
  min_age: 21
  max_age: 90
  max_income: 150000
split:
  holdout_size: 0.2
  seed: 7
gate:
  min_pr_auc: 0.25
  prevalence_margin: 0.05
  max_roc_auc_drop: 0.03
  min_recall: 0.2
  min_prediction_rate: 0.01
  max_prediction_rate: 0.5
  bootstrap_resamples: 500
  bootstrap_quantile: 0.05
  max_pr_auc_loss: 0.01
  seed: 3
"""


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "params.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_repo_params_match_code_defaults() -> None:
    params = load_params(REPO_PARAMS)

    assert params.cleaning == CleaningParams()
    assert params.split == SplitParams()
    assert params.gate == GateParams()


def test_params_values_loaded(tmp_path: Path) -> None:
    params = load_params(write(tmp_path, VALID))

    assert params.cleaning == CleaningParams(date(2015, 6, 30), 21, 90, 150000)
    assert params.split == SplitParams(holdout_size=0.2, seed=7)
    assert params.gate == GateParams(0.25, 0.05, 0.03, 0.2, 0.01, 0.5, 500, 0.05, 0.01, 3)


def test_params_quoted_date_accepted(tmp_path: Path) -> None:
    params = load_params(write(tmp_path, VALID.replace("2015-06-30", '"2015-06-30"')))

    assert params.cleaning.reference_date == date(2015, 6, 30)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("max_income: 150000", "max_incom: 150000", "Extra inputs are not permitted"),
        ("min_age: 21", "min_age: 95", "must be below max_age"),
        ("min_age: 21", "min_age: -1", "greater than or equal to 0"),
        ("holdout_size: 0.2", "holdout_size: 1.0", "less than 1"),
        ("seed: 7", "seed: lucky", "valid integer"),
        ("min_prediction_rate: 0.01", "min_prediction_rate: 0.7", "must be below max_prediction_rate"),
        ("bootstrap_resamples: 500", "bootstrap_resamples: 10", "greater than or equal to 100"),
        ("bootstrap_quantile: 0.05", "bootstrap_quantile: 0.5", "less than 0.5"),
    ],
)
def test_params_invalid_ValidationError(tmp_path: Path, old: str, new: str, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        load_params(write(tmp_path, VALID.replace(old, new)))


def test_params_missing_section_ValidationError(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="gate"):
        load_params(write(tmp_path, VALID.split("gate:")[0]))


def test_params_python_object_tag_rejected(tmp_path: Path) -> None:
    unsafe = "cleaning: !!python/object/apply:os.system ['echo pwned']\n"

    with pytest.raises(yaml.constructor.ConstructorError):
        load_params(write(tmp_path, unsafe))

"""Tests for loading and validating params.yaml."""

from datetime import date
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from campaign_response.data.cleaning import CleaningParams
from campaign_response.data.splitting import SplitParams
from campaign_response.params import load_params

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
"""


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "params.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_repo_params_match_code_defaults() -> None:
    params = load_params(REPO_PARAMS)

    assert params.cleaning == CleaningParams()
    assert params.split == SplitParams()


def test_params_values_loaded(tmp_path: Path) -> None:
    params = load_params(write(tmp_path, VALID))

    assert params.cleaning == CleaningParams(date(2015, 6, 30), 21, 90, 150000)
    assert params.split == SplitParams(holdout_size=0.2, seed=7)


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
    ],
)
def test_params_invalid_ValidationError(tmp_path: Path, old: str, new: str, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        load_params(write(tmp_path, VALID.replace(old, new)))


def test_params_missing_section_ValidationError(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="split"):
        load_params(write(tmp_path, VALID.split("split:")[0]))


def test_params_python_object_tag_rejected(tmp_path: Path) -> None:
    unsafe = "cleaning: !!python/object/apply:os.system ['echo pwned']\n"

    with pytest.raises(yaml.constructor.ConstructorError):
        load_params(write(tmp_path, unsafe))

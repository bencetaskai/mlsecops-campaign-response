"""Tests for the lineage tags of a training run."""

import hashlib
import platform
from datetime import date
from pathlib import Path

import pytest

from campaign_response.registry.lineage import UNKNOWN, dvc_md5, git_sha, lineage_tags, params_sha256

POINTER = "outs:\n- md5: 3b624fc2bdbe29fa5fb01bdd577f78cd\n  size: 5\n  hash: md5\n  path: raw.csv\n"


@pytest.fixture
def data(tmp_path: Path) -> Path:
    path = tmp_path / "raw.csv"
    path.write_bytes(b"a\tb\n")
    (tmp_path / "raw.csv.dvc").write_text(POINTER, encoding="utf-8")
    (tmp_path / "requirements.txt").write_bytes(b"pandas==3.1.1\n")
    return path


def tags(data: Path, **overrides: object) -> dict[str, str]:
    arguments: dict[str, object] = {
        "data_path": data, "params": {"seed": 42}, "trained_by": "api",
        "requirements_path": data.with_name("requirements.txt"), "environ": {"GIT_SHA": "abc123"},
    }
    arguments.update(overrides)
    return lineage_tags(**arguments)


def test_git_sha_from_environment() -> None:
    assert git_sha({"GIT_SHA": "4b2810f"}) == "4b2810f"


@pytest.mark.parametrize("environ", [{}, {"GIT_SHA": ""}])
def test_git_sha_missing_unknown(environ: dict[str, str]) -> None:
    assert git_sha(environ) == UNKNOWN


def test_dvc_md5_read_from_pointer(data: Path) -> None:
    assert dvc_md5(data.with_name("raw.csv.dvc")) == "3b624fc2bdbe29fa5fb01bdd577f78cd"


@pytest.mark.parametrize("content", [None, "outs: []\n", "something: else\n", "just text\n"])
def test_dvc_md5_missing_or_odd_pointer_unknown(tmp_path: Path, content: str | None) -> None:
    pointer = tmp_path / "raw.csv.dvc"
    if content is not None:
        pointer.write_text(content, encoding="utf-8")

    assert dvc_md5(pointer) == UNKNOWN


def test_params_hash_ignores_key_order() -> None:
    assert params_sha256({"a": 1, "b": {"c": 2, "d": 3}}) == params_sha256({"b": {"d": 3, "c": 2}, "a": 1})


def test_params_hash_changes_with_value() -> None:
    assert params_sha256({"seed": 42}) != params_sha256({"seed": 43})


def test_params_hash_handles_dates() -> None:
    assert len(params_sha256({"reference_date": date(2014, 12, 31)})) == 64


def test_lineage_tags_all_values(data: Path) -> None:
    result = tags(data)

    assert result == {
        "git_sha": "abc123",
        "data_sha256": hashlib.sha256(b"a\tb\n").hexdigest(),
        "dvc_md5": "3b624fc2bdbe29fa5fb01bdd577f78cd",
        "params_sha256": params_sha256({"seed": 42}),
        "python_version": platform.python_version(),
        "requirements_sha256": hashlib.sha256(b"pandas==3.1.1\n").hexdigest(),
        "trained_by": "api",
    }


def test_lineage_tags_data_hash_follows_file(data: Path) -> None:
    before = tags(data)["data_sha256"]
    data.write_bytes(b"a\tb\n1\t2\n")

    assert tags(data)["data_sha256"] != before


def test_lineage_tags_missing_requirements_unknown(data: Path) -> None:
    assert tags(data, requirements_path=data.with_name("absent.txt"))["requirements_sha256"] == UNKNOWN


def test_lineage_tags_unknown_caller_ValueError(data: Path) -> None:
    with pytest.raises(ValueError, match="trained_by must be one of"):
        tags(data, trained_by="cron")

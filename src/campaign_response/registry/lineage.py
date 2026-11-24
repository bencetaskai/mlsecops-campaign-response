"""Lineage tags for a training run: enough to trace any model back to its exact code, data, parameters and caller.

The chain a prediction can be followed along: prediction -> model version -> run -> these tags.
"""

import hashlib
import json
import os
import platform
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, get_args

import yaml

from campaign_response.data.io import file_sha256

TrainedBy = Literal["api", "airflow", "bootstrap", "cli"]
UNKNOWN = "unknown"


def git_sha(environ: Mapping[str, str] = os.environ) -> str:
    """The commit the code was built from; the image gets it as the GIT_SHA build argument.

    Read from the environment, not from git: the container has neither git nor the .git folder.
    """
    return environ.get("GIT_SHA") or UNKNOWN


def dvc_md5(pointer: Path) -> str:
    """The md5 that the .dvc pointer file records for the data, i.e. the data version in DVC's own terms."""
    if not pointer.is_file():
        return UNKNOWN
    content = yaml.safe_load(pointer.read_text(encoding="utf-8"))
    try:
        return str(content["outs"][0]["md5"])
    except (KeyError, IndexError, TypeError):
        return UNKNOWN


def params_sha256(params: Mapping[str, Any]) -> str:
    """Hash of the parameters as canonical JSON, so equal settings give equal hashes regardless of key order."""
    canonical = json.dumps(params, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


def lineage_tags(
    *,
    data_path: Path,
    params: Mapping[str, Any],
    trained_by: TrainedBy,
    requirements_path: Path,
    environ: Mapping[str, str] = os.environ,
) -> dict[str, str]:
    """All lineage tags of a run. data_sha256 is computed from the file actually read, not taken from DVC metadata."""
    if trained_by not in get_args(TrainedBy):
        raise ValueError(f"trained_by must be one of {get_args(TrainedBy)}, got {trained_by!r}")
    return {
        "git_sha": git_sha(environ),
        "data_sha256": file_sha256(data_path),
        "dvc_md5": dvc_md5(data_path.with_name(data_path.name + ".dvc")),
        "params_sha256": params_sha256(params),
        "python_version": platform.python_version(),
        "requirements_sha256": file_sha256(requirements_path) if requirements_path.is_file() else UNKNOWN,
        "trained_by": trained_by,
    }

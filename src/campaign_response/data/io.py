"""Reading the raw Kaggle file and fingerprinting data files for lineage."""

import hashlib
import logging
from collections.abc import Hashable
from pathlib import Path

import pandas as pd

from campaign_response.data.schema import RAW_DTYPES

logger = logging.getLogger(__name__)

_HASH_CHUNK_BYTES = 1024 * 1024


class RawDataError(ValueError):
    """The raw file exists but cannot be parsed with the schema's dtypes."""


def load_raw(path: Path) -> pd.DataFrame:
    """Load the tab-separated raw file with the schema dtypes; the file itself is never modified."""
    if not path.is_file():
        raise FileNotFoundError(f"raw data file not found: {path} (run `dvc pull` to fetch it)")
    # pandas types dtype as Mapping[Hashable, ...]; Mapping keys are invariant, so dict[str, str] is rejected.
    dtypes: dict[Hashable, str] = {column: dtype for column, dtype in RAW_DTYPES.items()}
    try:
        frame = pd.read_csv(path, sep="\t", dtype=dtypes, encoding="utf-8")
    except ValueError as error:
        raise RawDataError(f"cannot parse {path} with the raw schema: {error}") from error
    logger.info("loaded raw data", extra={"path": str(path), "rows": len(frame), "columns": frame.shape[1]})
    return frame


def file_sha256(path: Path) -> str:
    """Return the SHA-256 hex digest of a file, read in chunks so large files need little memory."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(_HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()

"""Tests for the raw loader and the file fingerprint."""

import hashlib
import logging
from pathlib import Path

import pandas as pd
import pytest

from campaign_response.data import io
from campaign_response.data.io import RawDataError, file_sha256, load_raw
from campaign_response.data.schema import RAW_COLUMNS, RAW_DTYPES

VALID_ROW = [
    "5524", "1957", "Graduation", "Single", "58138", "0", "0", "04-09-2012", "58",
    "635", "88", "546", "172", "88", "88", "3", "8", "10", "4", "7",
    "0", "0", "0", "0", "0", "0", "3", "11", "1",
]


def write_tsv(path: Path, rows: list[list[str]]) -> Path:
    """Write the header and rows the way the Kaggle file is stored: tab-separated, CRLF line endings."""
    lines = ["\t".join(RAW_COLUMNS), *("\t".join(row) for row in rows)]
    path.write_bytes(("\r\n".join(lines) + "\r\n").encode("utf-8"))
    return path


@pytest.fixture
def raw_file(tmp_path: Path) -> Path:
    no_income = VALID_ROW.copy()
    no_income[0], no_income[4] = "7", ""
    return write_tsv(tmp_path / "marketing_campaign.csv", [VALID_ROW, no_income])


def test_load_raw_crlf_tsv_rows(raw_file: Path) -> None:
    frame = load_raw(raw_file)

    assert frame.shape == (2, len(RAW_COLUMNS))
    assert tuple(frame.columns) == RAW_COLUMNS
    assert frame.loc[0, "Response"] == 1


def test_load_raw_schema_dtypes(raw_file: Path) -> None:
    frame = load_raw(raw_file)

    assert {column: str(dtype) for column, dtype in frame.dtypes.items()} == dict(RAW_DTYPES)


def test_load_raw_missing_income_na(raw_file: Path) -> None:
    frame = load_raw(raw_file)

    assert frame.loc[0, "Income"] == 58138
    assert pd.isna(frame.loc[1, "Income"])


def test_load_raw_logs_row_count(raw_file: Path, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="campaign_response.data.io"):
        load_raw(raw_file)

    assert caplog.records[-1].__dict__["rows"] == 2


def test_load_raw_missing_file_FileNotFoundError(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="dvc pull"):
        load_raw(tmp_path / "absent.csv")


def test_load_raw_bad_number_RawDataError(tmp_path: Path) -> None:
    broken = VALID_ROW.copy()
    broken[8] = "fifty"
    path = write_tsv(tmp_path / "broken.csv", [broken])

    with pytest.raises(RawDataError, match="cannot parse"):
        load_raw(path)


def test_file_sha256_known_digest(tmp_path: Path) -> None:
    path = tmp_path / "abc.txt"
    path.write_bytes(b"abc")

    assert file_sha256(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_file_sha256_many_chunks_same_digest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    content = bytes(range(256)) * 40
    path = tmp_path / "data.bin"
    path.write_bytes(content)
    monkeypatch.setattr(io, "_HASH_CHUNK_BYTES", 1000)

    assert file_sha256(path) == hashlib.sha256(content).hexdigest()

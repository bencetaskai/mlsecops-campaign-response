"""Tests for cutting data versions by enrollment date."""

from datetime import date
from pathlib import Path

import pytest

from campaign_response.data.versions import write_enrolled_before

HEADER = b"ID\tIncome\tDt_Customer\tResponse\r\n"
ROWS = [
    b"1\t58138\t04-09-2012\t1\r\n",
    b"2\t\t31-12-2013\t0\r\n",
    b"3\t71613\t01-01-2014\t0\r\n",
    b"4\t26646\t10-02-2014\t0\r\n",
]


@pytest.fixture
def full_file(tmp_path: Path) -> Path:
    path = tmp_path / "full.csv"
    path.write_bytes(HEADER + b"".join(ROWS))
    return path


def test_write_before_cutoff_keeps_earlier_rows(full_file: Path, tmp_path: Path) -> None:
    target = tmp_path / "out" / "v1.csv"

    count = write_enrolled_before(full_file, target, date(2014, 1, 1))

    assert count == 2
    assert target.read_bytes() == HEADER + ROWS[0] + ROWS[1]


def test_write_cutoff_after_all_rows_identical(full_file: Path, tmp_path: Path) -> None:
    target = tmp_path / "all.csv"

    write_enrolled_before(full_file, target, date(2030, 1, 1))

    assert target.read_bytes() == full_file.read_bytes()


def test_write_same_file_ValueError(full_file: Path) -> None:
    with pytest.raises(ValueError, match="never be overwritten"):
        write_enrolled_before(full_file, full_file, date(2014, 1, 1))


def test_write_no_date_column_ValueError(tmp_path: Path) -> None:
    source = tmp_path / "no_date.csv"
    source.write_bytes(b"ID\tIncome\r\n1\t5\r\n")

    with pytest.raises(ValueError, match="no Dt_Customer column"):
        write_enrolled_before(source, tmp_path / "out.csv", date(2014, 1, 1))


def test_write_bad_date_ValueError(tmp_path: Path) -> None:
    source = tmp_path / "bad_date.csv"
    source.write_bytes(HEADER + b"1\t5\t2012-09-04\t0\r\n")

    with pytest.raises(ValueError, match="does not match format"):
        write_enrolled_before(source, tmp_path / "out.csv", date(2014, 1, 1))

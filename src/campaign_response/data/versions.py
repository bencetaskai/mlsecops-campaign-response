"""Cut earlier data versions out of the full raw file to simulate how the dataset grew over time."""

from datetime import date, datetime
from pathlib import Path

from campaign_response.data.schema import DATE_COLUMN, DATE_FORMAT


def write_enrolled_before(source: Path, target: Path, cutoff: date) -> int:
    """Write the header and every row enrolled before ``cutoff`` from ``source`` to ``target``; return the row count.

    Lines are copied byte for byte (no pandas round trip), so separators, CRLF line endings and number formats stay
    exactly as in the source and the version differs from the full file only by the rows it leaves out.
    """
    if source.resolve() == target.resolve():
        raise ValueError("source and target are the same file; the raw file must never be overwritten")
    header, *rows = source.read_bytes().splitlines(keepends=True)
    columns = header.decode("utf-8").rstrip("\r\n").split("\t")
    if DATE_COLUMN not in columns:
        raise ValueError(f"{source} has no {DATE_COLUMN} column")
    date_index = columns.index(DATE_COLUMN)

    kept = [
        row for row in rows
        if datetime.strptime(row.decode("utf-8").split("\t")[date_index], DATE_FORMAT).date() < cutoff
    ]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(header + b"".join(kept))
    return len(kept)

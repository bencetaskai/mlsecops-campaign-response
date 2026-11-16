"""Structured logging: one JSON object per line (or readable text), with a correlation id on every record.

The correlation id ties together all log lines of one training run or one API request. It is stored in a
``ContextVar``, so concurrent requests in the async API each see their own id.
"""

import json
import logging
import sys
import uuid
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import TextIO

from campaign_response.config import LogFormat, LogLevel

NO_CORRELATION_ID = "-"
correlation_id_var: ContextVar[str] = ContextVar("correlation_id", default=NO_CORRELATION_ID)

TEXT_FORMAT = "%(asctime)s %(levelname)-7s %(name)s [%(correlation_id)s] %(message)s"

# Attributes every LogRecord has; anything else on a record came from ``extra={...}`` and is logged as a field.
_STANDARD_RECORD_ATTRIBUTES = frozenset(vars(logging.LogRecord("", 0, "", 0, "", None, None))) | {
    "message",
    "asctime",
    "correlation_id",
}


def new_correlation_id() -> str:
    """Create a fresh correlation id, make it the current one, and return it."""
    value = uuid.uuid4().hex[:16]
    correlation_id_var.set(value)
    return value


class CorrelationIdFilter(logging.Filter):
    """Attach the current correlation id to every record passing through the handler."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.correlation_id = correlation_id_var.get()
        return True


class JsonFormatter(logging.Formatter):
    """Format a record as a single-line JSON object that log tools can filter by field."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": getattr(record, "correlation_id", NO_CORRELATION_ID),
        }
        for key, value in vars(record).items():
            if key not in _STANDARD_RECORD_ATTRIBUTES:
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: LogLevel, log_format: LogFormat, stream: TextIO | None = None) -> None:
    """Replace the root logger's handlers with one handler in the requested level and format.

    Calling it again reconfigures logging instead of adding a duplicate handler.
    """
    handler = logging.StreamHandler(stream if stream is not None else sys.stderr)
    handler.addFilter(CorrelationIdFilter())
    handler.setFormatter(JsonFormatter() if log_format == "json" else logging.Formatter(TEXT_FORMAT))

    root = logging.getLogger()
    for old_handler in root.handlers[:]:
        root.removeHandler(old_handler)
    root.addHandler(handler)
    root.setLevel(level)

"""Tests for structured logging and the correlation id."""

import io
import json
import logging
from collections.abc import Iterator

import pytest

from campaign_response.logging_setup import (
    NO_CORRELATION_ID,
    configure_logging,
    correlation_id_var,
    new_correlation_id,
)


@pytest.fixture(autouse=True)
def isolated_logging() -> Iterator[None]:
    """Restore the root logger and the correlation id after each test, so no state leaks between tests."""
    root = logging.getLogger()
    saved_handlers, saved_level = root.handlers[:], root.level
    token = correlation_id_var.set(NO_CORRELATION_ID)
    yield
    correlation_id_var.reset(token)
    root.handlers[:] = saved_handlers
    root.setLevel(saved_level)


def test_new_correlation_id_sets_current_value() -> None:
    value = new_correlation_id()

    assert len(value) == 16
    assert correlation_id_var.get() == value


def test_new_correlation_id_unique() -> None:
    assert new_correlation_id() != new_correlation_id()


def test_json_log_line_fields() -> None:
    stream = io.StringIO()
    configure_logging("INFO", "json", stream)
    correlation_id = new_correlation_id()

    logging.getLogger("campaign_response.demo").info("test metrics", extra={"f1": 0.8, "rows": 3})

    record = json.loads(stream.getvalue())
    assert record["level"] == "INFO"
    assert record["logger"] == "campaign_response.demo"
    assert record["message"] == "test metrics"
    assert record["correlation_id"] == correlation_id
    assert record["f1"] == 0.8
    assert record["rows"] == 3
    assert record["ts"].endswith("+00:00")


def test_json_log_without_correlation_id_dash() -> None:
    stream = io.StringIO()
    configure_logging("INFO", "json", stream)

    logging.getLogger("x").warning("no run active")

    assert json.loads(stream.getvalue())["correlation_id"] == "-"


def test_json_log_exception_included() -> None:
    stream = io.StringIO()
    configure_logging("ERROR", "json", stream)

    try:
        raise ValueError("boom")
    except ValueError:
        logging.getLogger("x").exception("training failed")

    record = json.loads(stream.getvalue())
    assert "ValueError: boom" in record["exception"]


def test_text_log_line_format() -> None:
    stream = io.StringIO()
    configure_logging("INFO", "text", stream)
    correlation_id = new_correlation_id()

    logging.getLogger("campaign_response.demo").info("hello")

    line = stream.getvalue()
    assert f"INFO    campaign_response.demo [{correlation_id}] hello" in line


def test_level_filters_lower_records() -> None:
    stream = io.StringIO()
    configure_logging("WARNING", "text", stream)

    logging.getLogger("x").info("hidden")

    assert stream.getvalue() == ""


def test_configure_twice_one_handler() -> None:
    configure_logging("INFO", "text", io.StringIO())
    configure_logging("DEBUG", "json", io.StringIO())

    root = logging.getLogger()
    assert len(root.handlers) == 1
    assert root.level == logging.DEBUG


def test_configure_default_stream_stderr(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("INFO", "text")

    logging.getLogger("x").info("to stderr")

    assert "to stderr" in capsys.readouterr().err

"""Tests for the package root."""

import campaign_response


def test_package_exposes_version() -> None:
    assert campaign_response.__version__ == "0.1.0"

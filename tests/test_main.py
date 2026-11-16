"""Tests for the ``python -m campaign_response`` entry point."""

import runpy
import sys

import pytest

from campaign_response.__main__ import main


def test_main_prints_version_exit_0(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main()

    assert exit_code == 0
    assert capsys.readouterr().out == "campaign-response 0.1.0\n"


def test_python_m_exit_0(capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    # The import at the top of this file already loaded campaign_response.__main__. runpy would warn that it
    # executes a module that is already imported, so remove it first; monkeypatch restores it after the test.
    monkeypatch.delitem(sys.modules, "campaign_response.__main__")

    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module("campaign_response", run_name="__main__")

    assert exit_info.value.code == 0
    assert capsys.readouterr().out == "campaign-response 0.1.0\n"

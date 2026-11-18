"""Tests for the ``python -m campaign_response`` entry point."""

import importlib
import runpy
import sys

import pytest


def test_import_runs_nothing(capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, "campaign_response.__main__", raising=False)

    importlib.import_module("campaign_response.__main__")

    assert capsys.readouterr().out == ""


def test_python_m_exit_0(capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    # runpy warns if the module it executes is already imported; monkeypatch restores sys.modules and sys.argv.
    monkeypatch.delitem(sys.modules, "campaign_response.__main__", raising=False)
    monkeypatch.setattr(sys, "argv", ["campaign_response"])

    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module("campaign_response", run_name="__main__")

    assert exit_info.value.code == 0
    assert capsys.readouterr().out == "campaign-response 0.1.0\n"

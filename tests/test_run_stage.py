"""Tests for the DVC stage launcher in tools/."""

import runpy
import sys
from pathlib import Path

import pytest

LAUNCHER = Path(__file__).resolve().parents[1] / "tools" / "run_stage.py"


@pytest.fixture(autouse=True)
def restore_sys_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "path", sys.path.copy())


def test_launcher_runs_cli_exit_0(capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["run_stage.py"])

    with pytest.raises(SystemExit) as exit_info:
        runpy.run_path(str(LAUNCHER), run_name="__main__")

    assert exit_info.value.code == 0
    assert capsys.readouterr().out == "campaign-response 0.1.0\n"


def test_launcher_puts_src_first() -> None:
    namespace = runpy.run_path(str(LAUNCHER))

    assert sys.path[0] == str(LAUNCHER.parents[1] / "src")
    assert callable(namespace["main"])

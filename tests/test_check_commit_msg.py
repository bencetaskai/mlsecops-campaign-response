"""Tests for the commit message format checker (tools/check_commit_msg.py)."""

import runpy
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest

from check_commit_msg import check_message, clean_message, commits_in_range, main, run_git

SIGNED_OFF = "Signed-off-by: Bence Táskai <113670719+bencetaskai@users.noreply.github.com>"
VALID = f"""ci: Add commit message checker for the lecture format

Explain what changed and why.

References: MLS-010

{SIGNED_OFF}
"""
TOOL_PATH = Path(__file__).resolve().parents[1] / "tools" / "check_commit_msg.py"


def replace(old: str, new: str) -> str:
    """Return VALID with one part replaced, so each test breaks exactly one rule."""
    assert old in VALID
    return VALID.replace(old, new)


def test_check_valid_message_no_errors() -> None:
    assert check_message(VALID) == []


def test_check_merge_message_no_errors() -> None:
    message = replace("ci: Add commit message checker for the lecture format",
                      "merge: Project skeleton with quality gates and CI (#1)")

    assert check_message(message) == []


def test_check_several_references_no_errors() -> None:
    assert check_message(replace("References: MLS-010", "References: MLS-010, MLS-011")) == []


def test_clean_drops_comments_blanks_diff() -> None:
    comments = "# Please enter the commit message\n# ------------------------ >8 ------------------------\n"
    raw = VALID + "\n   \n" + comments + "diff\n"

    assert clean_message(raw) == VALID.rstrip("\n").split("\n")


@pytest.mark.parametrize(
    ("message", "expected_error"),
    [
        ("", "empty"),
        ("# only a comment\n", "empty"),
        (replace("ci: ", "feat: "), "unknown keyword 'feat'"),
        (replace("ci: ", "ci "), "must look like 'keyword: Summary'"),
        (replace("ci: Add", "ci: add"), "capital letter"),
        (replace("lecture format\n", "lecture format.\n"), "must not end with a period"),
        (replace("lecture format\n", "lecture format" + " and more" * 4 + "\n"), "(max 80)"),
        (replace("format\n\nExplain", "format\nExplain"), "missing parts"),
        (replace("format\n\nExplain", "format\nX\nExplain"), "line 2 must be empty"),
        (replace("Explain what changed and why.", ""), "body is missing"),
        (replace("why.\n\nReferences", "why.\nX\nReferences"), "empty line is required before 'References:'"),
        (replace("References: MLS-010", "References: #10"), "'References: MLS-<n>"),
        (replace("MLS-010\n\nSigned", "MLS-010\nX\nSigned"), "between 'References:' and 'Signed-off-by:'"),
        (replace(SIGNED_OFF, "Signed-off-by: Bence Táskai"), "the last line must be 'Signed-off-by"),
    ],
)
def test_check_broken_rule_reported(message: str, expected_error: str) -> None:
    errors = check_message(message)

    assert any(expected_error in error for error in errors), errors


def test_commits_in_range_parses_git_log() -> None:
    calls: list[Sequence[str]] = []

    def fake_git(args: Sequence[str]) -> str:
        calls.append(args)
        return f"abc1234\x1f{VALID}\x1e\ndef5678\x1fwip\n\x1e\n"

    commits = commits_in_range("origin/main..HEAD", fake_git)

    assert commits == [("abc1234", VALID), ("def5678", "wip\n")]
    assert calls[0][-1] == "origin/main..HEAD"


def test_commits_in_range_option_ValueError() -> None:
    with pytest.raises(ValueError, match="invalid revision range"):
        commits_in_range("--output=/tmp/x", lambda args: "")


def test_main_valid_bom_file_exit_0(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    message_file = tmp_path / "COMMIT_EDITMSG"
    message_file.write_text(VALID, encoding="utf-8-sig")  # with a BOM, as PowerShell 5.1 writes it

    assert main([str(message_file)]) == 0
    assert "1 commit message(s) OK" in capsys.readouterr().out


def test_main_range_invalid_exit_1(capsys: pytest.CaptureFixture[str]) -> None:
    def fake_git(args: Sequence[str]) -> str:
        return f"abc1234\x1f{VALID}\x1edef5678\x1fwip\n\x1e"

    assert main(["--range", "origin/main..HEAD"], fake_git) == 1
    err = capsys.readouterr().err
    assert "def5678: invalid commit message" in err
    assert "abc1234" not in err
    assert "1 of 2 commit message(s)" in err


def test_main_without_target_exit_2(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main([])

    assert exit_info.value.code == 2
    assert "required" in capsys.readouterr().err


def test_run_git_returns_stdout() -> None:
    assert run_git(["--version"]).startswith("git version")


def test_script_run_as_main_exit_0(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    message_file = tmp_path / "COMMIT_EDITMSG"
    message_file.write_text(VALID, encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["check_commit_msg.py", str(message_file)])

    with pytest.raises(SystemExit) as exit_info:
        runpy.run_path(str(TOOL_PATH), run_name="__main__")

    assert exit_info.value.code == 0

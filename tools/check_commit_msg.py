"""Check commit messages against the lecture format (version_control.pdf p.9-10).

    keyword: Summary in at most 80 characters
    <empty line>
    Body: what changed and why (at least one line)
    <empty line>
    References: MLS-010[, MLS-011]
    <empty line>
    Signed-off-by: Name <email@address>

Two modes:

* ``check_commit_msg.py <file>`` - the pre-commit ``commit-msg`` hook passes the file with the new message.
* ``check_commit_msg.py --range origin/main..HEAD`` - CI checks every commit of a pull request.

Exit code 0 = all messages valid, 1 = at least one violation (printed to stderr).
"""

import argparse
import re
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

ALLOWED_KEYWORDS = frozenset({
    "chore", "ci", "build", "data", "preprocess", "model", "pipeline", "registry",
    "api", "storage", "monitoring", "airflow", "security", "test", "fix", "docs", "merge",
})
MAX_SUBJECT_LENGTH = 80

SUBJECT_RE = re.compile(r"^(?P<keyword>[a-z]+): (?P<summary>\S.*)$")
REFERENCES_RE = re.compile(r"^References: [A-Z][A-Z0-9]*-\d+(, [A-Z][A-Z0-9]*-\d+)*$")
SIGNED_OFF_RE = re.compile(r"^Signed-off-by: [^<>]+ <[^<>\s@]+@[^<>\s@]+>$")

# git puts this line before the diff in `git commit -v`; everything below it is not part of the message.
SCISSORS_LINE = "# ------------------------ >8 ------------------------"

# Runs ``git <args>`` and returns its stdout; injected so that tests need no real repository.
GitRunner = Callable[[Sequence[str]], str]


def clean_message(raw: str) -> list[str]:
    """Return the message lines the way git stores them: no comments, no trailing whitespace or blank lines."""
    lines: list[str] = []
    for line in raw.splitlines():
        if line == SCISSORS_LINE:
            break
        if not line.startswith("#"):
            lines.append(line.rstrip())
    while lines and not lines[-1]:
        lines.pop()
    return lines


def check_subject(subject: str) -> list[str]:
    """Validate line 1: ``keyword: Summary`` with an allowed keyword, a capital letter and no final period."""
    errors: list[str] = []
    if len(subject) > MAX_SUBJECT_LENGTH:
        errors.append(f"line 1 is {len(subject)} characters long (max {MAX_SUBJECT_LENGTH})")
    match = SUBJECT_RE.match(subject)
    if match is None:
        errors.append("line 1 must look like 'keyword: Summary'")
        return errors
    if match["keyword"] not in ALLOWED_KEYWORDS:
        errors.append(f"unknown keyword '{match['keyword']}' (allowed: {', '.join(sorted(ALLOWED_KEYWORDS))})")
    summary = match["summary"]
    if not summary[0].isupper():
        errors.append("the summary must start with a capital letter")
    if summary.endswith("."):
        errors.append("the summary must not end with a period")
    return errors


def check_message(raw: str) -> list[str]:
    """Return every format violation of one commit message; an empty list means the message is valid."""
    lines = clean_message(raw)
    if not lines:
        return ["the commit message is empty"]

    errors = check_subject(lines[0])
    # Expected tail: ..., "", "References: ...", "", "Signed-off-by: ..."
    if len(lines) < 7:
        errors.append("missing parts: need subject, empty line, body, References and Signed-off-by blocks")
        return errors
    if lines[1]:
        errors.append("line 2 must be empty")
    if not any(lines[2:-4]):
        errors.append("the body is missing: explain what changed and why")
    if lines[-4]:
        errors.append("an empty line is required before 'References:'")
    if not REFERENCES_RE.match(lines[-3]):
        errors.append("the third line from the end must be 'References: MLS-<n>[, MLS-<m>]'")
    if lines[-2]:
        errors.append("an empty line is required between 'References:' and 'Signed-off-by:'")
    if not SIGNED_OFF_RE.match(lines[-1]):
        errors.append("the last line must be 'Signed-off-by: Name <email@address>'")
    return errors


def run_git(args: Sequence[str]) -> str:
    """Run git without a shell and return stdout decoded as UTF-8 (names like 'Táskai' survive on Windows)."""
    completed = subprocess.run(["git", *args], capture_output=True, check=True, encoding="utf-8")
    return completed.stdout


def commits_in_range(rev_range: str, git: GitRunner = run_git) -> list[tuple[str, str]]:
    """Return ``(short sha, raw message)`` for each commit in ``rev_range``, oldest first."""
    if rev_range.startswith("-"):
        raise ValueError(f"invalid revision range: {rev_range!r}")
    # %x1f / %x1e are the ASCII unit / record separators: they never appear in a commit message.
    output = git(["log", "--reverse", "--format=%h%x1f%B%x1e", rev_range])
    commits: list[tuple[str, str]] = []
    for record in output.split("\x1e"):
        record = record.lstrip("\n")  # git log ends every record with a newline, which lands before the next one
        if record:
            sha, message = record.split("\x1f", 1)
            commits.append((sha, message))
    return commits


def main(argv: Sequence[str] | None = None, git: GitRunner = run_git) -> int:
    parser = argparse.ArgumentParser(description="Check commit messages against the lecture format.")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("message_file", nargs="?", type=Path, help="file with the commit message (commit-msg hook)")
    target.add_argument("--range", dest="rev_range", help="git revision range to check, e.g. origin/main..HEAD")
    args = parser.parse_args(argv)

    if args.rev_range is not None:
        messages = commits_in_range(args.rev_range, git)
    else:
        # utf-8-sig drops a BOM, which Windows editors and PowerShell 5.1 like to write at the start of a file.
        messages = [("new commit", args.message_file.read_text(encoding="utf-8-sig"))]

    failed = 0
    for label, message in messages:
        errors = check_message(message)
        if errors:
            failed += 1
            print(f"{label}: invalid commit message", file=sys.stderr)
            for error in errors:
                print(f"  - {error}", file=sys.stderr)
    if failed:
        print(f"{failed} of {len(messages)} commit message(s) violate the format (version_control.pdf p.9)",
              file=sys.stderr)
        return 1
    print(f"{len(messages)} commit message(s) OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())

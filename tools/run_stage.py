"""Run a pipeline command from a checkout; dvc.yaml calls this, e.g. ``python tools/run_stage.py prepare``.

It puts src/ on sys.path itself: the package is not installed (an editable install's .pth file breaks on the
accented project path on Windows), and DVC runs stage commands in the platform shell, where setting PYTHONPATH
is written differently in cmd and bash.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from campaign_response.cli import main  # noqa: E402 - must come after the sys.path change above

if __name__ == "__main__":
    sys.exit(main())

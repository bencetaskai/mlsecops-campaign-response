"""Entry point for ``python -m campaign_response``: prints the package name and version."""

import sys

from campaign_response import __version__


def main() -> int:
    """Print the package version and return the process exit code (0 = success)."""
    print(f"campaign-response {__version__}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

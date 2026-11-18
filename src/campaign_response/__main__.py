"""Entry point for ``python -m campaign_response``; the commands live in campaign_response.cli."""

import sys

from campaign_response.cli import main

if __name__ == "__main__":
    sys.exit(main())

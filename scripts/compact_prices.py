"""One-off maintenance: collapse duplicate price/volume bars in ziidi_prices.db."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.data import compact_price_history


def main() -> None:
    parser = argparse.ArgumentParser(description="Compact duplicate rows in ziidi_prices.db")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report how many rows would be removed without writing",
    )
    args = parser.parse_args()
    summary = compact_price_history(dry_run=args.dry_run)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

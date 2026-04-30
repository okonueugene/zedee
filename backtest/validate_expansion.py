"""
backtest/validate_expansion.py

Offline validation of the watchlist expansion pipeline.
Run against today's DB state before enabling in live scanner.

Expected output:
  Universe (ADVT): 6–10 symbols
  Activated: 3–6 symbols

If activated = 0 → check ADVT threshold (MIN_ADVT_KES may be too high)
If activated > 8 → activation thresholds may be too loose
"""

import sqlite3
import sys
from pathlib import Path

# Ensure repo root is on sys.path when running as a script (Windows).
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config.params import PRICE_DB, WATCHLIST_EXTENDED
from core.activation import activation_reason, is_activated
from core.features import enrich_universe
from core.universe import build_universe


def run_validation():
    conn = sqlite3.connect(PRICE_DB, timeout=30)
    try:
        enriched = enrich_universe(WATCHLIST_EXTENDED, conn)
    finally:
        conn.close()

    print("\n" + ("=" * 60))
    print(f"WATCHLIST_EXTENDED: {len(WATCHLIST_EXTENDED)} symbols")
    print(f"Enriched (>=10 bars): {len(enriched)} symbols")

    skipped = [s for s in WATCHLIST_EXTENDED if s not in enriched]
    if skipped:
        print(f"Skipped (no history): {skipped}")

    universe = build_universe(enriched)
    print(f"Universe (ADVT filter): {len(universe)} symbols: {universe}")

    print("\n" + ("-" * 60))
    print("Per-symbol enrichment details:")
    for sym, m in enriched.items():
        advt_m = m["avg_value_20d"] / 1_000_000
        passed = sym in universe
        print(
            f"  {sym:6s}  avg_vol={m['avg_volume_20d']:>10,.0f}  "
            f"ADVT=KES {advt_m:>6.2f}M  high_5d={m['high_5d']:>8.2f}  "
            f"{'PASS' if passed else 'FAIL-ADVT'}"
        )

    print("\n" + ("-" * 60))
    print("Activation check (using last known live prices from DB):")

    activated = []
    for sym in universe:
        # Use most recent price from DB as proxy for live_row
        conn = sqlite3.connect(PRICE_DB, timeout=30)
        row = conn.execute(
            "SELECT price, volume FROM prices WHERE symbol=? ORDER BY timestamp DESC LIMIT 1",
            (sym,),
        ).fetchone()
        conn.close()

        if not row:
            print(f"  {sym:6s}  NO_LIVE_DATA")
            continue

        live_row = {"price": row[0], "volume": row[1]}
        metrics = enriched[sym]
        avg_vol = metrics["avg_volume_20d"]
        vol_ratio = row[1] / avg_vol if avg_vol else 0
        dist_high = row[0] / metrics["high_5d"] if metrics["high_5d"] else 0

        act = is_activated(live_row, metrics)
        reason = "" if act else f"-> {activation_reason(live_row, metrics)}"
        print(
            f"  {sym:6s}  vol_ratio={vol_ratio:>5.2f}x  "
            f"dist_high={dist_high:.4f}  {'ACTIVATED' if act else 'NOT_ACTIVATED'} {reason}"
        )

        if act:
            activated.append(sym)

    print("\n" + ("=" * 60))
    print(f"RESULT: {len(activated)} activated symbols: {activated}")
    print("Expected range: 3-6 activated")
    if len(activated) == 0:
        print("WARNING: 0 activated - check MIN_ADVT_KES or ACTIVATION thresholds")
    elif len(activated) > 8:
        print("WARNING: >8 activated - thresholds may be too loose")
    else:
        print("OK - proceed to wire into live scanner")


if __name__ == "__main__":
    run_validation()


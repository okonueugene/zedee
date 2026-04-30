"""
core/features.py

Feature enrichment from ziidi_prices.db.
Computes avg_volume_20d, avg_value_20d, high_5d from stored price history.
No high/low available in schema — ATR/range metrics are not computable here.
"""

from __future__ import annotations

import sqlite3

from config.params import ENRICHMENT_MIN_BARS, PRICE_DB


def get_enriched_metrics(symbol: str, conn: sqlite3.Connection) -> dict | None:
    """
    Query last 30 price rows for symbol and compute rolling metrics.

    Parameters
    ----------
    symbol : str
        NSE ticker symbol.
    conn : sqlite3.Connection
        Open connection to ziidi_prices.db. Caller owns the connection lifecycle.

    Returns
    -------
    dict with keys:
        avg_volume_20d : float   — mean volume over last 20 sessions
        avg_value_20d  : float   — mean (price × volume) over last 20 sessions (ADVT proxy)
        high_5d        : float   — max close price over last 5 sessions
    None if:
        - fewer than ENRICHMENT_MIN_BARS rows exist
        - all price or volume values are null
    """
    rows = conn.execute(
        """
        SELECT price, volume
        FROM prices
        WHERE symbol = ?
        ORDER BY timestamp DESC
        LIMIT 30
        """,
        (symbol,),
    ).fetchall()

    if len(rows) < ENRICHMENT_MIN_BARS:
        return None

    # Filter nulls defensively
    closes = [r[0] for r in rows if r[0] is not None]
    volumes = [r[1] for r in rows if r[1] is not None and r[1] > 0]

    if not closes or not volumes:
        return None

    # Use last 20 sessions (or all available if < 20)
    closes_20 = closes[:20]
    volumes_20 = volumes[:20]

    avg_volume = sum(volumes_20) / len(volumes_20)
    avg_close = sum(closes_20) / len(closes_20)
    avg_value = avg_volume * avg_close  # ADVT proxy (KES)

    # high_5d: max close over last 5 sessions (DESC order → closes[:5] = most recent)
    # Consistent with _mvp_high_5d() in main.py
    high_5d = max(closes[:5]) if len(closes) >= 5 else max(closes)

    return {
        "avg_volume_20d": avg_volume,
        "avg_value_20d": avg_value,
        "high_5d": high_5d,
    }


def enrich_universe(symbols: list[str], conn: sqlite3.Connection) -> dict[str, dict]:
    """
    Run get_enriched_metrics for every symbol in the list.
    Returns dict of symbol → metrics for symbols that pass enrichment.
    Symbols with insufficient history are silently dropped (caller logs ENRICHMENT_SKIP).
    """
    enriched = {}
    for sym in symbols:
        metrics = get_enriched_metrics(sym, conn)
        if metrics is not None:
            enriched[sym] = metrics
    return enriched


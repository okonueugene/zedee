"""
core/universe.py

Liquidity filter — ADVT-based universe construction.
Volatility filter dropped: high/low not stored in prices schema.
"""

from __future__ import annotations

from config.params import MIN_ADVT_KES


def passes_liquidity(metrics: dict) -> bool:
    """
    Return True if the symbol's average daily value traded meets the minimum.

    Uses avg_value_20d (avg_volume_20d × avg_close_20d) as ADVT proxy.
    Hard minimum: MIN_ADVT_KES (KES 3M by default for NSE reality).
    """
    return metrics.get("avg_value_20d", 0.0) >= MIN_ADVT_KES


def build_universe(enriched_data: dict[str, dict]) -> list[str]:
    """
    Filter enriched symbols by liquidity.

    Parameters
    ----------
    enriched_data : dict
        symbol → metrics dict from core/features.py::enrich_universe()

    Returns
    -------
    List of symbols that pass the liquidity filter. Order is deterministic
    (dict insertion order, Python 3.7+).
    """
    return [sym for sym, metrics in enriched_data.items() if passes_liquidity(metrics)]


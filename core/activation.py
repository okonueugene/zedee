"""
core/activation.py

Activation filter — determines which universe symbols are active enough
to run through MVP scoring this scan cycle.

A symbol is activated if:
  1. Volume today >= 1.5× its 20-session average (volume spike), OR
  2. Current price is within 3% of its 5-day high AND volume >= 80% of average

This prevents the MVP scorer from wasting cycles on:
  - Dead-volume drift days (EABL type: close=high_5d, volume=619)
  - Chronically flat names that cleared ADVT on old data
"""

from __future__ import annotations

from config.params import (
    ACTIVATION_MIN_PARTICIPATION,
    ACTIVATION_NEAR_HIGH_5D,
    ACTIVATION_VOLUME_SPIKE,
)


def is_activated(live_row: dict, metrics: dict) -> bool:
    """
    Return True if symbol passes activation criteria.

    Parameters
    ----------
    live_row : dict
        Live data for this symbol: must contain 'volume' (today's volume)
        and 'price' or 'close' (today's price).
    metrics : dict
        Enriched metrics from core/features.py: must contain
        'avg_volume_20d' and 'high_5d'.

    Returns
    -------
    bool
    """
    avg_vol = metrics.get("avg_volume_20d", 0.0)

    # Zero-guard: cannot compute ratio on zero average
    if not avg_vol or avg_vol == 0:
        return False

    today_vol = float(live_row.get("volume", 0) or 0)
    today_price = float(live_row.get("price") or live_row.get("close", 0))
    high_5d = float(metrics.get("high_5d", 0) or 0)

    volume_ratio = today_vol / avg_vol
    near_high = (high_5d > 0) and (today_price >= high_5d * ACTIVATION_NEAR_HIGH_5D)
    min_participation = volume_ratio >= ACTIVATION_MIN_PARTICIPATION

    # Condition 1: volume spike
    if volume_ratio >= ACTIVATION_VOLUME_SPIKE:
        return True

    # Condition 2: near high with minimum participation
    if near_high and min_participation:
        return True

    return False


def activation_reason(live_row: dict, metrics: dict) -> str:
    """
    Return a rejection reason string for ACTIVATION_CHECK logging.
    Only call this when is_activated() has returned False.
    """
    avg_vol = metrics.get("avg_volume_20d", 0.0)
    if not avg_vol or avg_vol == 0:
        return "ZERO_AVG_VOLUME"

    today_vol = float(live_row.get("volume", 0) or 0)
    today_price = float(live_row.get("price") or live_row.get("close", 0))
    high_5d = float(metrics.get("high_5d", 0) or 0)

    volume_ratio = today_vol / avg_vol
    near_high = (high_5d > 0) and (today_price >= high_5d * ACTIVATION_NEAR_HIGH_5D)

    if volume_ratio < ACTIVATION_MIN_PARTICIPATION:
        return "LOW_VOLUME"
    elif not near_high:
        return "BELOW_STRUCTURE"
    else:
        # Volume is 0.8–1.5× average but not spiking; not near high
        return "INSUFFICIENT_MOMENTUM"


def run_activation_check(
    symbol: str,
    live_row: dict,
    metrics: dict,
    log_fn,  # callable: log_fn("ACTIVATION_CHECK", {...})
) -> bool:
    """
    Run activation check and log rejection if not activated.

    Parameters
    ----------
    symbol   : str
    live_row : dict   — live scrape row for this symbol
    metrics  : dict   — enriched metrics from core/features.py
    log_fn   : callable — log_event from main.py

    Returns
    -------
    bool — True if activated, False otherwise
    """
    activated = is_activated(live_row, metrics)

    if not activated:
        avg_vol = metrics.get("avg_volume_20d", 0.0) or 0
        today_vol = float(live_row.get("volume", 0) or 0)
        today_price = float(live_row.get("price") or live_row.get("close", 0))
        high_5d = float(metrics.get("high_5d", 0) or 0)

        log_fn(
            "ACTIVATION_CHECK",
            {
                "symbol": symbol,
                "volume_ratio": round(today_vol / avg_vol, 2) if avg_vol else None,
                "distance_to_high_5d": round(today_price / high_5d, 4) if high_5d else None,
                "activated": False,
                "reason": activation_reason(live_row, metrics),
            },
        )

    return activated


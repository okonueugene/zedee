"""
core/activation.py

Activation filter — determines which universe symbols are active enough
to run through MVP scoring this scan cycle.

A symbol is activated if:
  1. Projected daily volume >= ACTIVATION_VOLUME_SPIKE × 20-bar average, OR
  2. Price within 3% of 5-day high AND projected volume >= ACTIVATION_MIN_PARTICIPATION × average

Intraday volume is projected to a full-session estimate before comparing to the
20-bar average (prior snapshots are typically end-of-session cumulative totals).
"""

from __future__ import annotations

from datetime import datetime

from config.params import (
    ACTIVATION_MIN_PARTICIPATION,
    ACTIVATION_NEAR_HIGH_5D,
    ACTIVATION_VOLUME_SPIKE,
)

# NSE cash session proxy for intraday volume projection (EAT)
_SESSION_OPEN_H = 10
_SESSION_OPEN_M = 0
_SESSION_CLOSE_H = 15
_SESSION_CLOSE_M = 0
_PROJECTION_CAP = 3.0


def _session_fraction(now: datetime) -> float:
    """Fraction of trading session elapsed between open and close (EAT)."""
    open_min = _SESSION_OPEN_H * 60 + _SESSION_OPEN_M
    close_min = _SESSION_CLOSE_H * 60 + _SESSION_CLOSE_M
    now_min = now.hour * 60 + now.minute
    if now_min <= open_min:
        return 0.1
    if now_min >= close_min:
        return 1.0
    return (now_min - open_min) / (close_min - open_min)


def compute_volume_ratio(today_vol: float, avg_vol: float, now: datetime) -> float:
    """
    Compare projected full-session volume to the 20-bar average.

    Caps projection multiplier at 3× to avoid absurd early-session estimates.
    """
    if not avg_vol or avg_vol == 0:
        return 0.0
    frac = _session_fraction(now)
    multiplier = min(1.0 / frac, _PROJECTION_CAP)
    projected = today_vol * multiplier
    return projected / avg_vol


def is_activated(live_row: dict, metrics: dict, now: datetime) -> bool:
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
    now : datetime
        Current scan time (EAT) for intraday volume projection.

    Returns
    -------
    bool
    """
    avg_vol = metrics.get("avg_volume_20d", 0.0)

    if not avg_vol or avg_vol == 0:
        return False

    today_vol = float(live_row.get("volume", 0) or 0)
    today_price = float(live_row.get("price") or live_row.get("close", 0))
    high_5d = float(metrics.get("high_5d", 0) or 0)

    volume_ratio = compute_volume_ratio(today_vol, avg_vol, now)
    near_high = (high_5d > 0) and (today_price >= high_5d * ACTIVATION_NEAR_HIGH_5D)
    min_participation = volume_ratio >= ACTIVATION_MIN_PARTICIPATION

    if volume_ratio >= ACTIVATION_VOLUME_SPIKE:
        return True

    if near_high and min_participation:
        return True

    return False


def activation_reason(live_row: dict, metrics: dict, now: datetime) -> str:
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

    volume_ratio = compute_volume_ratio(today_vol, avg_vol, now)
    near_high = (high_5d > 0) and (today_price >= high_5d * ACTIVATION_NEAR_HIGH_5D)

    if volume_ratio < ACTIVATION_MIN_PARTICIPATION:
        return "LOW_VOLUME"
    elif not near_high:
        return "BELOW_STRUCTURE"
    else:
        return "INSUFFICIENT_MOMENTUM"


def run_activation_check(
    symbol: str,
    live_row: dict,
    metrics: dict,
    log_fn,
    now: datetime,
) -> bool:
    """
    Run activation check and log rejection if not activated.

    Parameters
    ----------
    symbol   : str
    live_row : dict   — live scrape row for this symbol
    metrics  : dict   — enriched metrics from core/features.py
    log_fn   : callable — log_event from main.py
    now      : datetime — current scan time (EAT)

    Returns
    -------
    bool — True if activated, False otherwise
    """
    activated = is_activated(live_row, metrics, now)

    # Always emit an activation event so logs can be reconciled with scan summaries.
    avg_vol = metrics.get("avg_volume_20d", 0.0) or 0
    today_vol = float(live_row.get("volume", 0) or 0)
    today_price = float(live_row.get("price") or live_row.get("close", 0))
    high_5d = float(metrics.get("high_5d", 0) or 0)
    vol_ratio = compute_volume_ratio(today_vol, avg_vol, now) if avg_vol else None

    log_data = {
        "symbol": symbol,
        "volume_ratio": round(vol_ratio, 2) if vol_ratio is not None else None,
        "distance_to_high_5d": round(today_price / high_5d, 4) if high_5d else None,
        "activated": bool(activated),
    }
    if not activated:
        log_data["reason"] = activation_reason(live_row, metrics, now)

    log_fn("ACTIVATION_CHECK", log_data)

    return activated

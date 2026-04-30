"""
Frozen MVP modules 1–5 — do not expand entry logic or add filters here.

Threshold is a stability probe; MFE/MAE define capture vs pain for calibration.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any


HIGH_5D_PULLBACK = 0.98


def trading_days_since_entry(entry_date: date, as_of: date) -> int:
    """
    Trading sessions completed after entry_date through as_of (NSE weekday proxy).
    Entry day counts as 0; first weekday after entry = 1.
    """
    if as_of <= entry_date:
        return 0
    c = 0
    d = entry_date + timedelta(days=1)
    while d <= as_of:
        if d.weekday() < 5:
            c += 1
        d += timedelta(days=1)
    return c


def mvp_entry_rejection(row: Any, threshold: float) -> dict[str, Any] | None:
    """
    Module 1 — entry (frozen). Returns None if allowed, else kwargs for SIGNAL_REJECTED.

    Score gate is strict ``>`` so a day capped at the threshold does not enter
    (calibration: thin sessions stay flat; strong days push through).
    """
    ml_score = float(row["ml_score"])
    current_score = ml_score
    prev_score = row.get("prev_score") if isinstance(row, dict) else None
    if ml_score <= threshold:
        return {
            "reason": "SCORE_BELOW_THRESHOLD",
            "ml_score": round(ml_score, 2),
            "threshold": threshold,
            "delta_to_threshold": round(threshold - ml_score, 1),
            "score_velocity": round(current_score - prev_score, 1) if prev_score else None,
        }
    close = float(row["close"])
    high_5d = float(row["high_5d"])
    if close > high_5d * HIGH_5D_PULLBACK:
        ratio = close / high_5d if high_5d else 0.0
        return {
            "reason": "HIGH_5D_GUARD_BLOCKED",
            "close": round(close, 4),
            "high_5d": round(high_5d, 4),
            "ratio": round(ratio, 4),
        }
    return None


def should_enter(row: Any, threshold: float) -> bool:
    """Module 1 — entry (frozen); True iff ``mvp_entry_rejection`` is None."""
    return mvp_entry_rejection(row, threshold) is None


def new_position(entry_price: float) -> dict[str, Any]:
    """Module 2 — initial state for a new trade."""
    ep = float(entry_price)
    return {
        "entry_price": ep,
        "highest_price": ep,
        "lowest_price": ep,
        "mfe": 0.0,
        "mae": 0.0,
        "days": 0,
        "stop": ep * 0.95,
    }


def update_excursions(pos: dict[str, Any], high: float, low: float) -> None:
    """Module 3 — run every bar (high/low vs entry)."""
    ep = float(pos["entry_price"])
    hi = float(high)
    lo = float(low)
    pos["highest_price"] = max(float(pos["highest_price"]), hi)
    pos["lowest_price"] = min(float(pos["lowest_price"]), lo)
    pos["mfe"] = max(float(pos["mfe"]), (hi - ep) / ep)
    pos["mae"] = min(float(pos["mae"]), (lo - ep) / ep)


def compute_stop(pos: dict[str, Any], current_price: float) -> float:
    """Module 4 — deterministic monotonic tightening."""
    entry = float(pos["entry_price"])
    high = float(pos["highest_price"])
    px = float(current_price)
    profit = (px - entry) / entry

    if profit >= 0.10:
        new_stop = high * 0.92
    elif profit >= 0.06:
        new_stop = entry * 1.01
    else:
        new_stop = entry * 0.95

    pos["stop"] = max(float(pos["stop"]), float(new_stop))
    return float(pos["stop"])


def should_exit(pos: dict[str, Any], current_price: float, low: float | None = None) -> str | None:
    """
    Module 5 — stop or time decay (returns reason token or None).

    Stop uses intrabar low (long); time exit uses close for profit.
    """
    entry = float(pos["entry_price"])
    px = float(current_price)
    lo = float(low) if low is not None else px
    profit = (px - entry) / entry

    if lo <= float(pos["stop"]):
        return "STOP"

    max_hold = int(pos.get("max_hold_days", 10))
    if int(pos["days"]) >= max_hold and profit < 0.03:
        return "TIME_STOP"

    return None


def exit_fill_price(pos: dict[str, Any], low: float, close: float) -> float:
    """Long: if low breaches stop, fill at stop; else close."""
    st = float(pos["stop"])
    if float(low) <= st:
        return st
    return float(close)

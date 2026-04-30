"""
Signal Engine — phase detection, signal scoring, score fusion, and entry validation.

All functions are stateless: they receive data and return values.
No I/O, no side effects. Easy to unit-test and swap.

Layer responsibilities (V3 architecture):
  detect_phase   → what is the market doing right now?
  score_signal   → how actionable is this phase for a long entry?
  fuse_score     → Score Fusion: combine phase score with corporate action boost
  validate_entry → given a confirmed pullback, does the maths justify a trade?

Data ordering note
──────────────────
get_history() returns rows ORDER BY timestamp DESC, so iloc[0] is the
most recent bar and iloc[-1] is the oldest.  For single-bar comparisons
(CAPITULATION, EXPANSION volume check) this is correct as-is.

For vol_trend — which needs a chronological diff — we reverse the volume
series before calling .diff() so that a positive trend means rising volume
and a negative trend means falling volume.  Without the reversal the sign
is inverted and COMPRESSION/EXHAUSTION detection fires on rising volume.
"""

import logging
import math
from datetime import datetime

import pandas as pd

from config.params import ROUND_TRIP_FEE_PCT

logger = logging.getLogger(__name__)


def detect_phase(recent: pd.DataFrame, price: float, chg: float, params: dict) -> str:
    """
    Classify current market phase from the last N price/volume bars.

    Returns one of:
        CAPITULATION, EXHAUSTION, COMPRESSION, EXPANSION, NEUTRAL,
        or INSUFFICIENT_DATA if history is too short or volume data is absent.
    """
    if len(recent) < 15:
        return "INSUFFICIENT_DATA"

    # Volume is required for meaningful phase detection.
    # A missing column means we cannot distinguish real breakouts from noise.
    if 'volume' not in recent.columns or recent['volume'].isna().all():
        return "INSUFFICIENT_DATA"

    avg_vol   = recent['volume'].mean()
    range_pct = (recent['price'].max() - recent['price'].min()) / recent['price'].mean()

    # Reverse to chronological (oldest → newest) before diff so that
    # vol_trend > 0 means rising volume and < 0 means falling volume.
    vol_trend = recent['volume'].iloc[::-1].diff().mean()

    # iloc[0] is the most recent bar (DESC order from DB)
    current_vol = recent['volume'].iloc[0]

    if chg < -2.0 and current_vol > avg_vol * 1.2:
        return "CAPITULATION"
    if chg < 0 and chg >= -2.0 and vol_trend < 0:
        return "EXHAUSTION"
    if range_pct < params["compression_range_pct"] and vol_trend < 0:
        return "COMPRESSION"
    # recent['price'].iloc[1:] = all bars except the current one (older history)
    if price > recent['price'].iloc[1:].max() * 1.001 and current_vol > avg_vol * 1.3:
        return "EXPANSION"
    return "NEUTRAL"


def score_signal(phase: str, chg: float) -> float:
    """
    Return a numeric confidence score for the signal.
    Higher = more actionable for a long entry. Negative = avoid.

    EXPANSION + negative change (55 → 35): a breakout candle that is already
    pulling back on the same bar is a weak breakout, not a strong one.
    Scoring it above COMPRESSION positive (40) was misleading.
    """
    table: dict[str, float] = {
        "EXPANSION":         90.0 if chg > 0 else 35.0,
        "COMPRESSION":       75.0 if chg > 0 else 40.0,
        "EXHAUSTION":        50.0 if chg > 0 else 20.0,
        "CAPITULATION":     -80.0,
        "NEUTRAL":            0.0,
        "INSUFFICIENT_DATA":  0.0,
    }
    return table.get(phase, 0.0)


def fuse_score(
    phase: str,
    chg: float,
    event_info: dict | None = None,
) -> tuple[float, bool, str, int]:
    """
    Score Fusion layer — the explicit bridge between Signal Engine and Event Engine.

    Combines the raw phase score with any active corporate action boost into a
    single final score. This is the named layer in the V3 architecture:

        Signal Engine → Event Engine → Score Fusion → Notification Layer

    Returns:
        final_score      — the score the rest of the system acts on
        event_flag       — True if a current event influenced this score
        event_type       — e.g. "DIVIDEND", empty string if no event
        confidence_boost — integer points added by the event (0 if none)
    """
    from core.events import apply_event_boost  # deferred import avoids circular
    base = score_signal(phase, chg)
    return apply_event_boost(base, phase, event_info)


def entry_time_ok(now: datetime, params: dict) -> tuple[bool, str]:
    """
    Return (allowed, reason) for the given moment.

    Entry window is read from params so it can be adjusted without a code change.
    Defaults: 10:30–14:45 EAT on weekdays.

    Boundary verification (inclusive open, exclusive close):
        10:29 → h<10 False, h==10 and m<30 True  → blocked  EARLY_SESSION ✓
        10:30 → h<10 False, h==10 and m<30 False  → allowed              ✓
        14:44 → h>14 False, h==14 and m>=45 False → allowed              ✓
        14:45 → h>14 False, h==14 and m>=45 True  → blocked  LATE_SESSION ✓
        15:00 → h>14 True                          → blocked  LATE_SESSION ✓
    """
    if now.weekday() >= 5:
        return False, "WEEKEND"

    oh = int(params.get("entry_open_h",  10))
    om = int(params.get("entry_open_m",  30))
    ch = int(params.get("entry_close_h", 14))
    cm = int(params.get("entry_close_m", 45))

    h, m = now.hour, now.minute

    # Debug bypass: allow scans outside the default entry window.
    # This is to unblock the data→signal pipeline when NSE timing does not
    # match the assumptions. Not meant to be enabled long-term.
    allow_early = bool(params.get("ALLOW_EARLY_SESSION", params.get("allow_early_session", False)))
    allow_late  = bool(params.get("ALLOW_LATE_SESSION",  params.get("allow_late_session",  False)))

    if h < oh or (h == oh and m < om):
        if allow_early:
            return True, "ok"
        return False, "EARLY_SESSION"
    if h > ch or (h == ch and m >= cm):
        if allow_late:
            return True, "ok"
        return False, "LATE_SESSION"

    return True, "ok"


def vol_ok(current_vol: float, avg_vol: float, now: datetime, params: dict) -> bool:
    """
    Return True if current volume clears the time-adaptive threshold.

    avg_vol == 0 always returns False — never assume volume is OK when
    the DB has no data (cold start / missing column).  Without this guard
    current_vol / avg_vol would raise ZeroDivisionError.

    Thresholds are stored in params so the calibrator can tune them:
        vol_mult_morning   — before noon  (default 1.2)
        vol_mult_afternoon — noon onward  (default 1.1)
    """
    if avg_vol <= 0:
        return False

    morning_mult   = float(params.get("vol_mult_morning",   1.2))
    afternoon_mult = float(params.get("vol_mult_afternoon", 1.1))
    threshold      = morning_mult if now.hour < 12 else afternoon_mult

    return (current_vol / avg_vol) >= threshold


def validate_entry(
    sym: str,
    price: float,
    breakout_price: float,
    recent_prices: pd.Series,
    equity: float,
    params: dict,
    slippage_buffer: float,
) -> dict | None:
    """
    Validate a pullback entry and compute position sizing.

    Returns a candidate dict if the trade clears the net-return threshold,
    or None if it doesn't. The caller decides whether to act on it.

    Fields in the returned dict:
        symbol, qty, limit_price, expected_net, stop_distance
    """
    # Must have pulled back at least 0.5 % from the breakout high
    if price > breakout_price * 0.995:
        return None

    # Stop distance: 2 × price std-dev, floored at stop_floor_pct of price
    std = recent_prices.std() if not recent_prices.empty else math.nan
    stop_distance = max(
        std * 2 if not math.isnan(std) else 0.0,
        price * params["stop_floor_pct"],
    )
    if stop_distance <= 0:
        return None

    # Position sizing: risk a fixed % of equity per trade
    risk_amount = equity * params["max_risk_pct"]
    qty         = int(risk_amount / stop_distance)
    if qty * price < 1_000:
        qty = int(1_000 / price) + 1   # enforce minimum position size

    expected_entry = round(price * slippage_buffer, 2)

    if recent_prices.empty:
        logger.warning(
            "validate_entry: empty recent_prices for %s — using price * 0.08 as c_width fallback",
            sym,
        )
        c_width = price * 0.08
    else:
        c_width = recent_prices.max() - recent_prices.min()

    expected_gross = (price + c_width - expected_entry) / expected_entry

    # Use the canonical round-trip fee from config, not a hardcoded constant.
    # The calibrator kill-switch uses the same value — keeping them in sync
    # prevents trades entering the journal that the kill-switch would have blocked.
    expected_net = expected_gross - ROUND_TRIP_FEE_PCT

    if expected_net <= params["min_net_target_pct"]:
        return None

    return {
        "symbol":        sym,
        "qty":           qty,
        "limit_price":   expected_entry,
        "expected_net":  round(expected_net, 3),
        "stop_distance": round(stop_distance, 4),
    }

"""
Execution Layer — convert validated signals into Ziidi manual-trade alerts.

This layer is intentionally dumb and stable.
No risk checks, no calibration logic, no state.
In: structured signal data.  Out: alert written to the event log.

Keeping it simple means it never breaks at the wrong moment.
"""

import json
from datetime import datetime

from dateutil import tz

from config.params import LOG_FILE

EAT = tz.gettz('Africa/Nairobi')


def _log(event_type: str, data: dict) -> None:
    ts    = datetime.now(EAT).isoformat()
    entry = {"timestamp": ts, "event": event_type, **data}
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def emit_buy_alert(
    sym: str,
    phase: str,
    score: float,
    qty: int,
    limit_price: float,
    expected_net: float,
    reason: str,
    event_flag: bool = False,
    event_type: str = "",
    confidence_boost: int = 0,
) -> dict:
    """
    Format and log a manual BUY alert.
    Returns the alert dict so the caller can record the open position.
    Event fields are included when a corporate action boosted this signal.
    """
    alert: dict = {
        "action":                  "MANUAL BUY",
        "symbol":                  sym,
        "quantity":                qty,
        "order_type":              "Own Price (limit)",
        "limit_price":             limit_price,
        "phase":                   phase,
        "score":                   score,
        "reason":                  reason,
        "expected_net_after_fees": expected_net,
    }
    if event_flag:
        alert["event_flag"]        = True
        alert["event_type"]        = event_type
        alert["confidence_boost"]  = confidence_boost
    _log("ZIIDI_BUY_ALERT", alert)
    return alert


def emit_sell_alert(sym: str, price: float, reason: str) -> dict:
    """Format and log a manual SELL alert."""
    alert = {
        "action":          "SELL",
        "symbol":          sym,
        "reason":          reason,
        "suggested_price": round(price, 2),
    }
    _log("ZIIDI_SELL_ALERT", alert)
    return alert

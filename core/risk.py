"""
Risk Governor — the last line of defence before any order is placed.

This layer enforces hard limits and can override every other layer.
It does not log, does not adjust parameters, does not pause the system.
It returns booleans. The caller decides what to do with them.

Three checks, in order of severity:
  1. equity_drawdown_ok   — 10 % peak-to-trough = account-level kill switch
  2. portfolio_risk_ok    — max open exposure cap
  3. trailing_stop_hit    — individual position exit trigger
"""


# 10 % drawdown from peak triggers a full system pause.
# This is the number that prevents a bad week from becoming a bad month.
DRAWDOWN_LIMIT = 0.90


def equity_drawdown_ok(equity: float, peak_equity: float) -> bool:
    """
    Returns True if the account is within the acceptable drawdown band.
    Returns False if a 10 % decline from peak has been breached — stop trading.
    """
    if peak_equity <= 0:
        return True
    return equity >= peak_equity * DRAWDOWN_LIMIT


def portfolio_risk_ok(n_open_positions: int, params: dict) -> bool:
    """
    Returns True if there is capacity to add another position.
    Returns False if total open risk has reached the portfolio cap.
    """
    open_risk = n_open_positions * params["max_risk_pct"]
    return open_risk < params["max_total_risk_pct"]


def trailing_stop_hit(price: float, highest_price: float, params: dict) -> bool:
    """
    Returns True if the current price has fallen far enough below the
    highest recorded price to trigger the trailing stop.
    """
    stop_pct = params.get("trailing_stop_pct", 0.05)
    return price < highest_price * (1.0 - stop_pct)

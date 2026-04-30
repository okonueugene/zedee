"""
Calibration Engine — adjusts system parameters based on analytics output.

Separated from analytics so each half can evolve independently:
  analytics.py   → answers "what happened?"
  calibrator.py  → decides "what should change?"

The single public function run() consumes a trade journal DataFrame and
returns (new_params, should_pause, reason).  It never writes to disk and
never mutates state — the caller (main.py) owns those decisions.

Decision hierarchy (first match wins):
  1. Expectancy kill switch  — no edge after fees → full pause, no param change
  2. Equity drawdown         — 10 % from peak    → full pause, no param change
  3. Risk scaling            — win rate bands     → adjust max_risk_pct
  4. Target calibration      — realised wins data → adjust min_net_target_pct
  5. Excursion diagnostics   — MAE/MFE signals    → log recommendations only
"""

import pandas as pd

from config.params import ROUND_TRIP_FEE_PCT
from core.risk import equity_drawdown_ok
from learning.analytics import core_metrics
from learning.journal import excursion_diagnostics


def run(
    df: pd.DataFrame,
    params: dict,
    equity: float,
    peak_equity: float,
) -> tuple[dict, bool, str]:
    """
    Analyse the trade journal and compute parameter updates.

    Args:
        df           — full trade journal DataFrame from journal.load_trades()
        params       — current system parameters dict
        equity       — current account equity
        peak_equity  — highest equity reached since system start

    Returns:
        new_params   — updated params dict (may equal params if no change)
        should_pause — True if entries must be halted immediately
        reason       — human-readable explanation for logging
    """
    min_trades = int(params.get("min_trades_to_calibrate", 20))
    if len(df) < min_trades:
        return params, False, f"Insufficient data ({len(df)}/{min_trades} trades)"

    metrics = core_metrics(df)
    if not metrics:
        return params, False, "core_metrics returned empty — check journal schema"

    win_rate   = metrics["win_rate"]
    expectancy = metrics["expectancy"]

    # ── 0. Safety governor: 5 consecutive losses ──────────────────────────────
    if len(df) >= 5 and "result" in df.columns:
        last5 = df.tail(5)
        if (last5["result"] == "LOSS").all():
            return params, True, "SAFETY GOVERNOR — 5 consecutive losses (pause entries)"

    # ── 1. Expectancy kill switch ─────────────────────────────────────────────
    # This is not a parameter problem. The system has no positive edge after fees.
    # Adjusting risk or targets cannot create edge from a negative-expectancy system.
    if expectancy < ROUND_TRIP_FEE_PCT:
        return params, True, (
            f"KILL SWITCH — expectancy {expectancy*100:.2f}% is below the "
            f"{ROUND_TRIP_FEE_PCT*100:.3f}% round-trip fee floor. "
            "No calibration can fix negative expectancy — investigate system logic."
        )

    # ── 2. Equity drawdown kill switch ────────────────────────────────────────
    if not equity_drawdown_ok(equity, peak_equity):
        return params, True, (
            f"KILL SWITCH — equity {equity:.0f} has breached the 10% drawdown "
            f"limit from peak {peak_equity:.0f}."
        )

    new_params = params.copy()
    notes: list[str] = []

    # ── 3. Adaptive rules (simple, deterministic) ────────────────────────────
    # Risk scaling
    if win_rate < 0.40:
        new_params["max_risk_pct"] = max(0.01, params.get("max_risk_pct", 0.02) * 0.80)
        notes.append(f"win_rate {win_rate:.0%} < 40% → max_risk_pct *= 0.8")
    if win_rate > 0.60 and expectancy > 0.05:
        new_params["max_risk_pct"] = min(0.025, params.get("max_risk_pct", 0.02) * 1.10)
        notes.append("win_rate > 60% and expectancy > 5% → max_risk_pct *= 1.1")

    # ── 4. Target calibration from realised win data ──────────────────────────
    pnl_col = "pnl_pct" if "pnl_pct" in df.columns else ("net_return_pct" if "net_return_pct" in df.columns else None)
    if pnl_col and not df[df['result'] == 'WIN'].empty:
        win_avg = float(df.loc[df['result'] == 'WIN', pnl_col].mean())
        new_target = max(0.05, min(0.15, win_avg * 0.75))
        new_params["min_net_target_pct"] = round(new_target, 4)
        notes.append(f"min_net_target_pct set to {new_target:.2%} (75% of avg win)")

    # ── 5. Excursion-based stop diagnostics ───────────────────────────────────
    diag = excursion_diagnostics(df)

    exit_gap = diag.get("avg_mfe_vs_exit_gap", 0.0)
    if exit_gap > 0.03:
        new_params["trailing_stop_pct"] = min(0.12, params.get("trailing_stop_pct", 0.05) * 1.20)
        notes.append(f"avg_mfe_vs_exit_gap {exit_gap*100:.1f}% → trailing_stop_pct *= 1.2")

    avg_win_mae = diag.get("avg_win_mae")
    if avg_win_mae is not None and abs(avg_win_mae) < new_params["stop_floor_pct"] * 0.5:
        new_params["stop_floor_pct"] = max(0.02, params.get("stop_floor_pct", 0.04) * 0.85)
        notes.append("avg_win_mae < 0.5 × stop_floor → stop_floor_pct *= 0.85")

    reason = "; ".join(notes) if notes else "No significant parameter drift detected"
    return new_params, False, reason

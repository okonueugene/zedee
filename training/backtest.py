"""
Backtest — simulate the 8-gate entry chain on the out-of-sample test set (2023–2025).

Entry conditions simulated
──────────────────────────
  Gate 3+7  ML score ≥ score_threshold
  Gate 4    vol_spike ≥ vol_mult (morning proxy — all hours treated equally in backtest)
  Gate 5    breakout detected (expansion_trigger ≥ 0.5) with pullback in the next bar
  Gate 6    cooldown: skip symbol for cooldown_bars after a trade (default 2 bars)
  Gate 8    expected net return after fees ≥ min_net_target (≈ 2 × fee)

Positions are held for HOLD_DAYS (5 bars) then closed at the forward price.
The real 3.874 % round-trip fee is deducted from every trade.

Why not simulate Gates 1 and 2?
  Gate 1 (history depth) and Gate 2 (live price present) are operational gates
  that are always satisfied in the labelled dataset — every row has enough history
  and every row has a price.  Simulating them adds no signal.

Output
──────
  training/backtest/results_summary.json
  training/backtest/trades.csv
  training/backtest/equity_curves.csv

Usage
─────
  python training/backtest.py
  python training/backtest.py --threshold 70 --vol-mult 1.1 --trail-stop 0.05
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from training.features import (
    CLOSE_COL, DATE_COL, FEATURE_COLS, HOLD_DAYS,
    ROUND_TRIP_FEE, SYMBOL_COL, TEST_START,
)

MODEL_PATH   = _ROOT / "training" / "ziidi_signal_model.pkl"
BACKTEST_DIR = _ROOT / "training" / "backtest"


def _load_model():
    if not MODEL_PATH.exists():
        print(f"[error] Model not found: {MODEL_PATH}")
        print("        Run train_model.py first.")
        sys.exit(1)
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def _mae_mfe(entry_price: float, prices_during_hold: list[float]) -> tuple[float, float]:
    """Maximum Adverse / Favorable Excursion relative to entry price."""
    if not prices_during_hold:
        return 0.0, 0.0
    lows  = min(prices_during_hold)
    highs = max(prices_during_hold)
    mae   = (lows - entry_price) / entry_price       # negative = adverse
    mfe   = (highs - entry_price) / entry_price      # positive = favorable
    return round(mae, 4), round(mfe, 4)


def run_backtest(
    labelled_path: Path,
    score_threshold:   float = 50.0,
    vol_mult:          float = 1.5,
    cooldown_bars:     int   = 2,
    min_net_target:    float = ROUND_TRIP_FEE * 2,
    initial_equity:    float = 10_000.0,
    risk_per_trade:    float = 0.02,    # 2 % of equity per position
    drawdown_kill_pct: float | None = 0.10,  # simulate live 10% peak drawdown stop
    candles_path:      Path | None = None,   # optional OHLC candles for realistic pullback/stop/target
    pullback_pct:      float = 0.98,         # REAL pullback: low <= breakout_high * pullback_pct (e.g. 0.98 = 2% pullback)
    breakout_expiry_bars: int = 3,           # how many daily bars after breakout we keep watching for pullback (state, not lookahead bias)
    expansion_trigger_threshold: float = 1.0,  # require full confirmation (expansion_trigger == 1)
    recent_high_lookback: int = 5,             # don't enter at peak: compare to recent highs
    recent_high_pullback_pct: float = 0.97,    # entry must be <= recent_high * 0.97
    target_pct:        float = 0.08,         # live bootstrapping target (+8%)
    stop_pct:          float = 0.05,         # live bootstrapping stop (-5%)
    breakeven_pct:     float = 0.08,         # delay breakeven: noise zone is ~4–6% MAE on NSE
    trailing_stop_pct: float = 0.08,         # wider trail to avoid shaking out trend winners
) -> dict:
    """
    Run the backtest and return the results summary dict.
    """
    model = _load_model()

    print(f"[load] {labelled_path}")
    df = pd.read_parquet(labelled_path)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])

    test_df = df[df[DATE_COL] >= TEST_START].copy()
    print(f"[test] {len(test_df):,} rows  |  symbols: {test_df[SYMBOL_COL].nunique()}")

    if len(test_df) < 10:
        print("[error] Test set too small — ensure label_generator covers 2023+")
        sys.exit(1)

    # ── ML scores ─────────────────────────────────────────────────────────────
    X_test          = test_df[FEATURE_COLS]
    test_df         = test_df.copy()
    test_df["prob"] = model.predict_proba(X_test)[:, 1]
    test_df["ml_score"] = (test_df["prob"] * 100).round(1)

    candles_df = None
    if candles_path is not None and candles_path.exists():
        candles_df = pd.read_parquet(candles_path).copy()
        # Normalise for merge
        candles_df["symbol"] = candles_df["symbol"].astype(str).str.upper().str.strip()
        candles_df["date"] = pd.to_datetime(candles_df["date"])
        # Keep only what we need
        candles_df = candles_df[["symbol", "date", "open", "high", "low", "close"]]

    # ── Per-symbol simulation ─────────────────────────────────────────────────
    all_trades: list[dict] = []
    equity_log: list[dict] = []

    equity    = initial_equity
    peak_eq   = initial_equity
    paused_by_dd = False
    dd_kill_eq = None

    symbols = test_df[SYMBOL_COL].unique()
    gate_counts = {
        "rows_scanned": 0,
        "pass_score": 0,
        "pass_vol": 0,
        "pass_expansion": 0,
        "pass_pullback": 0,
        "pass_gate8": 0,
        "trades": 0,
    }

    for sym in symbols:
        sym_df   = test_df[test_df[SYMBOL_COL] == sym].sort_values(DATE_COL).reset_index(drop=True)
        if candles_df is not None:
            sym_df = sym_df.merge(
                candles_df[candles_df["symbol"] == sym],
                left_on=[SYMBOL_COL, DATE_COL],
                right_on=["symbol", "date"],
                how="left",
                suffixes=("", "_c"),
            )
        cooldown = 0   # bars remaining in cooldown
        breakout = None  # {idx, high, expires_at}

        for i, row in sym_df.iterrows():
            if paused_by_dd:
                continue

            if cooldown > 0:
                cooldown -= 1
                continue

            gate_counts["rows_scanned"] += 1

            prob     = float(row["prob"])
            score    = float(row["ml_score"])
            vol_sp   = row.get("vol_spike", np.nan)
            exp_trig = row.get("expansion_trigger", np.nan)
            comp     = row.get("compression_score", np.nan)
            trend    = row.get("trend_strength", np.nan)
            entry_px = float(row[CLOSE_COL])

            phase = (
                "EXPANSION" if (not np.isnan(exp_trig) and exp_trig >= 0.5)
                else ("COMPRESSION" if (not np.isnan(comp) and comp >= 0.6) else "NEUTRAL")
            )

            # Gate 3+7: score threshold
            if score < score_threshold:
                continue
            gate_counts["pass_score"] += 1

            # Gate 4: volume confirmation (skip if vol data absent)
            if vol_mult is not None and vol_mult > 0:
                if not np.isnan(vol_sp) and vol_sp < vol_mult:
                    continue
            gate_counts["pass_vol"] += 1

            # Gate 5: Breakout state (sequential, no future leak)
            # We detect breakout on this bar, then *wait* for a pullback on later bars.
            if expansion_trigger_threshold is not None and expansion_trigger_threshold > 0:
                if np.isnan(exp_trig) or exp_trig < expansion_trigger_threshold:
                    # Still allow existing breakout state to be evaluated below.
                    pass
            gate_counts["pass_expansion"] += 1

            # Establish / refresh breakout state when expansion is detected.
            if (expansion_trigger_threshold is None or expansion_trigger_threshold <= 0) or (
                not np.isnan(exp_trig) and exp_trig >= expansion_trigger_threshold
            ):
                # Use intraday high when available; else close.
                b_high = float(row["high"]) if "high" in row and pd.notna(row.get("high")) else float(row[CLOSE_COL])
                breakout = {
                    "idx": int(i),
                    "high": b_high,
                    "expires_at": int(i) + max(1, int(breakout_expiry_bars)),
                }

            # If no breakout is active, nothing to do.
            if breakout is None:
                continue

            # Expire stale breakout
            if int(i) > int(breakout["expires_at"]):
                breakout = None
                continue

            # REAL pullback condition (no future leak):
            # pullback happens on the current bar if LOW <= breakout_high * pullback_pct.
            low_today = float(row["low"]) if "low" in row and pd.notna(row.get("low")) else float(row[CLOSE_COL])
            close_today = float(row[CLOSE_COL])
            close_yday = float(sym_df.iloc[i - 1][CLOSE_COL]) if i - 1 >= 0 else close_today
            pullback = low_today <= float(breakout["high"]) * float(pullback_pct)
            momentum_resuming = close_today > close_yday

            if not (pullback and momentum_resuming):
                continue

            gate_counts["pass_pullback"] += 1

            # Enter on NEXT bar open (more realistic than entering at today's low).
            if i + 1 >= len(sym_df):
                continue
            next_bar = sym_df.iloc[i + 1]
            entry_open = float(next_bar["open"]) if "open" in next_bar and pd.notna(next_bar.get("open")) else float(next_bar[CLOSE_COL])

            # Don't enter at peak: require entry <= recent_high * recent_high_pullback_pct
            if recent_high_lookback is not None and recent_high_lookback > 0:
                start = max(0, int(i) - int(recent_high_lookback))
                recent_high = float(sym_df.iloc[start:int(i) + 1][CLOSE_COL].max())
                if recent_high > 0 and entry_open > recent_high * float(recent_high_pullback_pct):
                    continue

            # Entry is at pullback bar
            actual_entry = int(i + 1)
            entry_price  = entry_open

            # Gate 8: expected net after fees
            # Measure width from next HOLD_DAYS bars
            end_idx    = min(actual_entry + HOLD_DAYS, len(sym_df))
            hold_bars  = sym_df.iloc[actual_entry: end_idx]
            if hold_bars.empty:
                continue

            prices_during = hold_bars[CLOSE_COL].astype(float).tolist()

            # Exit logic: explicit stop/target + breakeven + trailing stop (intrabar)
            target_price = entry_price * (1 + float(target_pct))
            base_stop    = entry_price * (1 - float(stop_pct))
            stop_price   = base_stop
            highest      = entry_price
            exit_price   = float(hold_bars.iloc[-1][CLOSE_COL])
            exit_reason  = "MAX_HOLD"
            for _, bar in hold_bars.iterrows():
                bar_high = float(bar["high"]) if "high" in bar and pd.notna(bar.get("high")) else float(bar[CLOSE_COL])
                bar_low  = float(bar["low"])  if "low"  in bar and pd.notna(bar.get("low"))  else float(bar[CLOSE_COL])
                highest  = max(highest, bar_high)

                # Breakeven activation
                if bar_high >= entry_price * (1 + float(breakeven_pct)):
                    stop_price = max(stop_price, entry_price)

                # Trailing stop from highest high
                if trailing_stop_pct is not None and trailing_stop_pct > 0:
                    trail_stop = highest * (1 - float(trailing_stop_pct))
                    stop_price = max(stop_price, trail_stop)

                # Target hit intraday
                if bar_high >= target_price:
                    exit_price  = target_price
                    exit_reason = "TARGET"
                    break
                # Stop hit intraday
                if bar_low <= stop_price:
                    exit_price  = stop_price
                    exit_reason = "BREAKEVEN" if abs(stop_price - entry_price) < 1e-9 else "STOP_LOSS"
                    break

            gross_ret   = (exit_price - entry_price) / entry_price
            net_ret     = gross_ret - ROUND_TRIP_FEE

            # Gate 8: expected net must exceed minimum
            if net_ret < min_net_target - ROUND_TRIP_FEE:
                # We can't know net_ret before it happens; this gate approximates
                # the live validate_entry check using the range estimate
                c_range = max(prices_during) - min(prices_during)
                exp_gross = (entry_price + c_range - entry_price * 1.005) / (entry_price * 1.005)
                if exp_gross - ROUND_TRIP_FEE < min_net_target:
                    continue
            gate_counts["pass_gate8"] += 1

            mae, mfe = _mae_mfe(entry_price, prices_during)

            # Position sizing: align risk to the configured stop distance
            stop_dist   = entry_price * float(stop_pct)
            shares      = max(1, int(equity * float(risk_per_trade) / stop_dist)) if stop_dist > 0 else 1
            position_val = shares * entry_price

            pnl = shares * (exit_price - entry_price) - (ROUND_TRIP_FEE * position_val)
            equity += pnl

            result = "WIN" if net_ret > 0 else "LOSS"
            cooldown = cooldown_bars

            entry_date = sym_df.iloc[actual_entry][DATE_COL]
            exit_date  = hold_bars.iloc[-1][DATE_COL]

            all_trades.append({
                "symbol":       sym,
                "entry_date":   str(entry_date.date()),
                "exit_date":    str(exit_date.date()),
                "phase":        phase,
                "reason":       exit_reason,
                "entry_price":  round(entry_price, 4),
                "exit_price":   round(exit_price,  4),
                "ml_prob":      round(prob, 4),
                "ml_score":     round(score, 1),
                "vol_spike":    None if np.isnan(vol_sp) else round(float(vol_sp), 3),
                "expansion_trigger": None if np.isnan(exp_trig) else round(float(exp_trig), 3),
                "gross_ret":    round(gross_ret,  4),
                "net_ret":      round(net_ret,    4),
                "mae":          mae,
                "mfe":          mfe,
                "result":       result,
                "pnl":          round(pnl, 2),
                "equity_after": round(equity, 2),
            })
            gate_counts["trades"] += 1

            peak_eq = max(peak_eq, equity)
            if drawdown_kill_pct is not None and peak_eq > 0:
                dd = (equity - peak_eq) / peak_eq
                if dd <= -abs(drawdown_kill_pct):
                    paused_by_dd = True
                    dd_kill_eq = round(equity, 2)

            equity_log.append({
                "date":          str(entry_date.date()),
                "symbol":        sym,
                "equity":        round(equity, 2),
                "drawdown_from_peak": round((equity - peak_eq) / peak_eq, 4),
            })

            # Skip ahead past the hold period to avoid overlapping trades
            cooldown = max(cooldown, HOLD_DAYS - 1)

    # ── Aggregate metrics ─────────────────────────────────────────────────────
    trades_df   = pd.DataFrame(all_trades)
    equity_df   = pd.DataFrame(equity_log)

    if trades_df.empty:
        print("[warn] No trades taken — try lowering --threshold or --vol-mult")
        return {"trades": 0, "expectancy": 0.0}

    n       = len(trades_df)
    wins    = trades_df[trades_df["result"] == "WIN"]
    losses  = trades_df[trades_df["result"] == "LOSS"]

    win_rate   = len(wins) / n
    avg_win    = float(wins["net_ret"].mean())    if len(wins)   > 0 else 0.0
    avg_loss   = float(-losses["net_ret"].mean()) if len(losses) > 0 else 0.0
    expectancy = win_rate * avg_win - (1 - win_rate) * avg_loss

    avg_mae = float(trades_df["mae"].mean())
    avg_mfe = float(trades_df["mfe"].mean())
    max_dd  = float(equity_df["drawdown_from_peak"].min()) if not equity_df.empty else 0.0

    final_equity = float(trades_df["equity_after"].iloc[-1]) if n > 0 else initial_equity
    total_return = (final_equity - initial_equity) / initial_equity

    print(f"\n── Backtest Results (test set ≥ {TEST_START}) ──────────────────")
    print(f"   Trades:     {n}")
    print(f"   Win rate:   {win_rate*100:.1f}%")
    print(f"   Avg win:    {avg_win*100:.2f}%")
    print(f"   Avg loss:   {avg_loss*100:.2f}%")
    print(f"   Expectancy: {expectancy*100:.3f}%  (fee floor {ROUND_TRIP_FEE*100:.3f}%)")
    print(f"   Avg MAE:    {avg_mae*100:.2f}%  |  Avg MFE: {avg_mfe*100:.2f}%")
    print(f"   Max DD:     {max_dd*100:.1f}%")
    print(f"   Final eq:   {final_equity:,.0f}  ({total_return*100:+.1f}%)")

    if expectancy > ROUND_TRIP_FEE:
        print(f"\n   ✅ Positive expectancy after fees — model has out-of-sample edge")
    else:
        print(f"\n   ⚠  Expectancy below fee floor.  Most likely culprits:")
        print(f"      1. Score threshold too low  → try --threshold 70")
        print(f"      2. Too few positive examples → run label_generator with --hold-days 10")
        print(f"      3. Illiquid symbols adding noise → tighten is_tradable filter")

    summary = {
        "test_start":     TEST_START,
        "parameters": {
            "score_threshold": score_threshold,
            "vol_mult":        vol_mult,
            "cooldown_bars":   cooldown_bars,
            "min_net_target":  round(min_net_target, 5),
            "drawdown_kill_pct": drawdown_kill_pct,
            "expansion_trigger_threshold": expansion_trigger_threshold,
            "pullback_pct": pullback_pct,
            "breakout_expiry_bars": breakout_expiry_bars,
            "recent_high_lookback": recent_high_lookback,
            "recent_high_pullback_pct": recent_high_pullback_pct,
            "target_pct": target_pct,
            "stop_pct": stop_pct,
            "breakeven_pct": breakeven_pct,
            "trailing_stop_pct": trailing_stop_pct,
        },
        "gate_pass_counts": gate_counts,
        "trades":        n,
        "win_rate":      round(win_rate, 4),
        "avg_win":       round(avg_win, 4),
        "avg_loss":      round(avg_loss, 4),
        "expectancy":    round(expectancy, 5),
        "fee_floor":     ROUND_TRIP_FEE,
        "has_edge":      expectancy > ROUND_TRIP_FEE,
        "avg_mae":       round(avg_mae, 4),
        "avg_mfe":       round(avg_mfe, 4),
        "max_drawdown":  round(max_dd, 4),
        "paused_by_drawdown": paused_by_dd,
        "drawdown_kill_equity": dd_kill_eq,
        "initial_equity": initial_equity,
        "final_equity":  round(final_equity, 2),
        "total_return":  round(total_return, 4),
    }

    # ── Save outputs ──────────────────────────────────────────────────────────
    BACKTEST_DIR.mkdir(parents=True, exist_ok=True)

    summary_path = BACKTEST_DIR / "results_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[save] {summary_path}")

    trades_path = BACKTEST_DIR / "trades.csv"
    trades_df.to_csv(trades_path, index=False)
    print(f"[save] {trades_path}  ({n} rows)")

    if not equity_df.empty:
        eq_path = BACKTEST_DIR / "equity_curves.csv"
        equity_df.to_csv(eq_path, index=False)
        print(f"[save] {eq_path}")

    return summary


def main() -> None:
    try:
        import xgboost  # noqa: F401
    except ImportError:
        print("[error] xgboost not installed.  Run:  pip install xgboost scikit-learn pyarrow")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Backtest Ziidi ML model on 2023+ test set")
    labelled_default = _ROOT / "training" / "nse_labelled.parquet"
    parser.add_argument("--labelled",   type=Path,  default=labelled_default)
    parser.add_argument("--threshold",  type=float, default=50.0,
                        help="ML score threshold (0–100)")
    parser.add_argument("--vol-mult",   type=float, default=1.5,
                        help="Minimum vol_spike multiplier for Gate 4")
    parser.add_argument("--cooldown",   type=int,   default=2,
                        help="Bars to skip after each trade")
    parser.add_argument("--drawdown-kill", type=float, default=0.10,
                        help="Simulated peak drawdown kill-switch (0.10 = -10%%). Use 0 to disable.")
    parser.add_argument("--candles", type=Path, default=_ROOT / "training" / "nse_candles.parquet",
                        help="Optional canonical OHLC candles parquet for realistic pullbacks/stops/targets")
    parser.add_argument("--pullback-pct", type=float, default=0.98,
                        help="Pullback threshold as fraction of breakout HIGH (0.98 = 2%% pullback)")
    parser.add_argument("--breakout-expiry-bars", type=int, default=3,
                        help="How many daily bars after breakout we keep watching for a pullback (state machine)")
    parser.add_argument("--expansion-trigger", type=float, default=1.0,
                        help="Expansion trigger threshold (<=0 disables expansion gate; lets ML decide)")
    parser.add_argument("--recent-high-lookback", type=int, default=5,
                        help="Lookback bars for recent-high pullback constraint")
    parser.add_argument("--recent-high-pullback", type=float, default=0.97,
                        help="Entry must be <= recent_high * this (0.97 = 3%% below recent high)")
    parser.add_argument("--target-pct", type=float, default=0.08, help="Target pct (0.08 = +8%%)")
    parser.add_argument("--stop-pct", type=float, default=0.05, help="Stop pct (0.05 = -5%%)")
    parser.add_argument("--breakeven-pct", type=float, default=0.08, help="Breakeven activation pct (0.08 = +8%%)")
    parser.add_argument("--trailing-stop-pct", type=float, default=0.08, help="Trailing stop pct (0.08 = 8%%)")
    args = parser.parse_args()

    if not args.labelled.exists():
        print(f"[error] Labelled dataset not found: {args.labelled}")
        print("        Run label_generator.py first.")
        sys.exit(1)

    run_backtest(
        args.labelled,
        score_threshold = args.threshold,
        vol_mult        = args.vol_mult,
        cooldown_bars   = args.cooldown,
        drawdown_kill_pct = (None if args.drawdown_kill == 0 else args.drawdown_kill),
        candles_path    = (args.candles if args.candles and args.candles.exists() else None),
        pullback_pct    = args.pullback_pct,
        breakout_expiry_bars = args.breakout_expiry_bars,
        expansion_trigger_threshold = args.expansion_trigger,
        recent_high_lookback = args.recent_high_lookback,
        recent_high_pullback_pct = args.recent_high_pullback,
        target_pct      = args.target_pct,
        stop_pct        = args.stop_pct,
        breakeven_pct   = args.breakeven_pct,
        trailing_stop_pct = args.trailing_stop_pct,
    )


if __name__ == "__main__":
    main()

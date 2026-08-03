"""
Ziidi Signal Engine v3.0 — main orchestrator.

Scan phases (run_scan):
  1. Market Data      — fetch NSE prices (always, no gate)
  2. Position Mgmt    — excursions + trailing stops (NEVER gated)
  3. Entry Gates      — time → drawdown → pause → risk cap
  4. Signal Gen       — 8-gate per-symbol chain → emit alerts
  5. State save       — breakout state + equity snapshot

Separating position management from entry gating means open positions
are always protected even when the system is in defensive posture.

Runtime state:
    params            — live calibrated parameters (persisted to system_params.json)
    equity            — live account equity
    paper_equity      — paper-trading equity (updated on every closed trade)
    peak_equity       — highest equity reached (drawdown kill-switch anchor)
    breakout_detected — {sym: {"price": float, "detected_at": ISO}} persisted to JSON
    last_trade_time   — {sym: datetime} for cooldown enforcement
    consecutive_fails — fetch failure counter; pause at 3
    pause_entries     — global entry gate; set/cleared by risk and calibration
"""

import json
import logging
import os
import socket
import sqlite3
import time
from collections import Counter
from datetime import date, datetime, timedelta

import pandas as pd
import schedule
from dateutil import tz

from config.params import (
    BREAKOUTS_FILE,
    INITIAL_CAPITAL,
    LOG_FILE,
    PAPER_TRADE,
    PRICE_DB,
    ROUND_TRIP_FEE_PCT,
    SCAN_INTERVAL_MIN,
    SLIPPAGE_BUFFER,
    WATCHLIST,
    WATCHLIST_EXTENDED,
    load_params,
    save_params,
)
from core.data import (
    delete_position,
    fetch_nse_data,
    get_history,
    get_last_fetch_report,
    init_dbs,
    load_equity,
    load_positions,
    save_equity_snapshot,
    save_position,
    save_price_data,
    update_highest_price,
)
from core.events import apply_event_boost, fetch_active_events
from core.mvp_engine import mvp_entry_rejection
from core.ml_scorer import MODEL_LOADED as _ML_AVAILABLE
from core.ml_scorer import score_from_history as _ml_score
from core.execution import emit_buy_alert, emit_sell_alert
from core.log_forwarder import start_log_forwarder
from core.notifications import dispatch_notification, get_severity
from core.risk import equity_drawdown_ok, portfolio_risk_ok, trailing_stop_hit
from core.signals import detect_phase, entry_time_ok, fuse_score, validate_entry, vol_ok
from core.features import enrich_universe
from core.universe import build_universe
from core.activation import compute_volume_ratio, run_activation_check
from learning.analytics import core_metrics
from learning.calibrator import run as run_calibrator
from learning.journal import (
    close_trade,
    get_cooldown_hours,
    load_trades,
    load_trades_db,
    mvp_tick,
    open_trade,
    update_excursions,
)

EAT = tz.gettz('Africa/Nairobi')
server = socket.gethostname()

# ── File logger ────────────────────────────────────────────────────────────────
_LOG_DIR = "logs"
os.makedirs(_LOG_DIR, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(_LOG_DIR, "system.log"),
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    encoding="utf-8",
)
_logger = logging.getLogger(__name__)

_SEVERITY_TO_LEVEL: dict[str, int] = {
    "CRITICAL": logging.CRITICAL,
    "WARNING":  logging.WARNING,
    "INFO":     logging.INFO,
    "SERVER":   server,
}

_log_forwarder = start_log_forwarder(os.path.join(_LOG_DIR, "system.log"))

# ── Runtime state ──────────────────────────────────────────────────────────────
params        = load_params()
equity        = INITIAL_CAPITAL
paper_equity  = INITIAL_CAPITAL
peak_equity   = INITIAL_CAPITAL

# {sym: {"price": float, "detected_at": ISO}} — persisted across restarts
breakout_detected: dict = {}
last_trade_time: dict   = {}
consecutive_fails: int  = 0
pause_entries: bool     = False
fetch_pause_until: datetime | None = None

# ── Daily stats for DAILY_SUMMARY (EAT) ───────────────────────────────────────
_daily_date: date | None = None
_daily_scans_started: int = 0
_daily_fetch_ok: int = 0
_daily_fetch_fail: int = 0
_daily_symbols_considered: int = 0
_daily_signal_decisions: int = 0
_daily_forced_entries: int = 0
_daily_rejections_by_gate: Counter = Counter()


# ── Logging ────────────────────────────────────────────────────────────────────

def log_event(event_type: str, data: dict) -> None:
    ts       = datetime.now(EAT).isoformat()
    severity = get_severity(event_type)
    entry    = {"timestamp": ts, "event": event_type, "severity": severity, **data}

    msg = f"{event_type} | {json.dumps(data, default=str)}"
    print(f"[{ts.split('T')[1][:8]} EAT] {severity:8s} {msg}")
    _logger.log(_SEVERITY_TO_LEVEL.get(severity, logging.INFO), msg)

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

    dispatch_notification(event_type, data)


def log_rejection(sym: str, gate: str, **extra) -> None:
    """
    Structured rejection — answers 'why did SCOM not fire today?'
    After one week of paper trading, query which gate is blocking the most
    candidates to find where tuning effort should go.
    """
    global _daily_rejections_by_gate
    _daily_rejections_by_gate[gate] += 1
    log_event("SIGNAL_REJECTED", {"symbol": sym, "gate": gate, **extra})


def _daily_rollover(now: datetime) -> None:
    """Reset daily counters when the calendar day changes (EAT)."""
    global _daily_date, _daily_scans_started, _daily_fetch_ok, _daily_fetch_fail
    global _daily_symbols_considered, _daily_signal_decisions, _daily_forced_entries
    global _daily_rejections_by_gate

    today = now.date()
    if _daily_date is None or today != _daily_date:
        _daily_date = today
        _daily_scans_started = 0
        _daily_fetch_ok = 0
        _daily_fetch_fail = 0
        _daily_symbols_considered = 0
        _daily_signal_decisions = 0
        _daily_forced_entries = 0
        _daily_rejections_by_gate = Counter()


def send_daily_summary() -> None:
    """
    DAILY_SUMMARY at 15:00 EAT (15 min before close).
    Structured summary of scans, gating, and open-position P&L.
    """
    now = datetime.now(EAT)
    _daily_rollover(now)

    positions_df = load_positions()
    open_positions = []
    for _, pos in positions_df.iterrows():
        sym = pos.get("symbol")
        if not sym:
            continue
        try:
            hist = get_history(str(sym), n=1)
            if hist.empty:
                continue
            last_px = float(hist["price"].iloc[0])
            entry_px = float(pos.get("entry_price", last_px))
            pnl_pct = (last_px - entry_px) / entry_px if entry_px else 0.0
            open_positions.append({
                "symbol": str(sym),
                "entry": round(entry_px, 2),
                "last":  round(last_px, 2),
                "pnl_pct": round(pnl_pct * 100, 2),
                "mfe_pct": round(float(pos.get("mfe", 0.0)) * 100, 2),
                "mae_pct": round(float(pos.get("mae", 0.0)) * 100, 2),
                "target":  round(float(pos.get("target_price")), 2) if pd.notna(pos.get("target_price")) else None,
                "stop":    round(float(pos.get("stop_price")), 2) if pd.notna(pos.get("stop_price")) else None,
            })
        except Exception:
            continue

    log_event("DAILY_SUMMARY", {
        "date":               str(now.date()),
        "scans_started":      _daily_scans_started,
        "fetch_ok":           _daily_fetch_ok,
        "fetch_fail":         _daily_fetch_fail,
        "symbols_considered": _daily_symbols_considered,
        "signal_decisions":   _daily_signal_decisions,
        "forced_entries":     _daily_forced_entries,
        "top_rejection_gates": _daily_rejections_by_gate.most_common(5),
        "open_positions":     len(open_positions),
        "positions":          open_positions[:10],
        "pause":              pause_entries,
    })


# ── Breakout state ─────────────────────────────────────────────────────────────

def _load_breakout_state() -> None:
    global breakout_detected
    if os.path.exists(BREAKOUTS_FILE):
        try:
            with open(BREAKOUTS_FILE) as f:
                breakout_detected = json.load(f)
        except Exception:
            breakout_detected = {}


def _save_breakout_state() -> None:
    try:
        with open(BREAKOUTS_FILE, "w") as f:
            json.dump(breakout_detected, f, indent=2, default=str)
    except Exception:
        pass


def _expire_stale_breakouts() -> None:
    """
    Remove breakout entries older than breakout_expiry_sessions days.
    Prevents a breakout detected on Monday from firing on Friday on an
    unrelated price move.
    """
    expiry_days = int(params.get("breakout_expiry_sessions", 3))
    today       = date.today()
    expired     = [
        sym for sym, state in breakout_detected.items()
        if (today - date.fromisoformat(str(state.get("detected_at", ""))[:10])).days > expiry_days
    ]
    for sym in expired:
        del breakout_detected[sym]
        log_event("BREAKOUT_EXPIRED", {"symbol": sym, "expiry_days": expiry_days})


# ── Health-ping helpers ────────────────────────────────────────────────────────

def _market_open(now: datetime) -> bool:
    """Used for health-ping reporting only — NOT for entry gating."""
    return (
        now.weekday() < 5 and (
            (now.hour == 9 and now.minute >= 30) or
            (10 <= now.hour <= 14) or
            (now.hour == 15 and now.minute == 0)
        )
    )


# ── Phase 2: Position management (NEVER gated) ────────────────────────────────

def _mvp_high_5d(history: pd.DataFrame) -> float:
    """Rolling 5-bar high max for frozen entry guard (live: price proxy if no high)."""
    h = history.head(5)
    if "high" in h.columns and not h["high"].isna().all():
        return float(h["high"].max())
    return float(h["price"].max())


def manage_positions(df: pd.DataFrame, positions_df: pd.DataFrame, now: datetime) -> None:
    """
    Update excursions and check trailing stops for every open position.

    This function MUST run on every scan tick regardless of whether new entries
    are paused.  Skipping it during a kill switch means:
      - MAE/MFE calibration data goes dark
      - Trailing stops don't fire
      - Positions that should have been exited are held indefinitely
    """
    global paper_equity, peak_equity

    if params.get("use_mvp_engine"):
        for _, pos in positions_df.iterrows():
            sym = pos["symbol"]
            row = df[df["symbol"] == sym]
            if row.empty:
                continue
            try:
                price = float(row["price"].iloc[0])
                max_hold = int(params.get("max_hold_days", 10))
                reason, fill = mvp_tick(sym, price, price, price, now, max_hold_days=max_hold)
                if reason is None or fill is None:
                    continue
                emit_sell_alert(sym, fill, f"MVP {reason}")
                if PAPER_TRADE:
                    pnl = close_trade(sym, fill, reason)
                    paper_equity += pnl
                    peak_equity = max(peak_equity, paper_equity)
                delete_position(sym)
                log_event("POSITION_CLOSED", {
                    "symbol": sym,
                    "exit_price": round(fill, 2),
                    "reason": reason,
                    "engine": "MVP",
                })
            except Exception as exc:
                log_event("POSITION_ERROR", {"symbol": sym, "error": str(exc)})
        return

    for _, pos in positions_df.iterrows():
        sym = pos["symbol"]
        row = df[df["symbol"] == sym]
        if row.empty:
            continue

        try:
            price   = float(row["price"].iloc[0])
            highest = float(pos.get("highest_price", pos["entry_price"]))

            # MAE / MFE — must never be skipped
            update_excursions(sym, price)

            # Keep trailing stop anchor current
            new_highest = max(highest, price)
            if new_highest > highest:
                update_highest_price(sym, new_highest)

            # Explicit target / stop exits (must be explicit for learning)
            target_px = pos.get("target_price")
            stop_px   = pos.get("stop_price")
            if target_px is not None and pd.notna(target_px) and price >= float(target_px):
                emit_sell_alert(sym, price, "Target hit")
                if PAPER_TRADE:
                    pnl = close_trade(sym, price, "TARGET")
                    paper_equity += pnl
                    peak_equity   = max(peak_equity, paper_equity)
                delete_position(sym)
                log_event("POSITION_CLOSED", {
                    "symbol":      sym,
                    "exit_price":  round(price, 2),
                    "reason":      "TARGET",
                })
                continue

            if stop_px is not None and pd.notna(stop_px) and price <= float(stop_px):
                emit_sell_alert(sym, price, "Stop loss hit")
                if PAPER_TRADE:
                    pnl = close_trade(sym, price, "STOP_LOSS")
                    paper_equity += pnl
                    peak_equity   = max(peak_equity, paper_equity)
                delete_position(sym)
                log_event("POSITION_CLOSED", {
                    "symbol":      sym,
                    "exit_price":  round(price, 2),
                    "reason":      "STOP_LOSS",
                })
                continue

            # Trailing stop exit
            if trailing_stop_hit(price, new_highest, params):
                reason = f"Trailing stop {params['trailing_stop_pct']*100:.0f}% from high"
                emit_sell_alert(sym, price, reason)
                if PAPER_TRADE:
                    pnl = close_trade(sym, price, "TRAIL_STOP")
                    paper_equity += pnl
                    peak_equity   = max(peak_equity, paper_equity)
                delete_position(sym)
                log_event("POSITION_CLOSED", {
                    "symbol":      sym,
                    "exit_price":  round(price, 2),
                    "stop_anchor": round(new_highest, 2),
                    "reason":      "TRAILING_STOP",
                })

        except Exception as exc:
            log_event("POSITION_ERROR", {"symbol": sym, "error": str(exc)})


# ── Phase 4: Signal generation (gated) ────────────────────────────────────────

def scan_mvp_entries(df: pd.DataFrame, now: datetime, current_equity: float) -> None:
    """
    Frozen MVP entry (Module 1 only): ml_score > threshold and close <= high_5d * 0.98.
    No volume/breakout/net-target gates; operational checks only (history, data, cooldown).
    """
    # Hard time gate — mirror legacy behavior to avoid out-of-session entries.
    time_ok, reason = entry_time_ok(now, params)
    if not time_ok:
        log_event("SCAN_GATED", {"gate": "TIME", "reason": reason})
        return

    if not _ML_AVAILABLE:
        log_event("MVP_SCAN_SKIP", {"reason": "ML model not loaded"})
        return

    global _daily_signal_decisions
    min_rows = max(22, int(params.get("min_history_rows", 15)))
    score_threshold = float(params.get("score_threshold", 55))
    risk_pct = float(params.get("max_risk_pct", 0.02))
    current_positions = load_positions()
    min_net = float(params.get("min_net_target_pct", 0.045))

    # ── Watchlist expansion (2026-04-24): enrich → universe → activation ──────
    conn = sqlite3.connect(PRICE_DB, timeout=30)
    try:
        enriched = enrich_universe(WATCHLIST_EXTENDED, conn)
    finally:
        conn.close()

    skipped = [s for s in WATCHLIST_EXTENDED if s not in enriched]
    for sym in skipped:
        log_event("ENRICHMENT_SKIP", {"symbol": sym, "reason": "INSUFFICIENT_HISTORY"})

    universe = build_universe(enriched)

    active_symbols: list[str] = []
    for sym in universe:
        sym_rows = df[df["symbol"] == sym]
        if sym_rows.empty:
            log_event("ACTIVATION_CHECK", {"symbol": sym, "activated": False, "reason": "NO_LIVE_DATA"})
            continue

        live_row = sym_rows.iloc[0].to_dict()
        metrics = enriched[sym]
        if run_activation_check(sym, live_row, metrics, log_event, now):
            active_symbols.append(sym)

    log_event(
        "SCAN_SUMMARY",
        {
            "extended_watchlist": len(WATCHLIST_EXTENDED),
            "enriched": len(enriched),
            "universe_after_advt": len(universe),
            "activated": len(active_symbols),
            "symbols_to_score": active_symbols,
        },
    )

    for sym in active_symbols:
        try:
            global _daily_symbols_considered
            _daily_symbols_considered += 1
            history = get_history(sym)
            if len(history) < min_rows:
                log_rejection(sym, "HISTORY", rows=len(history), required=min_rows)
                continue

            stock = df[df["symbol"] == sym]
            if stock.empty:
                log_rejection(sym, "DATA")
                continue

            price = float(stock["price"].iloc[0])
            ml_result = _ml_score(history)
            if ml_result is None:
                log_rejection(sym, "MVP", reason="ML_SCORE_UNAVAILABLE")
                continue

            ml_score = float(ml_result[0])
            prev_score = None
            try:
                # Score "velocity": compare current bar's ML score to the prior bar.
                # history is DESC; dropping the newest row approximates "previous" state.
                prev_result = _ml_score(history.iloc[1:]) if len(history) >= 23 else None
                prev_score = float(prev_result[0]) if prev_result is not None else None
            except Exception:
                prev_score = None
            high_5d = _mvp_high_5d(history)
            sym_metrics = enriched.get(sym, {})
            avg_vol = float(sym_metrics.get("avg_volume_20d", 0) or 0)
            today_vol = float(stock["volume"].iloc[0] if "volume" in stock.columns else 0)
            vol_ratio = compute_volume_ratio(today_vol, avg_vol, now) if avg_vol else None
            row = {"ml_score": ml_score, "prev_score": prev_score, "close": price, "high_5d": high_5d}
            mvp_rej = mvp_entry_rejection(row, score_threshold, volume_ratio=vol_ratio)
            if mvp_rej is not None:
                log_rejection(sym, "MVP", **mvp_rej)
                continue

            if not current_positions[current_positions["symbol"] == sym].empty:
                log_rejection(sym, "COOLDOWN", reason="POSITION_OPEN")
                continue

            cooldown_h = get_cooldown_hours(sym, params)
            if sym in last_trade_time:
                elapsed_h = (now - last_trade_time[sym]).total_seconds() / 3600
                if elapsed_h < cooldown_h:
                    log_rejection(sym, "COOLDOWN",
                                  reason="WITHIN_COOLDOWN",
                                  hours_remaining=round(cooldown_h - elapsed_h, 1))
                    continue

            qty = max(1, int((current_equity * risk_pct) / price))
            stop_px = round(price * 0.95, 2)
            log_event("SIGNAL_DECISION", {
                "symbol": sym,
                "price": round(price, 2),
                "phase": "MVP",
                "score": round(ml_score, 1),
                "engine": "MVP",
            })
            _daily_signal_decisions += 1

            emit_buy_alert(
                sym, "MVP", ml_score, qty, price, min_net,
                "MVP frozen entry (ml_score + high_5d guard)",
            )
            last_trade_time[sym] = now
            save_position(
                sym, price, qty, "MVP", ml_score,
                target_price=None,
                stop_price=stop_px,
            )
            open_trade(
                symbol=sym,
                entry_time=now.isoformat(),
                entry_price=float(price),
                shares=int(qty),
                phase="MVP",
                score=float(ml_score),
                target_price=None,
                stop_price=stop_px,
            )
        except Exception as exc:
            log_event("SYMBOL_ERROR", {"symbol": sym, "error": str(exc)})


def scan_for_entries(df: pd.DataFrame, now: datetime, current_equity: float) -> None:
    """
    8-gate per-symbol decision chain.

    Every rejection is logged with its gate name.  After one week query:
        df[df['event']=='SIGNAL_REJECTED'].groupby('gate').size().sort_values()
    to find which gate is the binding constraint and where to tune.

    Gates (in order):
        HISTORY   — minimum price bars
        DATA      — symbol present in today's live feed
        PHASE     — only actionable phases pass
        VOLUME    — adaptive morning/afternoon threshold
        BREAKOUT  — breakout detected and pullback met
        COOLDOWN  — position open or within reason-based cooldown period
        SCORE     — fused score above threshold
        NET_TARGET — expected net return after fees above minimum
    """
    # MVP control-flow must be correct even if `system_params.json` was edited
    # while this process was already running. Reload the flag from disk.
    try:
        use_mvp_engine_runtime = bool(load_params().get("use_mvp_engine", params.get("use_mvp_engine", False)))
    except Exception:
        use_mvp_engine_runtime = bool(params.get("use_mvp_engine", False))

    # Debug: confirm we are reading the same system_params.json we edited.
    try:
        cwd = os.getcwd()
        sp_path = os.path.abspath("system_params.json")
        file_flag = None
        if os.path.exists(sp_path):
            with open(sp_path, "r", encoding="utf-8") as f:
                file_flag = json.load(f).get("use_mvp_engine", None)
    except Exception:
        cwd = None
        sp_path = None
        file_flag = None

    log_event("MVP_ENGINE_MODE", {
        "use_mvp_engine": use_mvp_engine_runtime,
        "cwd": cwd,
        "system_params_path": sp_path,
        "system_params_file_flag": file_flag,
    })

    if use_mvp_engine_runtime:
        scan_mvp_entries(df, now, current_equity)
        return

    active_events = fetch_active_events(WATCHLIST)
    if active_events:
        for esym, einfo in active_events.items():
            log_event("EVENT_DETECTED", {
                "symbol":     esym,
                "event_type": einfo["event_type"],
            })

    _expire_stale_breakouts()

    current_positions = load_positions()
    min_rows          = int(params.get("min_history_rows", 15))
    score_threshold   = float(params.get("score_threshold", 55))
    pullback_pct      = float(params.get("pullback_pct", 0.995))

    alerts = []
    bootstrap_pool = []

    for sym in WATCHLIST:
        try:
            global _daily_symbols_considered
            _daily_symbols_considered += 1
            # ── Gate 1: History depth ─────────────────────────────────────
            history = get_history(sym)
            if len(history) < min_rows:
                log_rejection(sym, "HISTORY", rows=len(history), required=min_rows)
                continue

            # ── Gate 2: Live price data ───────────────────────────────────
            stock = df[df["symbol"] == sym]
            if stock.empty:
                log_rejection(sym, "DATA")
                continue

            price = float(stock["price"].iloc[0])
            chg   = float(stock["chg_pct"].iloc[0])
            vol   = float(stock["volume"].iloc[0]) if "volume" in stock.columns else 0.0

            # ── Gate 3: Phase / ML base score ────────────────────────────
            # Try ML scorer first.  If the model is not loaded or cannot build
            # features (cold start, too few bars), fall back to rule-based.
            recent     = history.head(20)
            ml_result  = _ml_score(history) if _ML_AVAILABLE else None

            if ml_result is not None:
                # ML path: phase filtering is implicit in the score.
                # A CAPITULATION-equivalent situation produces a very low
                # probability and is eliminated by Gate 7 (score threshold).
                ml_base_score, phase = ml_result
            else:
                # Rule-based fallback: hard phase filter + rule score
                ml_base_score = None
                phase = detect_phase(recent, price, chg, params)
                if phase in ("CAPITULATION", "NEUTRAL", "INSUFFICIENT_DATA"):
                    log_rejection(sym, "PHASE", phase=phase, scorer="RULE")
                    continue

            # ── Gate 4: Volume ────────────────────────────────────────────
            avg_vol = (
                float(recent["volume"].mean())
                if "volume" in recent.columns and not recent["volume"].isna().all()
                else 0.0
            )
            if not vol_ok(vol, avg_vol, now, params):
                reason    = "NO_VOLUME_DATA" if avg_vol <= 0 else "BELOW_THRESHOLD"
                vol_ratio = round(vol / avg_vol, 2) if avg_vol > 0 else None
                log_rejection(sym, "VOLUME", reason=reason, vol_ratio=vol_ratio)
                continue

            # ── Gate 5: Breakout + pullback ───────────────────────────────
            if phase == "EXPANSION":
                breakout_detected[sym] = {
                    "price":       price,
                    "detected_at": now.isoformat(),
                }

            if sym not in breakout_detected:
                log_rejection(sym, "BREAKOUT", reason="NO_BREAKOUT_DETECTED")
                continue

            breakout_px = breakout_detected[sym]["price"]
            needs_below = round(breakout_px * pullback_pct, 2)
            if price > needs_below:
                log_rejection(sym, "BREAKOUT",
                              reason="PULLBACK_NOT_MET",
                              price=round(price, 2),
                              breakout_price=round(breakout_px, 2),
                              needs_below=needs_below)
                continue

            # ── Gate 6: Cooldown ──────────────────────────────────────────
            if not current_positions[current_positions["symbol"] == sym].empty:
                log_rejection(sym, "COOLDOWN", reason="POSITION_OPEN")
                continue

            cooldown_h = get_cooldown_hours(sym, params)
            if sym in last_trade_time:
                elapsed_h = (now - last_trade_time[sym]).total_seconds() / 3600
                if elapsed_h < cooldown_h:
                    log_rejection(sym, "COOLDOWN",
                                  reason="WITHIN_COOLDOWN",
                                  hours_remaining=round(cooldown_h - elapsed_h, 1),
                                  cooldown_hours=cooldown_h)
                    continue

            # ── Gate 7: Signal score ──────────────────────────────────────
            # ML path: apply event boost directly to the ML probability score.
            # Rule-based path: fuse_score() calls score_signal() then applies
            # the same apply_event_boost() internally.
            if ml_base_score is not None:
                score, event_flag, event_type, confidence_boost = apply_event_boost(
                    ml_base_score, phase, active_events.get(sym)
                )
                scorer_label = "ML"
            else:
                score, event_flag, event_type, confidence_boost = fuse_score(
                    phase, chg, active_events.get(sym)
                )
                scorer_label = "RULE"

            if params.get("DEBUG_LOG_SCAN_RESULT", False):
                try:
                    log_event("SCAN_RESULT", {
                        "symbol":   sym,
                        "phase":    phase,
                        "ml_score": round(float(ml_base_score), 3) if ml_base_score is not None else None,
                        "score":    round(float(score), 3),
                        "threshold": score_threshold,
                        "scorer":    scorer_label,
                    })
                except Exception:
                    pass

            if score < score_threshold:
                # Keep candidates that passed structural gates so we can force
                # at least one paper trade/day during bootstrap (learning needs trades).
                if score >= 50:
                    bootstrap_pool.append({
                        "symbol": sym,
                        "price": price,
                        "breakout_px": breakout_px,
                        "phase": phase,
                        "score": score,
                        "event_flag": event_flag,
                        "event_type": event_type,
                        "confidence_boost": confidence_boost,
                    })
                log_rejection(sym, "SCORE",
                              score=round(score, 1),
                              threshold=score_threshold,
                              scorer=scorer_label)
                continue

            # ── Gate 8: Net return after fees ─────────────────────────────
            recent_prices = history["price"].head(10)
            candidate = validate_entry(
                sym, price, breakout_px,
                recent_prices, current_equity, params, SLIPPAGE_BUFFER,
            )
            if not candidate:
                log_rejection(sym, "NET_TARGET",
                              phase=phase,
                              score=round(score, 1),
                              min_net_pct=params["min_net_target_pct"])
                continue

            # ── All gates passed ──────────────────────────────────────────
            log_event("SIGNAL_DECISION", {
                "symbol":       sym,
                "price":        round(price, 2),
                "phase":        phase,
                "score":        round(score, 1),
                "scorer":       scorer_label,
                "expected_net": round(candidate["expected_net"] * 100, 2),
                "event_flag":   event_flag,
                "event_type":   event_type if event_flag else None,
            })
            global _daily_signal_decisions
            _daily_signal_decisions += 1

            alerts.append({
                **candidate,
                "phase":            phase,
                "score":            score,
                "event_flag":       event_flag,
                "event_type":       event_type,
                "confidence_boost": confidence_boost,
                "reason":           (
                    f"First pullback after {phase}"
                    + (f" [+{confidence_boost}pt {event_type}]" if event_flag else "")
                ),
            })

        except Exception as exc:
            log_event("SYMBOL_ERROR", {"symbol": sym, "error": str(exc)})

    # Bootstrap forcing: if nothing passed Gate 7 today, force the best
    # structural candidate so the journal gets trades (temporary).
    if not alerts and bootstrap_pool:
        best = max(bootstrap_pool, key=lambda x: x["score"])
        if best["score"] >= 50:
            recent_prices = get_history(best["symbol"])["price"].head(10)
            candidate = validate_entry(
                best["symbol"], float(best["price"]), float(best["breakout_px"]),
                recent_prices, current_equity, params, SLIPPAGE_BUFFER,
            )
            if candidate:
                log_event("BOOTSTRAP_FORCED_ENTRY", {
                    "symbol": best["symbol"],
                    "phase":  best["phase"],
                    "score":  round(best["score"], 1),
                    "threshold": score_threshold,
                })
                global _daily_forced_entries
                _daily_forced_entries += 1
                alerts.append({
                    **candidate,
                    "phase": best["phase"],
                    "score": best["score"],
                    "event_flag": best["event_flag"],
                    "event_type": best["event_type"],
                    "confidence_boost": best["confidence_boost"],
                    "reason": "BOOTSTRAP_FORCED (needed trades for learning)",
                })

    # Emit top 2 signals ranked by fused score
    for alert in sorted(alerts, key=lambda x: x["score"], reverse=True)[:2]:
        emit_buy_alert(
            alert["symbol"], alert["phase"], alert["score"],
            alert["qty"], alert["limit_price"], alert["expected_net"],
            alert["reason"],
            event_flag=alert["event_flag"],
            event_type=alert["event_type"],
            confidence_boost=alert["confidence_boost"],
        )
        last_trade_time[alert["symbol"]] = now
        save_position(
            alert["symbol"], alert["limit_price"], alert["qty"],
            alert["phase"], alert["score"],
            target_price=round(float(alert["limit_price"]) * 1.08, 2),
            stop_price=round(float(alert["limit_price"]) * 0.95, 2),
            event_flag=alert["event_flag"],
            event_type=alert["event_type"],
        )
        # Create the canonical trades-table record (the system's brain)
        open_trade(
            symbol=alert["symbol"],
            entry_time=now.isoformat(),
            entry_price=float(alert["limit_price"]),
            shares=int(alert["qty"]),
            phase=str(alert["phase"]),
            score=float(alert["score"]),
            target_price=round(float(alert["limit_price"]) * 1.08, 2),
            stop_price=round(float(alert["limit_price"]) * 0.95, 2),
        )


# ── Main scan loop ─────────────────────────────────────────────────────────────

def run_scan() -> None:
    global consecutive_fails, paper_equity, peak_equity, pause_entries, fetch_pause_until

    now = datetime.now(EAT)
    _daily_rollover(now)

    # Pre-market reminder (best-effort; fires only if scheduler hits 09:25)
    if now.weekday() < 5 and now.hour == 9 and now.minute == 25:
        log_event("REMINDER", {"msg": "09:25 EAT — Prepare limit orders for today"})

    # Delay the *fetch loop* until scan_start_h/m.
    # This prevents the scraper from burning through consecutive_fails during
    # the market open data-population gap (entry window opens at 10:30 anyway).
    sh = int(params.get("scan_start_h", 10))
    sm = int(params.get("scan_start_m", 0))
    if now.weekday() < 5 and (now.hour < sh or (now.hour == sh and now.minute < sm)):
        return

    global _daily_scans_started
    _daily_scans_started += 1

    # If we paused due to fetch failures, automatically retry after a backoff
    # instead of staying locked until 15:30 calibration.
    if pause_entries and fetch_pause_until is not None and now >= fetch_pause_until:
        pause_entries      = False
        consecutive_fails  = 0
        fetch_pause_until  = None
        log_event("SYSTEM_RESUMED", {"reason": "Fetch backoff elapsed — retrying within session"})

    # ── Phase 1: Market data (always) ─────────────────────────────────────────
    df = fetch_nse_data()
    if df is None or df.empty:
        consecutive_fails += 1
        global _daily_fetch_fail
        _daily_fetch_fail += 1
        fetch_report = get_last_fetch_report()
        log_event("SCAN_ABORT", {
            "reason":            "NO_DATA",
            "consecutive_fails": consecutive_fails,
            "sources_tried":     fetch_report.get("sources_tried", []),
            "source_results":    fetch_report.get("source_results", []),
            "fetch_error":       fetch_report.get("error"),
        })
        if consecutive_fails >= 3:
            pause_entries = True
            backoff_min = int(params.get("fetch_recovery_minutes", 60))
            fetch_pause_until = now + timedelta(minutes=backoff_min)
            log_event("SYSTEM_PAUSED", {
                "reason": f"{consecutive_fails} consecutive fetch failures",
                "paused_until": fetch_pause_until.isoformat(),
                "backoff_minutes": backoff_min,
                "hostname": socket.gethostname(),
                "sources_tried": fetch_report.get("sources_tried", []),
                "source_results": fetch_report.get("source_results", []),
            })
        return

    consecutive_fails = 0
    had_fetch_pause = bool(pause_entries and fetch_pause_until is not None)
    global _daily_fetch_ok
    _daily_fetch_ok += 1
    fetch_pause_until = None
    # If we previously paused due to repeated fetch failures, resume immediately
    # after a successful scrape.
    if had_fetch_pause:
        pause_entries = False
        log_event("SYSTEM_RESUMED", {"reason": "Fetch recovered"})
    save_price_data(df)
    positions_df = load_positions()

    # ── Phase 2: Position management (NEVER gated) ───────────────────────────
    manage_positions(df, positions_df, now)

    # Per-scan health visibility (INFO-level; Telegram only if NOTIFY_INFO=True)
    log_event("HEALTH", {
        "open_positions": int(len(positions_df)),
        "pause":          pause_entries,
        "consecutive_fails": consecutive_fails,
    })

    # ── Phase 3: Entry gates ─────────────────────────────────────────────────
    current_equity = paper_equity if PAPER_TRADE else equity

    # Gate A — time window
    time_ok, time_reason = entry_time_ok(now, params)
    if not time_ok:
        log_event("SCAN_GATED", {"gate": "TIME", "reason": time_reason})
        _save_breakout_state()
        return

    # Gate B — equity drawdown kill switch
    if not equity_drawdown_ok(current_equity, peak_equity):
        pause_entries = True
        log_event("KILL_SWITCH", {
            "reason": "10% equity drawdown from peak",
            "equity": round(current_equity, 2),
            "peak":   round(peak_equity, 2),
        })
        _save_breakout_state()
        return

    # Gate C — system pause (set by calibration or consecutive fetch failures)
    if pause_entries:
        log_event("SCAN_GATED", {
            "gate":   "SYSTEM_PAUSE",
            "reason": "Safety pause active — position management ran normally",
        })
        _save_breakout_state()
        return

    # Gate D — portfolio risk cap
    positions_df = load_positions()     # reload after manage_positions may have closed some
    if not portfolio_risk_ok(len(positions_df), params):
        log_event("SCAN_GATED", {
            "gate":            "RISK_CAP",
            "open_positions":  len(positions_df),
        })
        _save_breakout_state()
        return

    # ── Debug: force one open trade (journal + lifecycle test) ─────────────
    if params.get("FORCE_TEST_ENTRY", False):
        try:
            open_now = set(load_positions()["symbol"].tolist())
            chosen = None
            chosen_price = None
            for s in WATCHLIST:
                if s in open_now:
                    continue
                sub = df[df["symbol"] == s]
                if sub.empty:
                    continue
                chosen = s
                chosen_price = float(sub["price"].iloc[0])
                break

            if chosen is not None and chosen_price is not None:
                phase = "FORCE_TEST"
                score = 0.0
                qty = 1  # minimal size; we just need the lifecycle wiring to work
                target_price = round(chosen_price * 1.08, 2)
                stop_price   = round(chosen_price * 0.95, 2)

                # Create the same kinds of records scan_for_entries would create
                emit_buy_alert(
                    chosen, phase, score, qty, chosen_price,
                    float(params.get("min_net_target_pct", 0.045)),
                    "FORCED_TEST_ENTRY (non-production debug)",
                )
                save_position(
                    chosen, chosen_price, qty, phase, score,
                    target_price=target_price,
                    stop_price=stop_price,
                )
                open_trade(
                    symbol=chosen,
                    entry_time=now.isoformat(),
                    entry_price=float(chosen_price),
                    shares=int(qty),
                    phase=phase,
                    score=float(score),
                    target_price=target_price,
                    stop_price=stop_price,
                )
                global last_trade_time
                last_trade_time[chosen] = now
                log_event("FORCE_TEST_ENTRY", {
                    "symbol": chosen,
                    "entry_price": round(chosen_price, 4),
                    "qty": qty,
                    "target_price": target_price,
                    "stop_price": stop_price,
                })
        except Exception as exc:
            log_event("FORCE_TEST_ENTRY_ERROR", {"error": str(exc)})

    # ── Phase 4: Signal generation ───────────────────────────────────────────
    scan_for_entries(df, now, current_equity)

    # ── Phase 5: State save ──────────────────────────────────────────────────
    snap = paper_equity if PAPER_TRADE else equity
    peak_equity = max(peak_equity, snap)
    save_equity_snapshot(snap)
    _save_breakout_state()


# ── Calibration (daily 15:30 EAT) ─────────────────────────────────────────────

def run_calibration() -> None:
    global params, pause_entries

    df = load_trades_db()
    new_params, should_pause, reason = run_calibrator(df, params, paper_equity, peak_equity)

    log_event("CALIBRATION", {
        "trades": len(df),
        "action": "PAUSE" if should_pause else "CONTINUE",
        "reason": reason,
    })

    if should_pause and not pause_entries:
        pause_entries = True
        log_event("SYSTEM_PAUSED", {"reason": reason, "hostname": socket.gethostname()})

    if new_params != params:
        log_event("PARAM_UPDATE", {"old": params, "new": new_params})
        params = new_params
        save_params(params)

    if pause_entries and not should_pause:
        pause_entries = False
        log_event("SYSTEM_RESUMED", {"reason": reason})


# ── Health ping (every 60 min) ────────────────────────────────────────────────

def send_health_ping() -> None:
    df     = load_trades()
    recent = df.tail(20) if len(df) >= 20 else df

    if not recent.empty and "result" in recent.columns and "net_return_pct" in recent.columns:
        metrics    = core_metrics(recent)
        win_rate   = metrics.get("win_rate", 0.0)
        expectancy = metrics.get("expectancy", 0.0)
    else:
        win_rate   = 0.0
        expectancy = 0.0

    if pause_entries:
        state = "PAUSED"
    elif expectancy < 0:
        state = "DEGRADED"
    elif expectancy < ROUND_TRIP_FEE_PCT:
        state = "BELOW_FEE_FLOOR"
    else:
        state = "HEALTHY"

    positions_df   = load_positions()
    current_equity = paper_equity if PAPER_TRADE else equity

    log_event("HEALTH_PING", {
        "state":          state,
        "market":         "OPEN" if _market_open(datetime.now(EAT)) else "CLOSED",
        "win_rate_pct":   round(win_rate * 100, 1),
        "expectancy_pct": round(expectancy * 100, 2),
        "fee_floor_pct":  round(ROUND_TRIP_FEE_PCT * 100, 3),
        "positions":      len(positions_df),
        "equity":         round(current_equity, 2),
        "peak":           round(peak_equity, 2),
        "pause":          pause_entries,
        "trades_total":   len(df),
    })


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    init_dbs()
    equity, paper_equity, peak_equity = load_equity(INITIAL_CAPITAL)
    _load_breakout_state()

    log_event("SYSTEM_START", {
        "version":     "3.0",
        "mode":        "Self-Healing Ziidi Signal Engine",
        "watchlist":   WATCHLIST,
        "paper_trade": PAPER_TRADE,
        "scorer":      "ML" if _ML_AVAILABLE else "RULE",
    })
    log_event("SCHEDULE", {
        "scan":        f"every {SCAN_INTERVAL_MIN} min",
        "calibration": "daily 15:30 EAT",
        "health_ping": "every 60 min",
    })

    schedule.every(SCAN_INTERVAL_MIN).minutes.do(run_scan)
    schedule.every().day.at("15:10").do(run_calibration)
    schedule.every().day.at("15:00").do(send_daily_summary)
    schedule.every(60).minutes.do(send_health_ping)

    run_scan()

    while True:
        schedule.run_pending()
        time.sleep(60)

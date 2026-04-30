"""
Persistent trade journal with MAE / MFE excursion and event tracking.

MAE (Maximum Adverse Excursion)  — worst intra-trade price move against entry.
MFE (Maximum Favorable Excursion) — best intra-trade price move in our favour.
Both are stored as decimal fractions relative to entry price
  (e.g. mae = -0.03 means price went 3 % against us before the trade closed).

Full trade record schema:
    timestamp, symbol, entry_price, exit_price, shares,
    gross_return_pct, net_return_pct,
    mae, mfe, mfe_vs_exit_gap,
    phase, score,
    event_flag, event_type,
    entry_time, exit_time, holding_hours,
    result, reason

Event columns make every corporate-action-influenced trade individually
traceable:  df.groupby("event_type")["net_return_pct"].mean()

update_excursions() MUST be called on every scan tick while a position is open.
Every day the system runs without it is a day of lost calibration data.
The analytics engine is useless without these columns.
"""

import json
import os
import sqlite3
from datetime import date, datetime, timezone
from uuid import uuid4

import pandas as pd
from dateutil import parser as dt_parser
from dateutil import tz

from config.params import DB_FILE, LOG_FILE, ROUND_TRIP_FEE_PCT, TRADES_FILE

from core.mvp_engine import (
    compute_stop,
    exit_fill_price,
    should_exit,
    trading_days_since_entry,
    update_excursions as mvp_update_excursions,
)

EAT = tz.gettz('Africa/Nairobi')


# ── Internal helpers ───────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(EAT).isoformat()


def _log(event_type: str, data: dict) -> None:
    entry = {"timestamp": _now(), "event": event_type, **data}
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def _holding_hours(entry_time_str: str, exit_time_str: str) -> float:
    """Compute elapsed hours between two ISO timestamp strings."""
    try:
        entry = dt_parser.parse(entry_time_str)
        exit_ = dt_parser.parse(exit_time_str)
        # Ensure both are timezone-aware for correct arithmetic
        if entry.tzinfo is None:
            entry = entry.replace(tzinfo=timezone.utc)
        if exit_.tzinfo is None:
            exit_ = exit_.replace(tzinfo=timezone.utc)
        return (exit_ - entry).total_seconds() / 3600.0
    except Exception:
        return 0.0


# ── Schema migration ───────────────────────────────────────────────────────────

def migrate_positions_schema() -> None:
    """
    Idempotent migration: add any missing columns to the positions table.
    Safe to call on every startup — skips columns that already exist.
    """
    conn = sqlite3.connect(DB_FILE, timeout=30)
    existing = {row[1] for row in conn.execute("PRAGMA table_info(positions)")}

    migrations = {
        "trade_id":      "TEXT DEFAULT ''",
        "mae":          "REAL    DEFAULT 0.0",
        "mfe":          "REAL    DEFAULT 0.0",
        "target_price": "REAL    DEFAULT NULL",
        "stop_price":   "REAL    DEFAULT NULL",
        "score":        "REAL    DEFAULT 0.0",
        "lowest_price": "REAL    DEFAULT NULL",
        "event_flag":   "INTEGER DEFAULT 0",
        "event_type":   "TEXT    DEFAULT ''",
    }
    for col, definition in migrations.items():
        if col not in existing:
            conn.execute(f"ALTER TABLE positions ADD COLUMN {col} {definition}")

    conn.commit()
    conn.close()


def migrate_trades_schema() -> None:
    """
    Create the canonical trades table (the system's brain).

    This is the foundation for analytics, calibration, and safety governors.
    """
    conn = sqlite3.connect(DB_FILE, timeout=30)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            trade_id TEXT PRIMARY KEY,
            symbol TEXT,

            entry_time TEXT,
            exit_time TEXT,

            entry_price REAL,
            exit_price REAL,

            target_price REAL,
            stop_price REAL,

            shares INTEGER,

            phase TEXT,
            score REAL,

            result TEXT,
            pnl_pct REAL,

            mfe REAL,
            mae REAL,

            duration_minutes INTEGER,

            exit_reason TEXT
        )
    """)
    conn.commit()
    conn.close()


# ── Core excursion tracking ────────────────────────────────────────────────────

def update_excursions(sym: str, current_price: float) -> None:
    """
    Update MAE, MFE, and lowest_price for an open position at the current price.

    Call this on every scan tick for every open position — without exception.
    Skips silently if the symbol is not in the positions table.

    MAE          = min(running_mae, excursion)   — worst adverse move (relative)
    MFE          = max(running_mfe, excursion)   — best favourable move (relative)
    lowest_price = min(lowest_price, current)    — absolute lowest price seen
    """
    conn = sqlite3.connect(DB_FILE, timeout=30)
    row  = conn.execute(
        "SELECT entry_price, mae, mfe, lowest_price FROM positions WHERE symbol = ?", (sym,)
    ).fetchone()

    if row is None or row[0] == 0:
        conn.close()
        return

    entry_price, mae, mfe, lowest_price = row
    excursion = (current_price - entry_price) / entry_price

    new_mae          = min(mae if mae is not None else 0.0, excursion)
    new_mfe          = max(mfe if mfe is not None else 0.0, excursion)
    new_lowest_price = min(lowest_price if lowest_price is not None else current_price,
                           current_price)

    conn.execute(
        "UPDATE positions SET mae = ?, mfe = ?, lowest_price = ? WHERE symbol = ?",
        (new_mae, new_mfe, new_lowest_price, sym),
    )
    conn.commit()
    conn.close()


def mvp_tick(
    sym: str,
    high: float,
    low: float,
    close: float,
    now: datetime,
    max_hold_days: int = 10,
) -> tuple[str | None, float | None]:
    """
    MVP modules 3–5 on one tick: excursions, monotonic stop, exit check.

    Returns (exit_reason, exit_price) when the position should close; else (None, None).
    Persists mae/mfe/highest/lowest/stop to the positions table.
    """
    conn = sqlite3.connect(DB_FILE, timeout=30)
    row = conn.execute(
        """SELECT entry_price, highest_price, lowest_price, mae, mfe, stop_price, entry_time
           FROM positions WHERE symbol = ?""",
        (sym,),
    ).fetchone()
    if row is None:
        conn.close()
        return None, None

    ep, hp, lp, mae, mfe, stop_price, entry_time = row
    ep = float(ep)
    ed = date.fromisoformat(str(entry_time)[:10])
    pos = {
        "entry_price": ep,
        "highest_price": float(hp if hp is not None else ep),
        "lowest_price": float(lp if lp is not None else ep),
        "mae": float(mae if mae is not None else 0.0),
        "mfe": float(mfe if mfe is not None else 0.0),
        "days": trading_days_since_entry(ed, now.date()),
        "stop": float(stop_price if stop_price is not None else ep * 0.95),
        "max_hold_days": int(max_hold_days),
    }

    mvp_update_excursions(pos, high, low)
    compute_stop(pos, close)

    conn.execute(
        """UPDATE positions SET mae = ?, mfe = ?, lowest_price = ?, highest_price = ?, stop_price = ?
           WHERE symbol = ?""",
        (
            pos["mae"],
            pos["mfe"],
            pos["lowest_price"],
            pos["highest_price"],
            pos["stop"],
            sym,
        ),
    )
    conn.commit()
    conn.close()

    reason = should_exit(pos, close, low=low)
    if reason is None:
        return None, None

    fill = exit_fill_price(pos, low, close)
    return reason, fill


# ── Trade lifecycle controller ────────────────────────────────────────────────

def open_trade(
    symbol: str,
    entry_time: str,
    entry_price: float,
    shares: int,
    phase: str,
    score: float,
    target_price: float | None = None,
    stop_price: float | None = None,
    trade_id: str | None = None,
) -> str:
    """
    Create a trade record at entry time.
    The trade remains open until close_trade() populates exit fields.
    """
    if trade_id is None:
        trade_id = str(uuid4())

    conn = sqlite3.connect(DB_FILE, timeout=30)
    conn.execute(
        """INSERT OR REPLACE INTO trades
           (trade_id, symbol, entry_time, exit_time, entry_price, exit_price,
            target_price, stop_price, shares, phase, score,
            result, pnl_pct, mfe, mae, duration_minutes, exit_reason)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            trade_id, symbol, entry_time, None, entry_price, None,
            target_price, stop_price, shares, phase, score,
            None, None, 0.0, 0.0, None, None,
        ),
    )
    conn.commit()
    conn.close()
    return trade_id


def update_trade(symbol: str, current_price: float) -> None:
    """
    Real-time excursion tracking for the currently open trade in positions.
    This runs every scan via main.manage_positions().
    """
    update_excursions(symbol, current_price)


def close_trade(symbol: str, exit_price: float, reason: str) -> float:
    """
    Close the open trade for a symbol:
      - reads trade_id, entry fields, MAE/MFE from positions
      - writes exit fields into trades table
      - appends the legacy CSV journal for backwards compatibility

    Returns realised PnL in KES (paper mode accounting).
    """
    conn = sqlite3.connect(DB_FILE, timeout=30)
    pos = conn.execute(
        """SELECT trade_id, entry_price, shares, entry_time, phase, score, mae, mfe
           FROM positions WHERE symbol = ?""",
        (symbol,),
    ).fetchone()
    if not pos:
        conn.close()
        return 0.0

    trade_id, entry_price, shares, entry_time, phase, score, mae, mfe = pos
    if not trade_id:
        trade_id = str(uuid4())

    exit_time = _now()
    holding_hours = _holding_hours(entry_time, exit_time)
    duration_minutes = int(round(holding_hours * 60))

    gross_return = (exit_price - entry_price) / entry_price
    net_return   = gross_return - ROUND_TRIP_FEE_PCT
    result       = "WIN" if net_return > 0 else "LOSS"

    pnl_kes = int(shares) * (exit_price - entry_price)
    pnl_pct = net_return

    conn.execute(
        """UPDATE trades
           SET exit_time = ?, exit_price = ?, result = ?, pnl_pct = ?,
               mfe = ?, mae = ?, duration_minutes = ?, exit_reason = ?
           WHERE trade_id = ?""",
        (
            exit_time, exit_price, result, pnl_pct,
            float(mfe or 0.0), float(mae or 0.0), duration_minutes, reason,
            trade_id,
        ),
    )
    conn.commit()
    conn.close()

    # Legacy CSV journal append (keeps existing analytics working during migration)
    record = {
        "timestamp":        exit_time,
        "symbol":           symbol,
        "entry_price":      float(entry_price),
        "exit_price":       float(exit_price),
        "shares":           int(shares),
        "gross_return_pct": round(gross_return, 4),
        "net_return_pct":   round(net_return, 4),
        "mae":              round(float(mae or 0.0), 4),
        "mfe":              round(float(mfe or 0.0), 4),
        "mfe_vs_exit_gap":  round(float(mfe or 0.0) - gross_return, 4),
        "phase":            phase or "",
        "score":            float(score or 0.0),
        "entry_time":       entry_time,
        "exit_time":        exit_time,
        "holding_hours":    round(holding_hours, 2),
        "duration_minutes": duration_minutes,
        "result":           result,
        "reason":           reason,
        "trade_id":         trade_id,
    }
    pd.DataFrame([record]).to_csv(
        TRADES_FILE, mode="a", header=not os.path.exists(TRADES_FILE), index=False
    )

    _log("TRADE_CLOSED", {
        "symbol":          symbol,
        "trade_id":        trade_id,
        "pnl":             round(pnl_kes, 2),
        "pnl_pct":         round(pnl_pct * 100, 2),
        "mae_pct":         round(float(mae or 0.0) * 100, 2),
        "mfe_pct":         round(float(mfe or 0.0) * 100, 2),
        "phase":           phase,
        "duration_minutes": duration_minutes,
        "reason":          reason,
    })

    return float(pnl_kes)


# ── Trade logging ──────────────────────────────────────────────────────────────

def log_trade(
    sym: str,
    entry_price: float,
    exit_price: float,
    shares: int,
    entry_time: str,
    reason: str,
) -> float:
    """
    Close a trade: read final MAE/MFE/phase/score from the open position,
    compute all derived fields, write a complete record to the CSV journal.

    Returns realised PnL in KES so the caller can update paper_equity.
    """
    conn = sqlite3.connect(DB_FILE, timeout=30)
    row  = conn.execute(
        "SELECT mae, mfe, phase, score, event_flag, event_type FROM positions WHERE symbol = ?",
        (sym,),
    ).fetchone()
    conn.close()

    mae        = row[0] if row and row[0] is not None else 0.0
    mfe        = row[1] if row and row[1] is not None else 0.0
    phase      = row[2] if row and row[2] is not None else ""
    score      = row[3] if row and row[3] is not None else 0.0
    event_flag = bool(row[4]) if row and row[4] is not None else False
    event_type = row[5] if row and row[5] is not None else ""

    exit_time     = _now()
    holding_hours = _holding_hours(entry_time, exit_time)

    gross_return_pct = (exit_price - entry_price) / entry_price
    net_return_pct   = gross_return_pct - ROUND_TRIP_FEE_PCT
    result           = "WIN" if net_return_pct > 0 else "LOSS"
    pnl              = shares * (exit_price - entry_price)

    record = {
        "timestamp":        exit_time,
        "symbol":           sym,
        "entry_price":      entry_price,
        "exit_price":       exit_price,
        "shares":           shares,
        "gross_return_pct": round(gross_return_pct, 4),
        "net_return_pct":   round(net_return_pct, 4),
        # Excursion data — calibrates stops and exits
        "mae":              round(mae, 4),
        "mfe":              round(mfe, 4),
        "mfe_vs_exit_gap":  round(mfe - gross_return_pct, 4),
        # Signal context — by_phase and by_score analytics
        "phase":            phase,
        "score":            score,
        # Corporate action context — enables event performance tracking
        # df.groupby("event_type")["net_return_pct"].mean()
        "event_flag":       event_flag,
        "event_type":       event_type,
        # Time data — by_hold_time analytics
        "entry_time":       entry_time,
        "exit_time":        exit_time,
        "holding_hours":    round(holding_hours, 2),
        "result":           result,
        "reason":           reason,
    }

    df = pd.DataFrame([record])
    df.to_csv(TRADES_FILE, mode="a", header=not os.path.exists(TRADES_FILE), index=False)

    _log("TRADE_CLOSED", {
        "symbol":          sym,
        "pnl":             round(pnl, 2),
        "net_pct":         round(net_return_pct * 100, 2),
        "mae_pct":         round(mae * 100, 2),
        "mfe_pct":         round(mfe * 100, 2),
        "exit_gap_pct":    round((mfe - gross_return_pct) * 100, 2),
        "phase":           phase,
        "holding_hours":   round(holding_hours, 2),
    })

    return pnl


# ── Cooldown helpers ───────────────────────────────────────────────────────────

def get_last_exit_reason(sym: str) -> str:
    """
    Return the exit reason string for the most recent closed trade on sym.

    Reads from the trades CSV sorted by exit_time descending.
    Returns "NONE" if no trade exists yet — meaning no cooldown applies.
    """
    if not os.path.exists(TRADES_FILE):
        return "NONE"
    try:
        df = pd.read_csv(TRADES_FILE)
        if df.empty or "symbol" not in df.columns:
            return "NONE"
        trades = df[df["symbol"] == sym]
        if trades.empty:
            return "NONE"
        if "exit_time" in trades.columns:
            trades = trades.sort_values("exit_time", ascending=False)
        return str(trades.iloc[0].get("reason", "NONE"))
    except Exception:
        return "NONE"


def get_cooldown_hours(sym: str, params: dict) -> float:
    """
    Return the cooldown period in hours for sym based on its last exit reason.

    The cooldown_by_reason table is stored in params so it can be tuned
    through system_params.json without a code change.

    Default rules (from params DEFAULTS):
        TRAILING_STOP    → 6 h   stop-out needs time to re-establish trend
        TARGET_HIT       → 2 h   clean exit; can re-enter sooner
        MAX_HOLD_EXPIRED → 4 h   timed-out trend may be exhausted
        MANUAL           → 2 h
        NONE             → 0 h   no prior trade; no cooldown
        <unknown>        → 3 h   conservative default
    """
    reason = get_last_exit_reason(sym)
    rules: dict = params.get("cooldown_by_reason", {
        "TRAILING_STOP":    6,
        "TARGET_HIT":       2,
        "MAX_HOLD_EXPIRED": 4,
        "MANUAL":           2,
        "NONE":             0,
    })
    return float(rules.get(reason, 3.0))


# ── Analytics helpers ───────────────────────────────────────────────────────────

def load_trades() -> pd.DataFrame:
    """Load the full trade journal. Returns an empty DataFrame if no file exists."""
    if not os.path.exists(TRADES_FILE):
        return pd.DataFrame()
    return pd.read_csv(TRADES_FILE)


def load_trades_db() -> pd.DataFrame:
    """
    Load closed trades from the canonical SQLite trades table.
    Returns an empty DataFrame if the table doesn't exist yet.
    """
    try:
        conn = sqlite3.connect(DB_FILE, timeout=30)
        df = pd.read_sql(
            "SELECT * FROM trades WHERE exit_time IS NOT NULL AND exit_price IS NOT NULL",
            conn,
        )
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


def excursion_diagnostics(df: pd.DataFrame) -> dict:
    """
    Derive stop / exit calibration signals from MAE / MFE columns.

    Returned keys:
      avg_mfe_vs_exit_gap — if consistently > 3%, trailing stop is too tight
      avg_win_mae         — if magnitude < 50% of stop_floor, stop is too wide
      avg_loss_mae        — how deep losing trades went before closing
      avg_loss_mfe        — how far losing trades went in our favour (near-wins)
    """
    if df.empty or "mfe" not in df.columns or "mae" not in df.columns:
        return {}

    out: dict = {}

    if "mfe_vs_exit_gap" in df.columns:
        out["avg_mfe_vs_exit_gap"] = float(df["mfe_vs_exit_gap"].mean())

    wins   = df[df["result"] == "WIN"]
    losses = df[df["result"] == "LOSS"]

    if not wins.empty:
        out["avg_win_mae"] = float(wins["mae"].mean())

    if not losses.empty:
        out["avg_loss_mae"] = float(losses["mae"].mean())
        out["avg_loss_mfe"] = float(losses["mfe"].mean())

    return out

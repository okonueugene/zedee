"""
Market Data Layer — fetch, persist, and retrieve NSE price data and positions.
All I/O with the two SQLite databases and the equity CSV lives here.
"""

import json
import logging
import os
import sqlite3
from datetime import datetime
from uuid import uuid4

import pandas as pd
from dateutil import tz

from config.params import DB_FILE, EQUITY_FILE, LOG_FILE, PRICE_DB

EAT = tz.gettz('Africa/Nairobi')
_logger = logging.getLogger(__name__)

_last_fetch_report: dict = {}


def get_last_fetch_report() -> dict:
    """Diagnostics from the most recent fetch_nse_data() call."""
    return dict(_last_fetch_report)


def _log(event_type: str, data: dict) -> None:
    entry = {"timestamp": datetime.now(EAT).isoformat(), "event": event_type, **data}
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")
    _logger.info("%s | %s", event_type, json.dumps(data, default=str))


# ── Schema setup ───────────────────────────────────────────────────────────────

def init_dbs() -> None:
    """Create tables on first run and migrate any pre-existing databases."""
    from learning.journal import migrate_positions_schema, migrate_trades_schema  # deferred to avoid circular import

    for db_file in [DB_FILE, PRICE_DB]:
        conn = sqlite3.connect(db_file, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        if db_file == PRICE_DB:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS prices (
                    timestamp TEXT, symbol TEXT, price REAL,
                    chg_pct REAL, volume INTEGER,
                    PRIMARY KEY (timestamp, symbol))""")
        else:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS positions (
                    symbol        TEXT PRIMARY KEY,
                    trade_id      TEXT,
                    entry_price   REAL,
                    shares        INTEGER,
                    filled_shares INTEGER,
                    entry_time    TEXT,
                    highest_price REAL,
                    phase         TEXT,
                    mae           REAL    DEFAULT 0.0,
                    mfe           REAL    DEFAULT 0.0,
                    target_price  REAL    DEFAULT NULL,
                    stop_price    REAL    DEFAULT NULL,
                    score         REAL    DEFAULT 0.0,
                    lowest_price  REAL    DEFAULT NULL,
                    event_flag    INTEGER DEFAULT 0,
                    event_type    TEXT    DEFAULT '')""")
        conn.commit()
        conn.close()

    migrate_positions_schema()
    migrate_trades_schema()


# ── Equity curve ───────────────────────────────────────────────────────────────

def load_equity(initial_capital: float) -> tuple[float, float, float]:
    """
    Read the last equity value from the CSV.
    Returns (equity, paper_equity, peak_equity).
    Falls back to initial_capital if no file exists or the file is corrupt.
    """
    if os.path.exists(EQUITY_FILE) and os.path.getsize(EQUITY_FILE) > 0:
        try:
            df   = pd.read_csv(EQUITY_FILE)
            last = float(df['Equity'].iloc[-1])
            peak = float(df['Equity'].max())
            return last, last, peak
        except Exception as e:
            _log("EQUITY_LOAD_ERROR", {"error": str(e)})
    return initial_capital, initial_capital, initial_capital


def save_equity_snapshot(value: float) -> None:
    ts = datetime.now(EAT).isoformat()
    pd.DataFrame([{"Timestamp": ts, "Equity": value}]).to_csv(
        EQUITY_FILE, mode='a', header=not os.path.exists(EQUITY_FILE), index=False,
    )


# ── Position CRUD ──────────────────────────────────────────────────────────────

def load_positions() -> pd.DataFrame:
    conn = sqlite3.connect(DB_FILE, timeout=30)
    df   = pd.read_sql("SELECT * FROM positions", conn)
    conn.close()
    return df


def save_position(
    sym: str,
    entry_price: float,
    shares: int,
    phase: str,
    score: float = 0.0,
    highest_price: float | None = None,
    target_price: float | None = None,
    stop_price: float | None = None,
    event_flag: bool = False,
    event_type: str = "",
) -> None:
    if highest_price is None:
        highest_price = entry_price
    trade_id = str(uuid4())
    conn = sqlite3.connect(DB_FILE, timeout=30)
    conn.execute(
        """INSERT OR REPLACE INTO positions
           (symbol, trade_id, entry_price, shares, filled_shares, entry_time,
            highest_price, phase, mae, mfe, target_price, stop_price, score, lowest_price,
            event_flag, event_type)
           VALUES (?,?,?,?,?,?,?, ?,0.0,0.0,?,?,?,?,?,?)""",
        (sym, trade_id, entry_price, shares, shares,
         datetime.now(EAT).isoformat(), highest_price, phase,
         target_price, stop_price, score, entry_price,
         1 if event_flag else 0, event_type),
    )
    conn.commit()
    conn.close()


def delete_position(sym: str) -> None:
    conn = sqlite3.connect(DB_FILE, timeout=30)
    conn.execute("DELETE FROM positions WHERE symbol = ?", (sym,))
    conn.commit()
    conn.close()


def update_highest_price(sym: str, new_highest: float) -> None:
    conn = sqlite3.connect(DB_FILE, timeout=30)
    conn.execute("UPDATE positions SET highest_price = ? WHERE symbol = ?", (new_highest, sym))
    conn.commit()
    conn.close()


# ── Price history ──────────────────────────────────────────────────────────────

_HISTORY_OVERFETCH = 8
_HISTORY_MIN_RAW = 120


def _quote_key(price, volume) -> tuple[float, int]:
    p = float(price)
    try:
        v = int(volume)
    except (TypeError, ValueError):
        v = 0
    if not (p == p):  # NaN
        p = 0.0
    return (p, v)


def _dedupe_price_bars(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse consecutive rows with identical (price, volume).

    Expects DESC order (newest first, as stored/returned by get_history).
    """
    if df.empty:
        return df

    keep_idx: list = []
    prev_key = None
    for idx, row in df.iterrows():
        key = _quote_key(row["price"], row["volume"])
        if key != prev_key:
            keep_idx.append(idx)
            prev_key = key
    return df.loc[keep_idx].reset_index(drop=True)


def _load_raw_history(conn: sqlite3.Connection, sym: str, limit: int) -> pd.DataFrame:
    return pd.read_sql(
        "SELECT * FROM prices WHERE symbol = ? ORDER BY timestamp DESC LIMIT ?",
        conn,
        params=(sym, limit),
    )


def _latest_quotes(conn: sqlite3.Connection, symbols: list[str]) -> dict[str, tuple | None]:
    latest: dict[str, tuple | None] = {}
    for sym in symbols:
        row = conn.execute(
            "SELECT price, volume FROM prices WHERE symbol = ? ORDER BY timestamp DESC LIMIT 1",
            (sym,),
        ).fetchone()
        latest[sym] = row
    return latest


def save_price_data(df: pd.DataFrame) -> None:
    """
    Append live quotes only when price or volume changed vs the latest stored bar.

    Skips duplicate 15-minute scans (including overnight/weekend stale EOD prints)
    that were polluting ML feature history.
    """
    if df is None or df.empty:
        return

    conn = sqlite3.connect(PRICE_DB, timeout=30)
    try:
        symbols = df["symbol"].astype(str).tolist()
        latest = _latest_quotes(conn, symbols)
        now_ts = datetime.now(EAT).isoformat()

        rows: list[dict] = []
        skipped = 0
        for _, row in df.iterrows():
            sym = str(row["symbol"])
            price = float(row["price"])
            volume = int(row["volume"])
            prev = latest.get(sym)
            if prev is not None and _quote_key(prev[0], prev[1]) == _quote_key(price, volume):
                skipped += 1
                continue

            rows.append({
                "timestamp": now_ts,
                "symbol": sym,
                "price": price,
                "chg_pct": float(row["chg_pct"]) if pd.notna(row["chg_pct"]) else 0.0,
                "volume": volume,
            })
            latest[sym] = (price, volume)

        if rows:
            pd.DataFrame(rows).to_sql("prices", conn, if_exists="append", index=False)

        if skipped:
            _log("PRICE_SAVE_SKIP", {
                "skipped": skipped,
                "saved": len(rows),
                "symbols": len(symbols),
            })
    finally:
        conn.commit()
        conn.close()


def get_history(sym: str, n: int = 30) -> pd.DataFrame:
    """
    Return the last ``n`` distinct quote bars for ``sym`` (DESC order).

    Over-fetches raw rows then collapses consecutive duplicate (price, volume)
    prints so ML/scoring see session transitions, not repeated EOD snapshots.
    """
    raw_limit = max(n * _HISTORY_OVERFETCH, _HISTORY_MIN_RAW)
    conn = sqlite3.connect(PRICE_DB, timeout=30)
    try:
        df = _load_raw_history(conn, sym, raw_limit)
    finally:
        conn.close()
    return _dedupe_price_bars(df).head(n).reset_index(drop=True)


def get_deduped_price_volume_rows(
    symbol: str,
    conn: sqlite3.Connection,
    limit: int = 30,
) -> list[tuple[float, float | int | None]]:
    """Price/volume pairs for enrichment metrics (DESC, deduped)."""
    raw_limit = max(limit * _HISTORY_OVERFETCH, _HISTORY_MIN_RAW)
    df = _load_raw_history(conn, symbol, raw_limit)
    df = _dedupe_price_bars(df).head(limit)
    if df.empty:
        return []
    return list(zip(df["price"], df["volume"]))


def compact_price_history(
    symbols: list[str] | None = None,
    dry_run: bool = False,
) -> dict:
    """
    Remove consecutive duplicate (price, volume) bars already stored in PRICE_DB.

    Walks each symbol chronologically and keeps the first row of every unchanged run.
    """
    conn = sqlite3.connect(PRICE_DB, timeout=30)
    try:
        if symbols is None:
            symbols = [
                row[0]
                for row in conn.execute(
                    "SELECT DISTINCT symbol FROM prices ORDER BY symbol"
                ).fetchall()
            ]

        summary = {"symbols": len(symbols), "before": 0, "after": 0, "removed": 0}
        for sym in symbols:
            raw = pd.read_sql(
                "SELECT timestamp, symbol, price, chg_pct, volume "
                "FROM prices WHERE symbol = ? ORDER BY timestamp ASC",
                conn,
                params=(sym,),
            )
            if raw.empty:
                continue

            summary["before"] += len(raw)
            keep_rows: list[dict] = []
            prev_key = None
            for _, row in raw.iterrows():
                key = _quote_key(row["price"], row["volume"])
                if key == prev_key:
                    continue
                keep_rows.append(row.to_dict())
                prev_key = key

            summary["after"] += len(keep_rows)
            if dry_run or len(keep_rows) == len(raw):
                continue

            conn.execute("DELETE FROM prices WHERE symbol = ?", (sym,))
            pd.DataFrame(keep_rows).to_sql("prices", conn, if_exists="append", index=False)

        summary["removed"] = summary["before"] - summary["after"]
        if not dry_run and summary["removed"] > 0:
            conn.commit()
            _log("PRICE_HISTORY_COMPACTED", summary)
        return summary
    finally:
        conn.close()


# ── Live market feed ───────────────────────────────────────────────────────────

def fetch_nse_data() -> pd.DataFrame | None:
    """
    Fetch NSE live prices from the configured feed chain.

    Tries each source in ``NSE_FEED_PRIORITY`` until one returns data.
    Default: MyStocks pricelist (Synergy / NSE licensed) → kwayisi fallback.
    """
    global _last_fetch_report
    from config.params import NSE_FEED_PRIORITY
    from core.nse_feeds import fetch_from_source

    last_error: str | None = None
    source_results: list[dict] = []
    for source in NSE_FEED_PRIORITY:
        try:
            result = fetch_from_source(source)
        except Exception as exc:
            last_error = f"{source}: {exc}"
            source_results.append({
                "source": source,
                "status": "error",
                "error": str(exc),
            })
            _log("FETCH_ERROR", {
                "source": source,
                "error": str(exc),
                "reason": "provider exception",
            })
            continue

        if result is not None and not result.empty:
            source_results.append({
                "source": source,
                "status": "ok",
                "rows": len(result),
            })
            _last_fetch_report = {
                "success_source": source,
                "sources_tried": list(NSE_FEED_PRIORITY),
                "source_results": source_results,
            }
            _log("FETCH_OK", {
                "source": source,
                "rows": len(result),
                "sample_symbols": result["symbol"].head(3).tolist(),
            })
            try:
                print(f"FETCH_DEBUG {source}:", len(result))
                print(result.head(5).to_string(index=False))
            except Exception:
                pass
            return result

        source_results.append({
            "source": source,
            "status": "miss",
            "reason": "empty or unparseable response",
        })
        _log("FETCH_SOURCE_MISS", {
            "source": source,
            "reason": "empty or unparseable response",
        })

    _last_fetch_report = {
        "success_source": None,
        "sources_tried": list(NSE_FEED_PRIORITY),
        "source_results": source_results,
        "error": last_error or "all sources failed",
    }
    _log("FETCH_ERROR", {
        "error": last_error or "all sources failed",
        "sources_tried": list(NSE_FEED_PRIORITY),
        "source_results": source_results,
        "reason": "all feeds exhausted",
    })
    return None

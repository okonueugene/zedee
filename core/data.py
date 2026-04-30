"""
Market Data Layer — fetch, persist, and retrieve NSE price data and positions.
All I/O with the two SQLite databases and the equity CSV lives here.
"""

import io
import json
import os
import sqlite3
from datetime import datetime
from uuid import uuid4

import pandas as pd
import requests
from bs4 import BeautifulSoup
from dateutil import tz

from config.params import DB_FILE, EQUITY_FILE, LOG_FILE, PRICE_DB

EAT = tz.gettz('Africa/Nairobi')


def _log(event_type: str, data: dict) -> None:
    entry = {"timestamp": datetime.now(EAT).isoformat(), "event": event_type, **data}
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")


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

def save_price_data(df: pd.DataFrame) -> None:
    if df is None or df.empty:
        return
    conn       = sqlite3.connect(PRICE_DB, timeout=30)
    df         = df.copy()
    df['timestamp'] = datetime.now(EAT).isoformat()
    df.to_sql('prices', conn, if_exists='append', index=False)
    conn.commit()
    conn.close()


def get_history(sym: str, n: int = 30) -> pd.DataFrame:
    conn = sqlite3.connect(PRICE_DB, timeout=30)
    df   = pd.read_sql(
        "SELECT * FROM prices WHERE symbol = ? ORDER BY timestamp DESC LIMIT ?",
        conn, params=(sym, n),
    )
    conn.close()
    return df


# ── Live market feed ───────────────────────────────────────────────────────────

def fetch_nse_data() -> pd.DataFrame | None:
    """
    Scrape afx.kwayisi.org/nse — robust 2026 version.

    The page has several tables (NASI summary, gainers/losers, main listings).
    We identify the main stock table by requiring a 'ticker' column after
    normalising all headers to lowercase — the NASI and summary tables never
    contain a column called 'ticker', so false matches are impossible.
    """
    url = "https://afx.kwayisi.org/nse/"
    try:
        r = requests.get(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/134.0 Safari/537.36"
                )
            },
            timeout=20,
        )
        r.raise_for_status()
        status_code = int(r.status_code)
        resp_bytes  = int(len(r.content)) if r.content is not None else 0

        soup   = BeautifulSoup(r.text, 'html.parser')
        tables = soup.find_all('table')
        table_found = False
        last_reason = "no table with ticker/price/volume columns"

        for table in tables:
            # io.StringIO prevents pandas 2.x from trying to open the HTML as a file path
            try:
                df = pd.read_html(io.StringIO(str(table)))[0]
            except Exception:
                continue

            # Normalise to lowercase stripped strings before any matching
            df.columns = [str(c).strip().lower() for c in df.columns]

            # The main stock table is uniquely identified by having 'ticker',
            # 'price', AND 'volume' — none of the summary tables have all three.
            if not ('ticker' in df.columns and 'price' in df.columns and 'volume' in df.columns):
                continue
            table_found = True

            col_map = {}
            for c in df.columns:
                if c == 'ticker':
                    col_map[c] = 'symbol'
                elif 'volume' in c:
                    col_map[c] = 'volume'
                elif 'price' in c:
                    col_map[c] = 'price'
                elif 'change' in c:
                    col_map[c] = 'change'

            df = df.rename(columns=col_map)

            if 'change' in df.columns:
                df['prev_price'] = df['price'] - df['change']
                df['chg_pct']    = (df['change'] / df['prev_price'] * 100).round(2)

            df['symbol'] = df['symbol'].astype(str).str.upper().str.strip()
            result       = df[['symbol', 'price', 'chg_pct', 'volume']].copy()
            result       = result.dropna(subset=['symbol', 'price'])

            _log("FETCH_OK", {
                "rows":           len(result),
                "sample_symbols": result['symbol'].head(3).tolist(),
                "status":         status_code,
                "resp_bytes":     resp_bytes,
                "table_found":    True,
            })
            # Debug visibility for pipeline diagnosis
            try:
                print("FETCH_DEBUG strict:", len(result))
                print(result.head(5).to_string(index=False))
            except Exception:
                pass
            return result

        # ── Fallback parsing ─────────────────────────────────────────────
        # afx.kwayisi table layouts sometimes change slightly (e.g. missing
        # `volume`, `ticker` renamed to `symbol/code`, or `change` column
        # renamed). If strict parsing fails, try a looser match that still
        # guarantees output columns required by the live scan:
        #   symbol, price, chg_pct, volume
        fallback_reason: str = ""
        fallback_result: pd.DataFrame | None = None

        for table in tables:
            try:
                df = pd.read_html(io.StringIO(str(table)))[0]
            except Exception:
                continue

            df.columns = [str(c).strip().lower() for c in df.columns]
            cols = set(df.columns)

            sym_candidates = [
                c for c in df.columns
                if c in ("ticker", "symbol", "code") or "ticker" in c or "symbol" in c or "code" in c
            ]
            price_candidates = [
                c for c in df.columns
                if "price" in c or c in ("last", "ltp") or "last" in c or "ltp" in c
            ]
            chg_candidates = [
                c for c in df.columns
                if ("chg" in c) or ("change" in c) or ("pct" in c) or ("%chg" in c)
            ]
            vol_candidates = [
                c for c in df.columns
                if "volume" in c or "vol" in c or "qty" in c
            ]

            if not sym_candidates or not price_candidates:
                continue

            sym_col = sym_candidates[0]
            price_col = price_candidates[0]
            chg_col = chg_candidates[0] if chg_candidates else None
            vol_col = vol_candidates[0] if vol_candidates else None

            sym = df[sym_col].astype(str).str.upper().str.strip()
            price = pd.to_numeric(df[price_col], errors="coerce")

            if sym.empty or price.empty:
                continue

            out = pd.DataFrame({
                "symbol": sym,
                "price": price,
            })

            # chg_pct: if we can interpret a change column, compute; otherwise 0.
            if chg_col is not None:
                chg_raw = pd.to_numeric(df[chg_col], errors="coerce")
                chg_name = str(chg_col).lower()

                # If it's already a percent column (name suggests %/pct), use it.
                if "%" in chg_name or "pct" in chg_name or "percent" in chg_name or "chgpct" in chg_name:
                    out["chg_pct"] = chg_raw.round(2)
                else:
                    # Assume absolute change; derive previous price as price - change.
                    prev_price = price - chg_raw
                    out["chg_pct"] = (chg_raw / prev_price.replace(0, float("nan")) * 100).round(2)
            else:
                out["chg_pct"] = 0.0

            if vol_col is not None:
                out["volume"] = pd.to_numeric(df[vol_col], errors="coerce").fillna(0).astype(int)
            else:
                out["volume"] = 0

            out = out.dropna(subset=["symbol", "price"])
            # Allow smaller tables; later filters will still check needed symbols.
            if len(out) <= 5:
                continue

            fallback_result = out[["symbol", "price", "chg_pct", "volume"]].copy()
            fallback_reason = {
                "sym_col": sym_col,
                "price_col": price_col,
                "chg_col": chg_col,
                "vol_col": vol_col,
                "parsed_rows": int(len(fallback_result)),
            }
            break

        if fallback_result is not None:
            _log("FETCH_OK", {
                "rows":           len(fallback_result),
                "sample_symbols": fallback_result["symbol"].head(3).tolist(),
                "status":         status_code,
                "resp_bytes":     resp_bytes,
                "table_found":    True,
                "fallback_used":  True,
                "fallback_reason": fallback_reason,
            })
            # Debug visibility for pipeline diagnosis
            try:
                print("FETCH_DEBUG fallback:", len(fallback_result))
                print(fallback_result.head(5).to_string(index=False))
            except Exception:
                pass
            return fallback_result

        _log("FETCH_NO_TABLE", {
            "tables_scanned": len(tables),
            "status":         status_code,
            "resp_bytes":     resp_bytes,
            "table_found":    table_found,
            "reason":         last_reason,
            "fallback_reason": fallback_reason if fallback_result is None else None,
        })
        return None

    except Exception as e:
        status_code = None
        resp_bytes  = None
        try:
            status_code = int(getattr(locals().get("r", None), "status_code", None))
        except Exception:
            status_code = None
        try:
            _r = locals().get("r", None)
            resp_bytes = int(len(getattr(_r, "content", b"") or b""))
        except Exception:
            resp_bytes = None

        _log("FETCH_ERROR", {
            "error":      str(e),
            "url":        url,
            "status":     status_code,
            "resp_bytes": resp_bytes,
            "table_found": False,
            "reason":     "request/parse failure",
        })
        return None

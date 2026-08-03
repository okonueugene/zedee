"""
Multi-source NSE live price fetchers.

When afx.kwayisi.org blocks an IP, fall back to licensed NSE vendor feeds
(e.g. Synergy MyStocks pricelist) that return the same contract:
    symbol, price, chg_pct, volume
"""

from __future__ import annotations

import io
import re

import pandas as pd
import requests
from bs4 import BeautifulSoup

from config.params import KWAYISI_NSE_URL, MYSTOCKS_PRICELIST_URL, AFRICANFINANCIALS_URL

_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/134.0 Safari/537.36"
    )
}

_UP = "\u25b2"
_DOWN = "\u25bc"


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure standard output columns and dtypes."""
    out = df[["symbol", "price", "chg_pct", "volume"]].copy()
    out["symbol"] = out["symbol"].astype(str).str.upper().str.strip()
    out["price"] = pd.to_numeric(out["price"], errors="coerce")
    out["chg_pct"] = pd.to_numeric(out["chg_pct"], errors="coerce").fillna(0.0).round(2)
    out["volume"] = pd.to_numeric(out["volume"], errors="coerce").fillna(0).astype(int)
    return out.dropna(subset=["symbol", "price"])


def _parse_mystocks_change(raw) -> float:
    text = str(raw).replace(_UP, "").replace(_DOWN, "").replace("▲", "").replace("▼", "").strip()
    if not text or text in ("-", "—"):
        return 0.0
    negative = text.startswith("-") or _DOWN in str(raw)
    m = re.search(r"([\d.]+)", text)
    if not m:
        return 0.0
    val = float(m.group(1))
    return -val if negative and val > 0 else val


def _parse_mystocks_volume(raw) -> int:
    text = str(raw).replace(",", "").strip()
    if not text or text in ("-", "—"):
        return 0
    m = re.search(r"([\d.]+)\s*([KMB])?", text, re.I)
    if not m:
        return 0
    val = float(m.group(1))
    mult = {"K": 1e3, "M": 1e6, "B": 1e9}.get((m.group(2) or "").upper(), 1)
    return int(val * mult)


def _parse_mystocks_symbol(raw: str) -> str | None:
    name = str(raw).strip()
    m = re.search(r"\(([A-Z0-9]{2,12})\)\s*$", name)
    if m:
        return m.group(1).upper()
    token = name.split()[0].upper()
    if re.match(r"^[A-Z0-9]{2,12}$", token):
        return token
    return None


def fetch_mystocks_pricelist(url: str | None = None) -> pd.DataFrame | None:
    """
    Scrape Synergy MyStocks mobile pricelist (NSE licensed vendor).

    Source: https://live.mystocks.co.ke/m/pricelist
    Returns ~60–90 listed symbols with price, change %, and volume.
    """
    target = url or MYSTOCKS_PRICELIST_URL
    try:
        r = requests.get(target, headers=_BROWSER_HEADERS, timeout=20)
        r.raise_for_status()
    except Exception:
        return None

    try:
        tables = pd.read_html(io.StringIO(r.text))
    except Exception:
        return None

    if not tables:
        return None

    table = tables[0]
    if isinstance(table.columns, pd.MultiIndex):
        table.columns = [str(level[0]).strip() for level in table.columns]

    rows: list[dict] = []
    for _, row in table.iterrows():
        sym = _parse_mystocks_symbol(row.iloc[0])
        if not sym:
            continue
        try:
            price = float(str(row.iloc[1]).replace(",", ""))
        except (TypeError, ValueError):
            continue
        rows.append({
            "symbol": sym,
            "price": price,
            "chg_pct": _parse_mystocks_change(row.iloc[2]),
            "volume": _parse_mystocks_volume(row.iloc[3]),
        })

    if len(rows) <= 5:
        return None

    return _normalize(pd.DataFrame(rows))


def fetch_kwayisi(url: str | None = None) -> pd.DataFrame | None:
    """
    Scrape afx.kwayisi.org/nse — original feed (may block repeat scrapers by IP).
    """
    target = url or KWAYISI_NSE_URL
    try:
        r = requests.get(target, headers=_BROWSER_HEADERS, timeout=20)
        r.raise_for_status()
    except Exception:
        return None

    soup = BeautifulSoup(r.text, "html.parser")
    tables = soup.find_all("table")

    for table in tables:
        try:
            df = pd.read_html(io.StringIO(str(table)))[0]
        except Exception:
            continue

        df.columns = [str(c).strip().lower() for c in df.columns]
        if not ("ticker" in df.columns and "price" in df.columns and "volume" in df.columns):
            continue

        col_map: dict[str, str] = {}
        for c in df.columns:
            if c == "ticker":
                col_map[c] = "symbol"
            elif "volume" in c:
                col_map[c] = "volume"
            elif "price" in c:
                col_map[c] = "price"
            elif "change" in c:
                col_map[c] = "change"

        df = df.rename(columns=col_map)
        if "change" in df.columns:
            df["prev_price"] = df["price"] - df["change"]
            df["chg_pct"] = (df["change"] / df["prev_price"] * 100).round(2)
        else:
            df["chg_pct"] = 0.0

        df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
        result = df[["symbol", "price", "chg_pct", "volume"]].copy()
        result = result.dropna(subset=["symbol", "price"])
        if len(result) > 5:
            return _normalize(result)

    # Looser fallback parser (same as legacy data.py)
    for table in tables:
        try:
            df = pd.read_html(io.StringIO(str(table)))[0]
        except Exception:
            continue

        df.columns = [str(c).strip().lower() for c in df.columns]
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

        out = pd.DataFrame({"symbol": sym, "price": price})

        if chg_col is not None:
            chg_raw = pd.to_numeric(df[chg_col], errors="coerce")
            chg_name = str(chg_col).lower()
            if "%" in chg_name or "pct" in chg_name or "percent" in chg_name or "chgpct" in chg_name:
                out["chg_pct"] = chg_raw.round(2)
            else:
                prev_price = price - chg_raw
                out["chg_pct"] = (chg_raw / prev_price.replace(0, float("nan")) * 100).round(2)
        else:
            out["chg_pct"] = 0.0

        if vol_col is not None:
            out["volume"] = pd.to_numeric(df[vol_col], errors="coerce").fillna(0).astype(int)
        else:
            out["volume"] = 0

        out = out.dropna(subset=["symbol", "price"])
        if len(out) > 5:
            return _normalize(out)

    return None


def fetch_africanfinancials(url: str | None = None) -> pd.DataFrame | None:
    """
    Fetch from AfricanFinancials page. Attempts to read the primary price table.
    Returns DataFrame with columns: symbol, price, chg_pct, volume (volume may be 0).
    """
    target = url or AFRICANFINANCIALS_URL
    try:
        r = requests.get(target, headers=_BROWSER_HEADERS, timeout=20)
        r.raise_for_status()
    except Exception:
        return None

    # Try to parse tables via pandas first
    try:
        tables = pd.read_html(io.StringIO(r.text))
    except Exception:
        tables = []

    for table in tables:
        df = table.copy()
        # Heuristics: look for columns like 'Symbol', 'Company', 'Price', 'Last'
        cols = [str(c).strip().lower() for c in df.columns]
        if any('price' in c for c in cols) and any('symbol' in c or 'code' in c for c in cols):
            # pick likely columns
            sym_col = next((c for c in df.columns if 'symbol' in str(c).lower() or 'code' in str(c).lower() or 'ticker' in str(c).lower()), None)
            price_col = next((c for c in df.columns if 'price' in str(c).lower() or 'last' in str(c).lower()), None)
            chg_col = next((c for c in df.columns if 'change' in str(c).lower() or '%' in str(c)), None)
            vol_col = next((c for c in df.columns if 'volume' in str(c).lower() or 'vol' in str(c).lower()), None)
            if sym_col is None or price_col is None:
                continue
            out = pd.DataFrame({
                'symbol': df[sym_col].astype(str),
                'price': pd.to_numeric(df[price_col], errors='coerce'),
            })
            if chg_col is not None:
                try:
                    out['chg_pct'] = pd.to_numeric(df[chg_col], errors='coerce').fillna(0.0)
                except Exception:
                    out['chg_pct'] = 0.0
            else:
                out['chg_pct'] = 0.0
            if vol_col is not None:
                try:
                    out['volume'] = pd.to_numeric(df[vol_col], errors='coerce').fillna(0).astype(int)
                except Exception:
                    out['volume'] = 0
            else:
                out['volume'] = 0

            out = out.rename(columns={'symbol': 'symbol', 'price': 'price', 'chg_pct': 'chg_pct', 'volume': 'volume'})
            out = out.dropna(subset=['symbol', 'price'])
            if len(out) > 5:
                return _normalize(out)

    # As a last resort, attempt simple HTML scraping for price lines (very loose)
    soup = BeautifulSoup(r.text, 'html.parser')
    rows = []
    # look for table rows with two numeric columns
    for tr in soup.find_all('tr'):
        cols = [td.get_text(separator=' ').strip() for td in tr.find_all(['td','th'])]
        if len(cols) < 2:
            continue
        # heuristic: last column numeric = price, first token contains symbol-like text
        first = cols[0]
        last = cols[-1].replace(',', '')
        try:
            price = float(re.search(r'[\d\.]+', last).group(0))
        except Exception:
            continue
        sym = None
        m = re.search(r"\(([A-Z0-9]{2,12})\)", first)
        if m:
            sym = m.group(1)
        else:
            token = first.split()[0]
            if re.match(r'^[A-Z0-9]{2,12}$', token):
                sym = token
        if sym:
            rows.append({'symbol': sym, 'price': price, 'chg_pct': 0.0, 'volume': 0})
    if rows and len(rows) > 5:
        return _normalize(pd.DataFrame(rows))

    return None


_FEED_FETCHERS = {
    "mystocks_pricelist": lambda: fetch_mystocks_pricelist(),
    "kwayisi": lambda: fetch_kwayisi(),
    "africanfinancials": lambda: fetch_africanfinancials(),
}


def fetch_from_source(source: str) -> pd.DataFrame | None:
    """Run one named feed. Returns None if unknown or fetch failed."""
    fn = _FEED_FETCHERS.get(source)
    if fn is None:
        return None
    return fn()

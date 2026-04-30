"""
Label Generator — compute features and forward-return labels from the NSE master dataset.

Input
─────
  Searches in order:
    nse_master/nse_master.parquet   ← fastest, preferred
    nse_master/nse_master.json      ← single-file JSON (list of records)
    nse_master/*.json               ← one JSON per symbol

  Required columns in source data:
    symbol, date, close (or price), volume
  Optional (will be computed if absent):
    chg_pct, open, high, low

Output
──────
  training/nse_labelled.parquet  — all symbols + features + labels
  training/label_stats.json      — dataset metadata (class balance, date range)
  training/nse_candles.parquet   — canonical candle contract for learning (Phase 0)

Label definition
────────────────
  label_5d = 1  if  fwd_5d_return > ROUND_TRIP_FEE  (3.874 %)
             0  otherwise
             row dropped if forward window is incomplete (last HOLD_DAYS rows per symbol)

  The threshold equals the kill-switch floor in config/params.py so the model
  is trained to predict trades that actually clear the cost of execution.

Usage
─────
  python training/label_generator.py
  python training/label_generator.py --data-dir path/to/master --out training/nse_labelled.parquet
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── resolve project root so imports work when run from any directory ──────────
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from training.features import (
    CLOSE_COL, DATE_COL, FEATURE_COLS, HOLD_DAYS,
    LABEL_COL, ROUND_TRIP_FEE, SYMBOL_COL, VOLUME_COL,
)

# ── Feature computation helpers ───────────────────────────────────────────────

def _canonicalise_candles(df_sym: pd.DataFrame) -> pd.DataFrame:
    """
    Enforce the canonical candle contract required for learning.

    Required columns (created if missing):
        symbol, date, open, high, low, close, volume,
        atr, rolling_high_20, rolling_low_20

    ATR here is computed as a 14-period rolling mean of True Range:
        TR = max(high-low, abs(high-prev_close), abs(low-prev_close))
    """
    df_sym = df_sym.copy()

    # Ensure OHLC exist (fallback: close-only bars)
    if "open" not in df_sym.columns:
        df_sym["open"] = df_sym[CLOSE_COL]
    if "high" not in df_sym.columns:
        df_sym["high"] = df_sym[CLOSE_COL]
    if "low" not in df_sym.columns:
        df_sym["low"] = df_sym[CLOSE_COL]

    # Canonical column names
    df_sym["close"] = df_sym[CLOSE_COL]
    if "volume" not in df_sym.columns:
        df_sym["volume"] = np.nan

    # True Range and ATR (rolling mean)
    prev_close = df_sym["close"].shift(1)
    tr = pd.concat([
        (df_sym["high"] - df_sym["low"]).abs(),
        (df_sym["high"] - prev_close).abs(),
        (df_sym["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    df_sym["atr"] = tr.rolling(14).mean()

    # Rolling 20-bar highs/lows
    df_sym["rolling_high_20"] = df_sym["high"].rolling(20).max()
    df_sym["rolling_low_20"]  = df_sym["low"].rolling(20).min()

    return df_sym

def _rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Wilder-smoothed RSI (pandas ewm with com=period-1)."""
    delta = prices.diff()
    gain  = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    loss  = (-delta).clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _bb_pct(prices: pd.Series, period: int = 20, std_mult: float = 2.0) -> pd.Series:
    """Bollinger %B: position within the band (0 = lower, 1 = upper, 0.5 = mid)."""
    mid   = prices.rolling(period).mean()
    band  = std_mult * prices.rolling(period).std()
    lower = mid - band
    upper = mid + band
    width = (upper - lower).replace(0, np.nan)
    return (prices - lower) / width


def _atr_norm(close: pd.Series, period: int = 14) -> pd.Series:
    """Close-to-close ATR proxy, normalised by rolling mean price."""
    tr    = close.diff().abs()
    atr   = tr.ewm(com=period - 1, adjust=False).mean()
    return atr / close.rolling(period).mean().replace(0, np.nan)


def _compression(prices: pd.Series, period: int = 10) -> pd.Series:
    """1 − range/mean over a rolling window (higher = more compressed)."""
    r_max  = prices.rolling(period).max()
    r_min  = prices.rolling(period).min()
    r_mean = prices.rolling(period).mean().replace(0, np.nan)
    return 1 - (r_max - r_min) / r_mean


def _expansion_trigger(prices: pd.Series, volumes: pd.Series, price_w: int = 20) -> pd.Series:
    """(price_above_prev_high + vol_above_avg) / 2 — proxy for breakout probability."""
    prev_high   = prices.shift(1).rolling(price_w).max()
    avg_vol     = volumes.rolling(price_w).mean().replace(0, np.nan)
    price_break = (prices > prev_high * 1.001).astype(float)
    vol_confirm = (volumes > avg_vol * 1.3).astype(float)
    return (price_break + vol_confirm) / 2


def _vol_features(volumes: pd.Series, period: int = 20) -> pd.DataFrame:
    """vol_spike, vol_stability, rel_volume_norm."""
    avg_vol       = volumes.rolling(period).mean().replace(0, np.nan)
    vol_spike     = volumes / avg_vol
    vol_std       = volumes.rolling(period).std()
    # coefficient of variation: std/mean; invert so high stability → high score
    vol_cv        = (vol_std / avg_vol).clip(upper=1)
    vol_stability = (1 - vol_cv).clip(lower=0)
    rel_vol_norm  = np.log1p(vol_spike)
    return pd.DataFrame({
        "vol_spike":    vol_spike,
        "vol_stability": vol_stability,
        "rel_volume_norm": rel_vol_norm,
    })


def compute_features(df_sym: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all FEATURE_COLS for a single-symbol DataFrame sorted ASC by date.
    Returns the same DataFrame with feature columns appended/overwritten.
    """
    c = df_sym[CLOSE_COL].astype(float)
    v = df_sym[VOLUME_COL].astype(float) if VOLUME_COL in df_sym.columns else pd.Series(
        np.nan, index=df_sym.index
    )

    df_sym = df_sym.copy()
    df_sym["rsi_14"]       = _rsi(c)
    df_sym["ema_9"]        = c.ewm(span=9,  adjust=False).mean() / c
    df_sym["ema_21"]       = c.ewm(span=21, adjust=False).mean() / c
    df_sym["trend_strength"] = (
        c.ewm(span=9, adjust=False).mean() /
        c.ewm(span=21, adjust=False).mean().replace(0, np.nan) - 1
    )
    df_sym["bb_pct"]         = _bb_pct(c)
    df_sym["atr_14_norm"]    = _atr_norm(c)
    df_sym["compression_score"] = _compression(c)
    df_sym["expansion_trigger"] = _expansion_trigger(c, v)

    vol_feats = _vol_features(v)
    for col in ["vol_spike", "vol_stability", "rel_volume_norm"]:
        df_sym[col] = vol_feats[col].values

    # chg_pct: compute from close if not already present
    if "chg_pct" not in df_sym.columns:
        df_sym["chg_pct"] = c.pct_change() * 100

    # high_30d_pct: current price relative to 30-day high (≈ 52w position proxy)
    df_sym["high_30d_pct"] = c / c.rolling(30).max().replace(0, np.nan) - 1

    # gap_type: classify large overnight gaps
    chg = df_sym["chg_pct"].fillna(0)
    df_sym["gap_type"] = np.where(chg > 2, 1, np.where(chg < -2, -1, 0)).astype(float)

    return df_sym


def add_labels(df_sym: pd.DataFrame) -> pd.DataFrame:
    """
    Add forward return columns and binary labels (requires ASC sort).
    Drops last HOLD_DAYS rows per symbol where the forward window is incomplete.
    """
    c = df_sym[CLOSE_COL].astype(float)

    df_sym = df_sym.copy()
    df_sym["fwd_1d"]  = c.shift(-1)  / c - 1
    df_sym["fwd_5d"]  = c.shift(-HOLD_DAYS) / c - 1
    df_sym["fwd_10d"] = c.shift(-10) / c - 1

    df_sym[LABEL_COL] = (df_sym["fwd_5d"] > ROUND_TRIP_FEE).astype(float)

    # Drop rows where the forward window extends past available data
    df_sym = df_sym.iloc[:-HOLD_DAYS].copy()

    return df_sym


# ── Data loading ──────────────────────────────────────────────────────────────

def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename common variants to canonical column names."""
    df.columns = df.columns.str.lower().str.strip()
    renames = {
        "ticker": "symbol", "sym": "symbol", "code": "symbol",
        "timestamp": "date", "time": "date", "datetime": "date",
        "price":  "close",  "adj_close": "close", "adj close": "close",
        "vol":    "volume",
        "change_pct": "chg_pct", "change": "chg_pct", "pct_change": "chg_pct",
    }
    return df.rename(columns={k: v for k, v in renames.items() if k in df.columns})


def load_master(data_dir: Path) -> pd.DataFrame:
    """
    Load the NSE master dataset from data_dir.  Tries formats in order:
        1. nse_master.parquet
        2. nse_master.json   (flat list of records)
        3. *.json per symbol (one file per symbol in data_dir)
    """
    parquet = data_dir / "nse_master.parquet"
    single  = data_dir / "nse_master.json"

    if parquet.exists():
        print(f"[load] Reading {parquet}")
        df = pd.read_parquet(parquet)

    elif single.exists():
        print(f"[load] Reading {single}")
        df = pd.read_json(single)

    else:
        json_files = sorted(data_dir.glob("*.json"))
        if not json_files:
            raise FileNotFoundError(
                f"No master data found in {data_dir}.\n"
                "Expected one of:\n"
                f"  {parquet}\n  {single}\n  {data_dir}/*.json"
            )
        print(f"[load] Reading {len(json_files)} per-symbol JSON files from {data_dir}")
        frames = []
        for fp in json_files:
            try:
                chunk = pd.read_json(fp)
                chunk = _normalise_columns(chunk)
                if SYMBOL_COL not in chunk.columns:
                    chunk[SYMBOL_COL] = fp.stem.upper()
                frames.append(chunk)
            except Exception as exc:
                print(f"  ⚠  Skipping {fp.name}: {exc}")
        df = pd.concat(frames, ignore_index=True)

    df = _normalise_columns(df)

    # Validate required columns
    missing = {SYMBOL_COL, DATE_COL, CLOSE_COL} - set(df.columns)
    if missing:
        raise ValueError(f"Master data missing required columns: {missing}")

    if VOLUME_COL not in df.columns:
        print("⚠  'volume' column absent — vol features will be NaN")
        df[VOLUME_COL] = np.nan

    # Normalise date
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])

    return df


# ── Main pipeline ─────────────────────────────────────────────────────────────

def build_labelled_dataset(data_dir: Path, out_path: Path) -> pd.DataFrame:
    df_raw = load_master(data_dir)

    # Filter is_tradable flag if present
    if "is_tradable" in df_raw.columns:
        before = len(df_raw)
        df_raw = df_raw[df_raw["is_tradable"].astype(bool)]
        print(f"[filter] is_tradable: {before:,} → {len(df_raw):,} rows")

    symbols = df_raw[SYMBOL_COL].unique()
    print(f"[info] Symbols: {len(symbols)}  |  Rows: {len(df_raw):,}")

    frames: list[pd.DataFrame] = []
    candle_frames: list[pd.DataFrame] = []

    for sym in symbols:
        sym_df = (
            df_raw[df_raw[SYMBOL_COL] == sym]
            .sort_values(DATE_COL)          # ASC — required for pct_change / shift
            .reset_index(drop=True)
        )

        if len(sym_df) < 30:
            # Too short to compute 30d rolling features
            continue

        # Phase 0 contract: enforce canonical candles and required learning fields
        sym_df = _canonicalise_candles(sym_df)
        candle_frames.append(sym_df[[SYMBOL_COL, DATE_COL, "open", "high", "low", "close", "volume",
                                     "atr", "rolling_high_20", "rolling_low_20"]].copy())

        # Compute features (uses pre-computed columns if present, otherwise derives)
        existing_feats = set(FEATURE_COLS) & set(sym_df.columns)
        missing_feats  = set(FEATURE_COLS) - existing_feats

        if missing_feats:
            sym_df = compute_features(sym_df)
        else:
            # Still compute high_30d_pct / gap_type even if other features exist,
            # since they may have different names in the master data
            if "high_30d_pct" not in sym_df.columns:
                c = sym_df[CLOSE_COL].astype(float)
                sym_df["high_30d_pct"] = c / c.rolling(30).max().replace(0, np.nan) - 1
            if "gap_type" not in sym_df.columns:
                chg = sym_df.get("chg_pct", sym_df[CLOSE_COL].pct_change() * 100).fillna(0)
                sym_df["gap_type"] = np.where(chg > 2, 1, np.where(chg < -2, -1, 0)).astype(float)

        sym_df = add_labels(sym_df)
        frames.append(sym_df)

    if not frames:
        raise ValueError("No usable symbols after filtering.  Check data requirements.")

    labelled = pd.concat(frames, ignore_index=True)

    # Keep only columns we need downstream
    keep = [SYMBOL_COL, DATE_COL, CLOSE_COL, VOLUME_COL] + FEATURE_COLS + [
        LABEL_COL, "fwd_1d", "fwd_5d", "fwd_10d"
    ]
    keep = [c for c in keep if c in labelled.columns]
    labelled = labelled[keep]

    # Drop rows where ALL feature columns are NaN (cold-start warmup rows)
    feat_rows = labelled[FEATURE_COLS].dropna(how="all")
    labelled  = labelled.loc[feat_rows.index]

    print(f"[label] Total labelled rows: {len(labelled):,}")
    pos = int(labelled[LABEL_COL].sum())
    neg = len(labelled) - pos
    print(f"[label] Positive (profitable) : {pos:,}  ({pos/len(labelled)*100:.1f}%)")
    print(f"[label] Negative (unprofitable): {neg:,}  ({neg/len(labelled)*100:.1f}%)")
    print(f"[label] Date range: {labelled[DATE_COL].min().date()} → {labelled[DATE_COL].max().date()}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    labelled.to_parquet(out_path, index=False)
    print(f"[save] {out_path}")

    # Save canonical candles contract for learning/backtests
    candles = pd.concat(candle_frames, ignore_index=True) if candle_frames else pd.DataFrame()
    candles_path = out_path.with_name("nse_candles.parquet")
    if not candles.empty:
        candles.to_parquet(candles_path, index=False)
        print(f"[save] {candles_path}")

    # Write metadata
    stats = {
        "rows":          len(labelled),
        "symbols":       int(labelled[SYMBOL_COL].nunique()),
        "positive":      pos,
        "negative":      neg,
        "pos_rate":      round(pos / len(labelled), 4),
        "scale_pos_weight": round(neg / pos, 2) if pos > 0 else None,
        "date_min":      str(labelled[DATE_COL].min().date()),
        "date_max":      str(labelled[DATE_COL].max().date()),
        "fee_threshold": ROUND_TRIP_FEE,
        "hold_days":     HOLD_DAYS,
    }
    stats_path = out_path.with_suffix(".json").with_name("label_stats.json")
    stats_path.write_text(json.dumps(stats, indent=2))
    print(f"[save] {stats_path}")

    return labelled


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate forward-return labels for NSE data")
    parser.add_argument("--data-dir", type=Path, default=_ROOT / "nse_master",
                        help="Directory containing master NSE data (default: nse_master/)")
    parser.add_argument("--out", type=Path, default=_ROOT / "training" / "nse_labelled.parquet",
                        help="Output parquet path")
    args = parser.parse_args()

    if not args.data_dir.exists():
        print(f"[error] Data directory not found: {args.data_dir}")
        print("        Create nse_master/ and place your NSE historical data there.")
        sys.exit(1)

    build_labelled_dataset(args.data_dir, args.out)


if __name__ == "__main__":
    main()

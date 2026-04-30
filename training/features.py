"""
Shared feature constants for the Ziidi ML training pipeline.

Imported by label_generator, train_model, backtest, param_optimizer,
AND by core/ml_scorer so the training and inference feature sets are
guaranteed to be identical.

Design rule: if a feature cannot be computed from close+volume in the
live prices DB, it is either approximated or excluded.  This keeps
training/inference consistent without requiring OHLC data at runtime.

Feature set
───────────
    rsi_14           — 14-period RSI (close prices)
    ema_9            — 9-period EMA (close prices, normalised by close)
    ema_21           — 21-period EMA (close prices, normalised by close)
    trend_strength   — ema_9/ema_21 - 1  (positive = uptrend)
    bb_pct           — Bollinger %B (20-period, 2σ)
    atr_14_norm      — close-to-close ATR proxy / mean_price
    compression_score — 1 - 10d_range/mean  (higher = more compressed)
    expansion_trigger — (price_above_20d_high + vol_above_avg) / 2
    vol_spike        — current_vol / 20d_avg_vol
    vol_stability    — 1 - coefficient_of_variation(20d_vol)  (capped 0–1)
    rel_volume_norm  — log1p(vol_spike)
    chg_pct          — daily price change %
    high_30d_pct     — current_price / 30d_high - 1  (approximates 52w range position)
    gap_type         — -1 (gap down >2%), 0 (normal), +1 (gap up >2%)
"""

from __future__ import annotations

# Ordered list — MUST match exactly between training scripts and ml_scorer.py
FEATURE_COLS: list[str] = [
    "rsi_14",
    "ema_9",
    "ema_21",
    "trend_strength",
    "bb_pct",
    "atr_14_norm",
    "compression_score",
    "expansion_trigger",
    "vol_spike",
    "vol_stability",
    "rel_volume_norm",
    "chg_pct",
    "high_30d_pct",
    "gap_type",
]

LABEL_COL        = "label_5d"
DATE_COL         = "date"
SYMBOL_COL       = "symbol"
CLOSE_COL        = "close"   # or "price" — loader normalises
VOLUME_COL       = "volume"
CHG_COL          = "chg_pct"

# Chronological train/val/test split boundaries
TRAIN_END        = "2020-12-31"
VAL_START        = "2021-01-01"
VAL_END          = "2022-12-31"
TEST_START       = "2023-01-01"

# Must match config/params.py:ROUND_TRIP_FEE_PCT (kept local to avoid live dep)
ROUND_TRIP_FEE   = 0.03874     # 3.874 % NSE round-trip cost on trades < 100 k KES
HOLD_DAYS        = 5           # forward window for label and backtest exit

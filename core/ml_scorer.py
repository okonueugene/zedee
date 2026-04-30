"""
ML Scorer — probabilistic drop-in for detect_phase() + score_signal().

Module-level behaviour
──────────────────────
At import time this module attempts to load training/ziidi_signal_model.pkl.

    MODEL_LOADED = True   → model found; score_from_history() returns (score, phase)
    MODEL_LOADED = False  → model absent; score_from_history() returns None

scan_for_entries() in main.py checks MODEL_LOADED once.  If False it falls back
to the rule-based detect_phase() + fuse_score() path completely unchanged.

Integration contract
────────────────────
    result = score_from_history(history_df)
    if result is not None:
        base_score, phase = result   # (float 0–100, phase hint string)
    else:
        # rule-based path

score_from_history() returns (ml_score_0_100, phase_hint) where:
  • ml_score_0_100 — P(profitable) × 100
  • phase_hint — a phase string derived from ML features for use by the event
                 boost system (apply_event_boost needs a phase to pick its multiplier)

Feature computation
───────────────────
The live prices DB stores close + volume only (ORDER BY timestamp DESC).
We compute close-only versions of all FEATURE_COLS and set NaN for anything
that strictly requires OHLC.  XGBoost handles NaN natively — missing features
degrade accuracy rather than causing errors.

Feature alignment is guaranteed because this module imports FEATURE_COLS from
training/features.py, the same constant used at training time.

Thread safety
─────────────
The model object is loaded once at import time and is read-only thereafter.
Concurrent calls to score_from_history() are safe.
"""

from __future__ import annotations

import logging
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_logger = logging.getLogger(__name__)

# ── Path resolution ───────────────────────────────────────────────────────────
_ROOT       = Path(__file__).resolve().parent.parent
_MODEL_PATH = _ROOT / "training" / "ziidi_signal_model.pkl"

# ── Import shared feature constants ──────────────────────────────────────────
try:
    _TRAINING_DIR = str(_ROOT)
    if _TRAINING_DIR not in sys.path:
        sys.path.insert(0, _TRAINING_DIR)
    from training.features import FEATURE_COLS
except ImportError:
    # Fallback: duplicate the list so the scorer works even if training/ is absent.
    # This list MUST be identical to training/features.py:FEATURE_COLS.
    FEATURE_COLS = [
        "rsi_14", "ema_9", "ema_21", "trend_strength",
        "bb_pct", "atr_14_norm", "compression_score", "expansion_trigger",
        "vol_spike", "vol_stability", "rel_volume_norm",
        "chg_pct", "high_30d_pct", "gap_type",
    ]

# ── Model loading (once at import) ────────────────────────────────────────────
MODEL_LOADED: bool = False
_model = None

try:
    if _MODEL_PATH.exists():
        with open(_MODEL_PATH, "rb") as _f:
            _model = pickle.load(_f)
        MODEL_LOADED = True
        _logger.info("ml_scorer: model loaded from %s", _MODEL_PATH)
    else:
        _logger.info(
            "ml_scorer: model not found at %s — rule-based fallback active",
            _MODEL_PATH,
        )
except Exception as _exc:
    _logger.warning("ml_scorer: failed to load model (%s) — rule-based fallback active", _exc)

# ── Phase hint thresholds ─────────────────────────────────────────────────────
# Derived purely from ML feature values so the event boost can use a phase label.
_EXPANSION_TRIGGER_THRESHOLD   = 0.5
_COMPRESSION_SCORE_THRESHOLD   = 0.6
_TREND_STRENGTH_UPTREND        = 0.002  # ema9/ema21 - 1 > 0.2 % → uptrend


# ── Feature computation (from DESC-ordered history DataFrame) ─────────────────

def _safe_float(val) -> float:
    """Return float or NaN on any exception."""
    try:
        v = float(val)
        return v if math.isfinite(v) else math.nan
    except Exception:
        return math.nan


def _rsi(prices: pd.Series, period: int = 14) -> float:
    """Wilder-smoothed RSI from a price series in chronological (ASC) order."""
    if len(prices) < period + 1:
        return math.nan
    delta = prices.diff().dropna()
    gain  = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    loss  = (-delta).clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    last_loss = loss.iloc[-1]
    if last_loss == 0:
        return 100.0
    rs = gain.iloc[-1] / last_loss
    return _safe_float(100 - 100 / (1 + rs))


def _ema_last(prices: pd.Series, span: int) -> float:
    """Last value of EMA(span) from a chronological price series."""
    if prices.empty:
        return math.nan
    return _safe_float(prices.ewm(span=span, adjust=False).mean().iloc[-1])


def _bb_pct(prices: pd.Series, period: int = 20, std_mult: float = 2.0) -> float:
    """Bollinger %B for the most recent bar."""
    if len(prices) < period:
        return math.nan
    mid   = prices.rolling(period).mean().iloc[-1]
    band  = std_mult * prices.rolling(period).std().iloc[-1]
    if band == 0:
        return math.nan
    lower = mid - band
    upper = mid + band
    return _safe_float((prices.iloc[-1] - lower) / (upper - lower))


def _atr_norm(prices: pd.Series, period: int = 14) -> float:
    """Close-to-close ATR proxy / mean_price (last bar)."""
    if len(prices) < period + 1:
        return math.nan
    tr    = prices.diff().abs()
    atr   = tr.ewm(com=period - 1, adjust=False).mean().iloc[-1]
    mean  = prices.mean()
    if mean == 0:
        return math.nan
    return _safe_float(atr / mean)


def _compression_score(prices: pd.Series, period: int = 10) -> float:
    """1 - range/mean over last `period` bars.  Higher = more compressed."""
    if len(prices) < period:
        return math.nan
    recent = prices.iloc[-period:]
    mean   = recent.mean()
    if mean == 0:
        return math.nan
    range_pct = (recent.max() - recent.min()) / mean
    return _safe_float(1 - min(range_pct, 1))


def _expansion_trigger(prices: pd.Series, volumes: pd.Series, period: int = 20) -> float:
    """(price_above_period_high + vol_above_avg) / 2."""
    if len(prices) < period + 1:
        return math.nan
    current_price = prices.iloc[-1]
    prev_high     = prices.iloc[-period - 1:-1].max()   # period bars before current
    price_break   = 1.0 if current_price > prev_high * 1.001 else 0.0

    if volumes.isna().all() or volumes.mean() == 0:
        return _safe_float(price_break / 2)             # vol data absent → partial score

    avg_vol    = volumes.iloc[-period:].mean()
    current_vol = volumes.iloc[-1]
    vol_confirm = 1.0 if (avg_vol > 0 and current_vol > avg_vol * 1.3) else 0.0

    return _safe_float((price_break + vol_confirm) / 2)


def _vol_features(volumes: pd.Series, period: int = 20) -> tuple[float, float, float]:
    """Returns (vol_spike, vol_stability, rel_volume_norm)."""
    if volumes.isna().all():
        return math.nan, math.nan, math.nan

    valid = volumes.dropna()
    if len(valid) < 2 or valid.mean() == 0:
        return math.nan, math.nan, math.nan

    recent_vol  = valid.iloc[-period:] if len(valid) >= period else valid
    avg_vol     = recent_vol.mean()
    if avg_vol == 0:
        return math.nan, math.nan, math.nan

    current_vol  = _safe_float(valid.iloc[-1])
    vol_spike    = _safe_float(current_vol / avg_vol)
    vol_cv       = _safe_float(recent_vol.std() / avg_vol)
    vol_stability = _safe_float(max(0.0, 1 - min(vol_cv, 1)))
    rel_vol_norm  = _safe_float(np.log1p(vol_spike) if not math.isnan(vol_spike) else math.nan)

    return vol_spike, vol_stability, rel_vol_norm


def _build_features(history: pd.DataFrame) -> dict[str, float]:
    """
    Compute all FEATURE_COLS from the history DataFrame.

    history is ORDER BY timestamp DESC (newest row first, as returned by get_history()).
    We reverse to chronological order before all rolling/ewm computations.

    Returns a dict {feature_name: float_or_nan}.
    """
    # Reverse to chronological (oldest → newest) for rolling/ewm
    prices  = history["price"].astype(float).iloc[::-1].reset_index(drop=True)
    volumes = (
        history["volume"].astype(float).iloc[::-1].reset_index(drop=True)
        if "volume" in history.columns
        else pd.Series(dtype=float)
    )
    chg_last = (
        _safe_float(history["chg_pct"].iloc[0])   # iloc[0] = most recent (DESC)
        if "chg_pct" in history.columns
        else _safe_float((prices.iloc[-1] / prices.iloc[-2] - 1) * 100 if len(prices) >= 2 else math.nan)
    )

    ema9  = _ema_last(prices, 9)
    ema21 = _ema_last(prices, 21)

    trend = math.nan
    if not math.isnan(ema9) and not math.isnan(ema21) and ema21 != 0:
        trend = _safe_float(ema9 / ema21 - 1)

    # Normalise EMAs by current price for scale-invariance across different stocks
    current_px = prices.iloc[-1]
    ema9_n  = _safe_float(ema9  / current_px) if not math.isnan(ema9)  and current_px else math.nan
    ema21_n = _safe_float(ema21 / current_px) if not math.isnan(ema21) and current_px else math.nan

    # 30-day high position (approximates 52w percentile with available data)
    high_30d = prices.max() if len(prices) >= 1 else math.nan
    high_30d_pct = _safe_float(current_px / high_30d - 1) if high_30d else math.nan

    # gap_type from chg_pct
    gap_type = 0.0
    if not math.isnan(chg_last):
        gap_type = 1.0 if chg_last > 2.0 else (-1.0 if chg_last < -2.0 else 0.0)

    vol_spike, vol_stability, rel_vol_norm = _vol_features(volumes)

    feats = {
        "rsi_14":             _rsi(prices),
        "ema_9":              ema9_n,
        "ema_21":             ema21_n,
        "trend_strength":     trend,
        "bb_pct":             _bb_pct(prices),
        "atr_14_norm":        _atr_norm(prices),
        "compression_score":  _compression_score(prices),
        "expansion_trigger":  _expansion_trigger(prices, volumes),
        "vol_spike":          vol_spike,
        "vol_stability":      vol_stability,
        "rel_volume_norm":    rel_vol_norm,
        "chg_pct":            chg_last,
        "high_30d_pct":       high_30d_pct,
        "gap_type":           gap_type,
    }

    return feats


def _infer_phase(feats: dict[str, float]) -> str:
    """
    Derive a phase string from the feature vector for use by apply_event_boost().

    This is not the phase produced by detect_phase() — it is a coarse label so
    the event engine can apply the correct phase multiplier (e.g. COMPRESSION
    events get a higher boost than EXPANSION events).
    """
    exp_trig   = feats.get("expansion_trigger", math.nan)
    comp_score = feats.get("compression_score", math.nan)
    trend      = feats.get("trend_strength",    math.nan)

    if not math.isnan(exp_trig) and exp_trig >= _EXPANSION_TRIGGER_THRESHOLD:
        return "EXPANSION"
    if not math.isnan(comp_score) and comp_score >= _COMPRESSION_SCORE_THRESHOLD:
        return "COMPRESSION"
    if not math.isnan(trend) and trend > _TREND_STRENGTH_UPTREND:
        return "EXHAUSTION"    # mild uptrend without breakout or compression
    return "NEUTRAL"


# ── Public API ────────────────────────────────────────────────────────────────

def score_from_history(history: pd.DataFrame) -> tuple[float, str] | None:
    """
    Compute ML score from a live history DataFrame.

    Args:
        history — DataFrame from get_history(sym), ORDER BY timestamp DESC.
                  Must contain at least 'price' and ideally 'volume', 'chg_pct'.

        Returns:
        (ml_score_0_100, phase_hint)  — if model is loaded and features can be built
        None                          — if model is not loaded or feature build fails

    ml_score_0_100 maps directly to the existing score gate:
        if ml_score < params['score_threshold']: reject

    phase_hint is passed to apply_event_boost() to select the phase multiplier.
    """
    if _model is None:
        return None

    if history.empty or "price" not in history.columns:
        return None

    # Need at least 22 bars for EMA-21 warmup
    if len(history) < 22:
        return None

    try:
        feats = _build_features(history)
    except Exception as exc:
        _logger.warning("ml_scorer._build_features failed: %s", exc)
        return None

    # Build feature row in exact FEATURE_COLS order (NaN for any missing)
    feat_row = [feats.get(col, math.nan) for col in FEATURE_COLS]

    # Check if too many features are NaN — model will likely produce noise
    nan_count = sum(1 for v in feat_row if math.isnan(v))
    if nan_count > len(FEATURE_COLS) // 2:
        _logger.debug("ml_scorer: >50%% features NaN (%d/%d) — skipping", nan_count, len(FEATURE_COLS))
        return None

    try:
        X = pd.DataFrame([feat_row], columns=FEATURE_COLS)
        prob = float(_model.predict_proba(X)[0][1])
    except Exception as exc:
        _logger.warning("ml_scorer.predict failed: %s", exc)
        return None

    ml_score   = round(prob * 100, 1)
    phase_hint = _infer_phase(feats)

    return ml_score, phase_hint


def reload_model() -> bool:
    """
    Hot-reload the model file without restarting the process.

    Returns True if model loaded successfully, False otherwise.
    Useful after running the training pipeline with the system running.
    """
    global _model, MODEL_LOADED
    try:
        if not _MODEL_PATH.exists():
            _logger.warning("ml_scorer.reload_model: file not found at %s", _MODEL_PATH)
            return False
        with open(_MODEL_PATH, "rb") as f:
            _model = pickle.load(f)
        MODEL_LOADED = True
        _logger.info("ml_scorer: model hot-reloaded from %s", _MODEL_PATH)
        return True
    except Exception as exc:
        _logger.warning("ml_scorer.reload_model failed: %s", exc)
        return False

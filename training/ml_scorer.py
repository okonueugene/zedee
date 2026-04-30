"""
core/ml_scorer.py
─────────────────────────────────────────────────────────────────
Drop-in ML scoring module for the live Ziidi signal engine.

Replaces detect_phase() + score_signal() in scan_for_entries().
The 8-gate architecture is unchanged — this module only affects
Gate 3 (phase) and Gate 7 (score).

All other gates (time, history, volume, breakout, cooldown, risk,
net-target) remain rule-based and are unaffected.

Usage in scan_for_entries():
    from core.ml_scorer import MLScorer, FALLBACK_SCORE

    # Initialise once at startup (model load is expensive)
    scorer = MLScorer()

    # Per symbol in scan loop:
    score, phase, confidence = scorer.score(history_df, current_price, current_chg)

    if score < params['score_threshold']:
        reject(sym, 'SCORE', score=score)
        continue
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

ROOT        = Path(__file__).parent.parent
MODEL_FILE  = ROOT / "models" / "ziidi_signal_model.pkl"

# If the model is not available, fall back to rule-based scoring
# The live system continues without interruption during training.
FALLBACK_SCORE = None   # None = use rule-based detect_phase + score_signal

# Feature columns expected by the model (from training)
# These must match feature_cols saved in the model bundle.
_DIRECTION_MAP = {"bullish": 1, "bearish": -1, "neutral": 0, "doji": 0}


class MLScorer:
    """
    Loads the trained XGBoost model at startup and provides per-symbol scoring.

    Thread-safe for read (predict_proba is stateless after load).
    """

    def __init__(self, model_path: Path = MODEL_FILE):
        self.model        = None
        self.feature_cols = []
        self.encoders     = {}
        self.available    = False
        self._load(model_path)

    def _load(self, path: Path) -> None:
        if not path.exists():
            print(f"[MLScorer] Model not found at {path}. Using rule-based fallback.")
            return
        try:
            with open(path, "rb") as f:
                bundle = pickle.load(f)
            self.model        = bundle["model"]
            self.feature_cols = bundle["feature_cols"]
            self.encoders     = bundle.get("encoders", {})
            self.available    = True
            print(f"[MLScorer] Loaded: {path.name} | {len(self.feature_cols)} features")
        except Exception as e:
            print(f"[MLScorer] Load failed: {e}. Using rule-based fallback.")

    def _build_feature_row(
        self,
        history: pd.DataFrame,
        current_price: float,
        current_chg: float,
        symbol: str = "",
        sector: str = "",
    ) -> pd.DataFrame | None:
        """
        Build a single-row feature DataFrame from live scan data.

        history: DataFrame from get_history(sym) — most recent first (DESC order).
        current_price / current_chg: from the live fetch, not from history.

        LEAKAGE GUARD: we use history.iloc[1:] for rolling computations —
        this excludes the current live bar (which may be mid-session) and
        uses only completed prior sessions.

        Returns None if insufficient data to compute features.
        """
        if history.empty or len(history) < 15:
            return None

        # Work on completed sessions only (exclude current live bar)
        hist = history.sort_values("timestamp", ascending=True).copy()

        row = {}

        # Price action (from live bar)
        if "range_pct" in hist.columns:
            row["range_pct"]    = hist["range_pct"].iloc[-1]
        if "body_pct" in hist.columns:
            row["body_pct"]     = hist["body_pct"].iloc[-1]
        if "gap" in hist.columns:
            row["gap"]          = hist["gap"].iloc[-1]
        if "gap_type" in hist.columns:
            row["gap_type"]     = hist["gap_type"].iloc[-1]
        if "true_range" in hist.columns:
            row["true_range"]   = hist["true_range"].iloc[-1]
        if "atr_14_norm" in hist.columns:
            row["atr_14_norm"]  = hist["atr_14_norm"].iloc[-1]
        if "volatility_5" in hist.columns:
            row["volatility_5"] = hist["volatility_5"].iloc[-1]

        # Volume features
        for col in ["vol_ratio","vol_spike","vol_stability","rel_volume","rel_volume_norm"]:
            if col in hist.columns:
                row[col] = hist[col].iloc[-1]

        # Trend
        for col in ["trend_strength","return_20d","rsi_14"]:
            if col in hist.columns:
                row[col] = hist[col].iloc[-1]

        # Bollinger
        for col in ["bb_width","bb_std"]:
            if col in hist.columns:
                row[col] = hist[col].iloc[-1]

        # Breakout / compression (v4 features)
        for col in ["distance_to_high","breakout_strength","range_compression",
                    "expansion_trigger","compression_score","high_52w_pct"]:
            if col in hist.columns:
                row[col] = hist[col].iloc[-1]

        # Categoricals
        direction_last = hist["direction"].iloc[-1] if "direction" in hist.columns else "neutral"
        row["direction"] = _DIRECTION_MAP.get(str(direction_last).lower(), 0)

        sec_map = self.encoders.get("sector", {})
        row["sector_encoded"] = sec_map.get(sector, -1)

        sym_map = self.encoders.get("symbol", {})
        row["symbol_encoded"] = sym_map.get(symbol, -1)

        # Fill any remaining features with 0 (handles optional features)
        for col in self.feature_cols:
            if col not in row:
                row[col] = 0.0

        return pd.DataFrame([row])

    def score(
        self,
        history: pd.DataFrame,
        current_price: float,
        current_chg: float,
        symbol: str = "",
        sector: str = "",
    ) -> tuple[float, str, float]:
        """
        Score a trading candidate.

        Returns:
            (ml_score: float 0-100, phase_label: str, confidence: float 0-1)

        ml_score maps directly to the existing score_threshold gate.
        phase_label is a human-readable interpretation of the score.
        confidence is the raw model probability (same as ml_score / 100).
        """
        if not self.available:
            return FALLBACK_SCORE, "UNKNOWN", 0.0

        features = self._build_feature_row(
            history, current_price, current_chg, symbol, sector
        )
        if features is None:
            return 0.0, "INSUFFICIENT_DATA", 0.0

        try:
            proba = float(self.model.predict_proba(features.fillna(0))[0][1])
        except Exception as e:
            return 0.0, "SCORE_ERROR", 0.0

        ml_score = round(proba * 100, 1)

        # Map probability to a phase label for logging / human readability
        if proba >= 0.75:
            phase = "EXPANSION"
        elif proba >= 0.55:
            phase = "COMPRESSION"
        elif proba >= 0.40:
            phase = "EXHAUSTION"
        elif proba < 0.20 and current_chg < -1.5:
            phase = "CAPITULATION"
        else:
            phase = "NEUTRAL"

        return ml_score, phase, proba

    def is_available(self) -> bool:
        return self.available


# ── Integration shim for scan_for_entries() ───────────────────────────────────
# Drop-in replacement usage in main.py / scan_for_entries():
#
#   from core.ml_scorer import MLScorer
#   from core.signals import detect_phase, score_signal   # keep as fallback
#
#   _ml_scorer = MLScorer()   # load once at module level
#
#   # Inside scan_for_entries() per-symbol loop:
#   if _ml_scorer.is_available():
#       score, phase, confidence = _ml_scorer.score(
#           history, price, chg, sym, sector
#       )
#   else:
#       phase = detect_phase(history, price, chg, params)
#       score = score_signal(phase, chg)
#       confidence = score / 100
#
#   log_event("PHASE", {"symbol": sym, "phase": phase, "score": score,
#                        "source": "ML" if _ml_scorer.is_available() else "RULES"})
#
#   if score < params["entry"]["score_threshold"]:
#       reject(sym, "SCORE", score=score, source="ML")
#       continue

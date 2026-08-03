"""
Central configuration for the Ziidi Signal Engine.
All constants, file paths, and calibration defaults live here.
Import from this module; never hard-code values in other files.
"""

import json
import os

# Load local .env (kept out of git) for secrets/config.
# If python-dotenv isn't installed, proceed without it.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


# ── Telegram notifications ────────────────────────────────────────────────────
# Set in .env (see .env.example). Names match core/notifications.py:
#   TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
# TELEGRAM_TOKEN is accepted as an alias for the bot token.
#
# How to get them:
#   1. Open Telegram → search @BotFather → /newbot → copy the token
#   2. Send any message to your bot, then visit:
#      https://api.telegram.org/bot<TOKEN>/getUpdates
#      and copy the "id" value inside "chat"
#   3. Run:  python -c "from core.notifications import test_telegram; test_telegram()"
#
TELEGRAM_TOKEN = (
    os.getenv("TELEGRAM_BOT_TOKEN")
    or os.getenv("TELEGRAM_TOKEN")
    or ""
)
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID") or ""
NOTIFY_INFO = os.getenv("NOTIFY_INFO", "1").lower() in ("1", "true", "yes")

# ── Market & execution constants ──────────────────────────────────────────────
INITIAL_CAPITAL    = 10_000.0
PAPER_TRADE        = True
WATCHLIST          = ['SCOM', 'EQTY', 'KCB', 'NCBA', 'EABL', 'COOP', 'KQ' , 'ABSA', 'STANCHART', 'IMH', 'KPLC', 'KEGN']
SCAN_INTERVAL_MIN  = 15
SLIPPAGE_BUFFER    = 1.005   # 0.5 % buffer for limit-order realism

# ── NSE live data sources (fallback chain) ───────────────────────────────────
# Prioritized sources (production-ready):
# 1. mystocks_pricelist (primary) — clean table, good volume; occasional outages
# 2. kwayisi (legacy) — stable long history; HTML structure may vary
# 3. africanfinancials — alternative coverage; may lag a few minutes
NSE_FEED_PRIORITY = ["mystocks_pricelist", "kwayisi", "africanfinancials"]
KWAYISI_NSE_URL = "https://afx.kwayisi.org/nse/"
MYSTOCKS_PRICELIST_URL = "https://live.mystocks.co.ke/m/pricelist"
AFRICANFINANCIALS_URL = "https://africanfinancials.com/nairobi-securities-exchange-kenya-share-prices/"

# ── Watchlist Expansion (added 2026-04-24) ──────────────────────────────────
#
# Extended watchlist — all symbols that the expansion pipeline will consider.
# The pipeline will further filter by ADVT and activation; not all will score.
WATCHLIST_EXTENDED = [
    # Tier 1 — core anchors (always included if history exists)
    "SCOM", "EQTY", "KCB", "COOP", "NCBA", "EABL",
    # Tier 2 — validated additions (must pass ADVT filter live)
    "ABSA", "IMH",
    # Tier 3 — conditional (must pass ADVT filter live; drop if they fail)
    # DTBK removed until sufficient price history accumulates
    "KPLC", "KEGN",
    # Exclusions (do NOT add): BAT, JUB, BRITAM
]

# Liquidity filter — minimum 20-session average daily value traded (KES)
# Set at KES 3M based on NSE reality (only EQTY/KCB clear KES 20M reliably)
MIN_ADVT_KES = 3_000_000

# Activation filter — volume spike threshold (ratio vs 20-session average)
# Tuned on 2026-04-24 offline validation to keep activated symbols ~3–6/session.
ACTIVATION_VOLUME_SPIKE = 1.5       # 150% of projected daily volume = activated

# Activation filter — minimum participation (avoids zero-volume drift days)
# Retuned with intraday volume projection; target ~3–6 activated symbols/session.
ACTIVATION_MIN_PARTICIPATION = 0.85  # projected volume >= 85% of avg

# Activation filter — proximity to 5-day high (structure check)
ACTIVATION_NEAR_HIGH_5D = 0.97      # within 3% of 5-day high = activated

# Minimum history bars required before enrichment is attempted
ENRICHMENT_MIN_BARS = 10

# NSE round-trip cost on trades < 100 k KES
# (CDS levy 1.5 % each way + broker + CDSC ≈ 3.874 % total)
# Expectancy must exceed this for the system to have positive edge after costs.
ROUND_TRIP_FEE_PCT = 0.03874

# ── File paths ─────────────────────────────────────────────────────────────────
DB_FILE        = "ziidi_positions.db"
PRICE_DB       = "ziidi_prices.db"
LOG_FILE       = "ziidi_alert_log.jsonl"
EQUITY_FILE    = "ziidi_equity.csv"
TRADES_FILE    = "ziidi_trades.csv"
PARAMS_FILE    = "system_params.json"
BREAKOUTS_FILE = "ziidi_breakouts.json"   # persisted breakout state (survives restarts)

# ── Calibration defaults ───────────────────────────────────────────────────────
DEFAULTS: dict = {
    # Entry filters
    "min_net_target_pct":     0.045,  # minimum net return to accept a trade (bootstrap mode)
    "compression_range_pct":  0.025,  # price range width that defines compression (wider for NSE spreads)

    # Risk limits
    "max_risk_pct":           0.02,   # max equity risked per position
    "max_total_risk_pct":     0.04,   # max total portfolio risk (sum of open positions)

    # Exit parameters
    "trailing_stop_pct":      0.07,   # trailing stop distance from highest price (legacy path)
    "stop_floor_pct":         0.04,   # minimum stop distance as % of price
    "max_hold_days":          10,     # MVP TIME_STOP: sessions before time-decay exit can apply

    # Operational
    "cooldown_hours":         2,      # fallback flat cooldown (overridden by cooldown_by_reason)
    "min_trades_to_calibrate": 20,    # journal must have this many trades before calibration runs

    # Scan scheduling (EAT)
    # Avoid burning the fetch failure counter at the open when the page is often empty.
    # Entry window still starts at 10:30 — this just delays the *data fetch loop*.
    "scan_start_h":           10,
    "scan_start_m":           0,
    "fetch_recovery_minutes": 60,     # after fetch-failure pause, retry within the session

    # MVP frozen engine (modules 1–5): when True, entry/exit follow core/mvp_engine.py only
    "use_mvp_engine":         False,

    # Entry decision chain
    "score_threshold":        55,     # MVP/ML: strict > this; ~55 avoids calibration starvation vs ~60 ceiling
    "min_history_rows":       15,     # minimum price bars required before phase detection runs
    "vol_mult_morning":       1.2,    # current vol must be >= avg * this before noon (calibratable)
    "vol_mult_afternoon":     1.1,    # current vol must be >= avg * this after noon  (calibratable)
    "breakout_expiry_sessions": 3,    # expire tracked breakouts after this many calendar days

    # Time gate (entry window, EAT)
    "entry_open_h":   10,
    "entry_open_m":   30,
    "entry_close_h":  14,
    "entry_close_m":  45,

    # Debug: temporarily bypass strict entry window gating.
    # This is NOT trading-logic, only scan-time gating.
    "ALLOW_EARLY_SESSION": False,
    "ALLOW_LATE_SESSION": False,

    # Debug visibility
    "DEBUG_LOG_SCAN_RESULT": True,

    # Debug: force-create one open trade after data passes all scan gates.
    # This is to validate journal + position lifecycle + alerts wiring.
    "FORCE_TEST_ENTRY": False,

    # Reason-based cooldowns (hours) — calibratable per exit type
    "cooldown_by_reason": {
        "TRAILING_STOP":    6,   # stop-out; needs time to re-establish trend
        "TARGET_HIT":       2,   # clean exit; can re-enter sooner
        "MAX_HOLD_EXPIRED": 4,   # timed-out; trend may be exhausted
        "STOP":             6,   # MVP monotonic stop
        "TIME_STOP":        4,   # MVP time decay exit
        "STOP_LOSS":        6,   # legacy explicit stop
        "MANUAL":           2,
        "NONE":             0,   # no prior trade; no cooldown
    },
}


def load_params() -> dict:
    """Load params from disk, merging with defaults to fill any missing keys."""
    if os.path.exists(PARAMS_FILE):
        with open(PARAMS_FILE) as f:
            stored = json.load(f)
        return {**DEFAULTS, **stored}
    p = DEFAULTS.copy()
    save_params(p)
    return p


def save_params(p: dict) -> None:
    with open(PARAMS_FILE, "w") as f:
        json.dump(p, f, indent=2)

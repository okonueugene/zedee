"""
Notification Dispatch — severity-based routing for all system events.

Severity levels
───────────────
CRITICAL  [!]  Kill switches, system pauses, account-safety events.
               Always dispatched to Telegram. Require immediate human attention.

WARNING   [~]  Data pipeline failures, degraded performance, trade alerts.
               Always dispatched to Telegram. Action may be required.

INFO      [i]  Health pings, event detections, calibration updates, trade closes.
               Dispatched only when NOTIFY_INFO=True in config/params.py
               (or NOTIFY_INFO=1 env var). Default: log only.

Credential lookup order (most → least preferred)
─────────────────────────────────────────────────
1. Environment variables  TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID
2. config/params.py       TELEGRAM_TOKEN      / TELEGRAM_CHAT_ID
   (edit the file directly — no restart required for env vars)

Delivery failures are written directly to the JSONL log so they are
visible in analytics without creating a notification→log→notification loop.
"""

import json
import logging
import os
import socket
from datetime import datetime

from dateutil import tz

_logger = logging.getLogger(__name__)

EAT = tz.gettz('Africa/Nairobi')

# ── Severity constants ─────────────────────────────────────────────────────────
CRITICAL = "CRITICAL"
WARNING  = "WARNING"
INFO     = "INFO"

# ── Event → severity map ───────────────────────────────────────────────────────
SEVERITY: dict[str, str] = {

    # ── CRITICAL — account safety, stop everything ────────────────────────────
    "KILL_SWITCH":       CRITICAL,
    "SYSTEM_PAUSED":     CRITICAL,
    "DRAWDOWN_KILL":     CRITICAL,
    "EXPECTANCY_KILL":   CRITICAL,

    # ── WARNING — degraded / action required ──────────────────────────────────
    "FETCH_ERROR":       WARNING,
    "FETCH_NO_TABLE":    WARNING,
    "CALIBRATION_ERROR": WARNING,
    "DRAWDOWN_MODE":     WARNING,
    "ZIIDI_BUY_ALERT":   WARNING,
    "ZIIDI_SELL_ALERT":  WARNING,
    "DAILY_SUMMARY":     WARNING,

    # ── INFO — informational, no immediate action ─────────────────────────────
    "HEALTH_PING":       INFO,
    "EVENT_DETECTED":    INFO,
    "CALIBRATION":       INFO,
    "PARAM_UPDATE":      INFO,
    "SYSTEM_RESUMED":    INFO,
    "TRADE_CLOSED":      INFO,
    "FETCH_OK":          INFO,
    "SYSTEM_START":      WARNING,
    "SCHEDULE":          INFO,
    "EXCURSION_SIGNAL":  INFO,
    "REMINDER":          INFO,
    "RISK_CAP":          INFO,
    "EQUITY_LOAD_ERROR": INFO,
}

_LABEL: dict[str, str] = {
    CRITICAL: "[CRITICAL]",
    WARNING:  "[WARNING] ",
    INFO:     "[INFO]    ",
}


# ── Public helpers ─────────────────────────────────────────────────────────────

def get_severity(event_type: str) -> str:
    """Return the severity level string for a given event type."""
    return SEVERITY.get(event_type, INFO)


def test_telegram() -> bool:
    """
    Send a one-off test message to verify your Telegram credentials work.

    Usage (run once after setting your token and chat_id):
        python -c "from core.notifications import test_telegram; test_telegram()"

    Returns True when the API accepted the message, False on any failure.
    """
    ts  = datetime.now(EAT).strftime("%H:%M:%S")
    msg = (
        "<b>[TEST] Ziidi Signal Engine</b>\n"
        f"{ts} EAT\n\n"
        "Telegram notifications are working.\n"
        "You will receive CRITICAL and WARNING alerts here."
    )
    ok = _send_telegram(msg)
    status = "OK — message delivered" if ok else "FAILED — check TELEGRAM_TOKEN and TELEGRAM_CHAT_ID"
    print(f"Telegram test: {status}")
    return ok


# ── Credential resolution ──────────────────────────────────────────────────────

def _get_credentials() -> tuple[str, str]:
    """
    Return (token, chat_id).  Env vars beat params.py so CI/CD can override
    without touching source files.
    """
    # env vars — highest priority
    token   = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")

    # fall back to params.py constants if env vars absent
    if not token or not chat_id:
        try:
            from config import params as _p
            token   = token   or getattr(_p, "TELEGRAM_TOKEN",   "")
            chat_id = chat_id or getattr(_p, "TELEGRAM_CHAT_ID", "")
        except Exception:
            pass

    return token, chat_id


def _notify_info_enabled() -> bool:
    if os.environ.get("NOTIFY_INFO", "0") == "1":
        return True
    try:
        from config import params as _p
        return bool(getattr(_p, "NOTIFY_INFO", False))
    except Exception:
        return False


# ── Internal transport ─────────────────────────────────────────────────────────

def _send_telegram(message: str) -> bool:
    """
    POST to Telegram Bot API.  Returns True on HTTP 200, False otherwise.
    Delivery failures are appended to the JSONL log directly (not via
    log_event) to avoid a notification → log → notification loop.
    """
    token, chat_id = _get_credentials()
    if not token or not chat_id:
        return False

    try:
        import requests as _req
        last_err = None
        # Small retry for transient network timeouts.
        for attempt in (1, 2):
            try:
                r = _req.post(
                    f"https://api.telegram.org/bot{token}/sendMessage",
                    json={"chat_id": chat_id, "text": message, "parse_mode": "HTML"},
                    timeout=15,
                )
                if r.status_code == 200:
                    return True
                last_err = f"HTTP {r.status_code}: {r.text[:300]}"
            except Exception as exc:
                last_err = str(exc)
        _log_send_failure(str(last_err))
        return False
    except Exception as exc:
        _log_send_failure(str(exc))
        return False


def _log_send_failure(reason: str) -> None:
    """Append a NOTIFY_FAIL entry to the JSONL log without going through log_event."""
    try:
        from config.params import LOG_FILE
        entry = {
            "timestamp": datetime.now(EAT).isoformat(),
            "event":     "NOTIFY_FAIL",
            "severity":  WARNING,
            "reason":    reason,
        }
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass


# ── Message formatters ─────────────────────────────────────────────────────────

def _fmt_trade_alert(event_type: str, data: dict) -> str:
    ts     = datetime.now(EAT).strftime("%H:%M:%S")
    action = "BUY" if "BUY" in event_type else "SELL"
    sym    = data.get("symbol", "?")
    lines  = [f"<b>[{action}] {sym}  |  {ts} EAT</b>"]

    for key in ("action", "quantity", "limit_price", "suggested_price",
                "phase", "score", "expected_net_after_fees"):
        val = data.get(key)
        if val is not None:
            if key == "expected_net_after_fees":
                lines.append(f"  Expected net: {round(float(val) * 100, 1)}%")
            else:
                lines.append(f"  {key.replace('_', ' ').title()}: {val}")

    if data.get("event_flag"):
        lines.append(
            f"  Event: {data.get('event_type', '?')}  "
            f"(+{data.get('confidence_boost', 0)} pt)"
        )
    return "\n".join(lines)


def _fmt_kill_switch(event_type: str, data: dict) -> str:
    ts    = datetime.now(EAT).strftime("%H:%M:%S")
    server = socket.gethostname()
    lines = [f"<b>SYSTEM STOPPED [{event_type}]  |  {ts} EAT</b>", f"  Server: {server}"]
    for key in ("reason", "equity", "peak", "drawdown_pct",
                "expectancy_pct", "threshold_pct", "fee_threshold_pct"):
        val = data.get(key)
        if val is not None:
            lines.append(f"  {key.replace('_', ' ').title()}: {val}")
    lines.append("\nManual review required before restarting.")
    return "\n".join(lines)


def _fmt_health_ping(data: dict) -> str:
    ts    = datetime.now(EAT).strftime("%H:%M:%S")
    state = data.get("state", "?")
    icons = {"HEALTHY": "🟢", "BELOW_FEE_FLOOR": "🟡",
             "DEGRADED": "🟠", "PAUSED": "🔴"}
    icon  = icons.get(state, "⚪")
    lines = [
        f"<b>{icon} HEALTH PING  |  {ts} EAT</b>",
        f"  State:      {state}",
        f"  Win rate:   {data.get('win_rate_pct', 0)}%",
        f"  Expectancy: {data.get('expectancy_pct', 0)}%",
        f"  Positions:  {data.get('positions', 0)}",
        f"  Equity:     {data.get('equity', '?')}",
        f"  Market:     {data.get('market', '?')}",
        f"  Pause:      {data.get('pause', False)}",
    ]
    return "\n".join(lines)


def _fmt_generic(event_type: str, severity: str, data: dict) -> str:
    ts    = datetime.now(EAT).strftime("%H:%M:%S")
    label = _LABEL[severity]
    lines = [f"<b>{label}  {event_type}  |  {ts} EAT</b>"]
    for k, v in data.items():
        lines.append(f"  {k}: {v}")
    return "\n".join(lines)


def _fmt_system_start(data: dict) -> str:
    ts   = datetime.now(EAT).strftime("%H:%M:%S")
    mode = data.get("mode", "Ziidi Signal Engine")
    server = socket.gethostname()
    ver  = data.get("version", "?")
    lines = [
        f"<b>SYSTEM ONLINE  |  {ts} EAT</b>",
        f"<b> Server Name: {server} </b>",
        f"  Version: {ver}",
        f"  Mode:    {mode}",
        f"  Watching: {', '.join(data.get('watchlist', []))}",
        f"  Paper trade: {data.get('paper_trade', True)}",
    ]
    return "\n".join(lines)


def _format(event_type: str, severity: str, data: dict) -> str:
    """Route to the right formatter based on event type."""
    if "BUY_ALERT" in event_type or "SELL_ALERT" in event_type:
        return _fmt_trade_alert(event_type, data)
    if event_type in ("KILL_SWITCH", "DRAWDOWN_KILL", "EXPECTANCY_KILL", "SYSTEM_PAUSED"):
        return _fmt_kill_switch(event_type, data)
    if event_type == "HEALTH_PING":
        return _fmt_health_ping(data)
    if event_type == "SYSTEM_START":
        return _fmt_system_start(data)
    return _fmt_generic(event_type, severity, data)


# ── Dispatch ───────────────────────────────────────────────────────────────────

def dispatch_notification(event_type: str, data: dict) -> None:
    """
    Route an event to the appropriate channel based on severity.

    CRITICAL  → Telegram (always)
    WARNING   → Telegram (always)   ← SYSTEM_START is WARNING, always fires
    INFO      → Telegram only if NOTIFY_INFO=True/1, otherwise suppressed

    Failed deliveries for CRITICAL/WARNING are printed to console so they
    are never silently swallowed. Check TELEGRAM_TOKEN / TELEGRAM_CHAT_ID
    in config/params.py if you see the warning below.
    """
    severity = get_severity(event_type)

    if severity == INFO and not _notify_info_enabled():
        return

    sent = _send_telegram(_format(event_type, severity, data))

    if not sent and severity in (CRITICAL, WARNING):
        token, chat_id = _get_credentials()
        reason = "no credentials set" if not token or not chat_id else "network / API error"
        fail_msg = (
            f"[NOTIFY FAIL] {event_type} ({severity}) not delivered to Telegram — {reason}  "
            "Set TELEGRAM_TOKEN + TELEGRAM_CHAT_ID in config/params.py"
        )
        print(fail_msg)
        _logger.warning(fail_msg)

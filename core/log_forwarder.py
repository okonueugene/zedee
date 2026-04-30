"""
Log Forwarder — tail logs/system.log and push ERROR / CRITICAL lines to Telegram.

Why this exists
───────────────
dispatch_notification() handles events routed through log_event().
This module is the safety net for everything else:

  • logging.error() / logging.critical() called directly from any module
  • Uncaught exceptions Python writes to the root logger
  • Any future module that logs without going through log_event()

How it works
────────────
A daemon thread seeks to the current end of the log file on startup
(so historical lines are never replayed) then polls for new lines every
POLL_INTERVAL seconds.  Any line whose level field is ERROR or CRITICAL
is forwarded to Telegram immediately.

De-duplication note
───────────────────
CRITICAL events from log_event() are already sent via dispatch_notification()
AND will be caught here — you may receive the message twice for a CRITICAL
event.  For a kill-switch or drawdown event that is intentional: a duplicate
alert is far preferable to a missed one.

ERROR-level lines (Python logging.ERROR = 40, between WARNING and CRITICAL)
are NEW information — log_event() never emits at that level, so there is
no duplication.

Lifecycle
─────────
Call start_log_forwarder(path) once at startup.
The daemon thread stops automatically when the main process exits.
Call .stop() explicitly for a clean shutdown.
"""

import os
import threading
import time
from datetime import datetime

from dateutil import tz

EAT            = tz.gettz('Africa/Nairobi')
POLL_INTERVAL  = 5    # seconds between reads when no new lines arrive
_FORWARD_MARKS = ("| ERROR   ", "| CRITICAL")   # match basicConfig levelname-8 format


class LogForwarder:
    """
    Daemon thread that tails a log file and forwards high-severity lines to Telegram.
    """

    def __init__(self, log_path: str) -> None:
        self._log_path = log_path
        self._stop     = threading.Event()
        self._thread   = threading.Thread(
            target=self._run,
            daemon=True,
            name="LogForwarder",
        )

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    # ── Internal loop ──────────────────────────────────────────────────────────

    def _run(self) -> None:
        # Wait up to 30 s for the log file to be created (first log_event call)
        deadline = time.monotonic() + 30
        while not self._stop.is_set():
            if os.path.exists(self._log_path):
                break
            if time.monotonic() > deadline:
                return          # log file never appeared — exit silently
            time.sleep(1)

        try:
            with open(self._log_path, "r", encoding="utf-8", errors="replace") as fh:
                fh.seek(0, os.SEEK_END)     # start at current EOF; ignore history

                while not self._stop.is_set():
                    line = fh.readline()

                    if not line:            # no new data yet
                        time.sleep(POLL_INTERVAL)
                        # File may have been rotated — re-check size
                        try:
                            if fh.tell() > os.path.getsize(self._log_path):
                                fh.seek(0)
                        except OSError:
                            pass
                        continue

                    line = line.rstrip()
                    if self._should_forward(line):
                        self._forward(line)

        except Exception:
            pass    # daemon thread must never crash the main process

    @staticmethod
    def _should_forward(line: str) -> bool:
        return any(mark in line for mark in _FORWARD_MARKS)

    def _forward(self, line: str) -> None:
        """Send one log line to Telegram, formatted for readability."""
        try:
            from core.notifications import _send_telegram     # late import avoids circular
            ts    = datetime.now(EAT).strftime("%H:%M:%S")
            level = "ERROR" if "| ERROR" in line else "CRITICAL"
            icon  = "" if level == "CRITICAL" else ""

            # Trim to keep the message inside Telegram's 4096-char limit
            body  = line[:450] + ("…" if len(line) > 450 else "")

            msg = (
                f"<b>{icon} [{level}] {ts} EAT</b>\n"
                f"<code>{body}</code>\n"
                f"<i>Source: system.log</i>"
            )
            _send_telegram(msg)
        except Exception:
            pass


# ── Public factory ─────────────────────────────────────────────────────────────

def start_log_forwarder(log_path: str) -> LogForwarder:
    """
    Create, start, and return a LogForwarder for the given log file path.

    Usage in main.py:
        from core.log_forwarder import start_log_forwarder
        _forwarder = start_log_forwarder(os.path.join(_LOG_DIR, "system.log"))
    """
    forwarder = LogForwarder(log_path)
    forwarder.start()
    return forwarder

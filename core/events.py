"""
Event Engine — corporate action detection and score boosting.

Scrapes NSE announcements for watchlist symbols and applies a
phase-conditional, hard-capped (+10 points max) boost to the raw
signal score produced by the Signal Engine.

Design principles
─────────────────
  Time-sensitive only  — only events from the current calendar year are
                         considered. Old announcements in the archive never
                         trigger a boost.
  Phase-conditional    — events matter most before a breakout (COMPRESSION)
                         and least after momentum is already visible (EXPANSION).
                         Dead-phase entries (CAPITULATION, NEUTRAL) get no boost.
  Capped influence     — the maximum boost is +10 score points regardless of
                         event type or phase. Events add information; they should
                         not override the base signal.
  Fail silently        — events are alpha-enhancement, not critical path.
                         Any network or parsing failure returns an empty dict.

Event types detected: DIVIDEND, BONUS, SPLIT, RIGHTS
"""

from datetime import datetime

import requests
from bs4 import BeautifulSoup
from dateutil import tz

EAT = tz.gettz('Africa/Nairobi')

_ANNOUNCEMENTS_URL = "https://afx.kwayisi.org/nse/"

# Raw point boost per event type (before phase multiplier and cap)
_EVENT_BOOSTS: dict[str, int] = {
    "DIVIDEND": 8,
    "BONUS":    6,
    "SPLIT":    5,
    "RIGHTS":   4,
}

# Events matter most before the breakout; less so once expansion is visible
_PHASE_MULTIPLIERS: dict[str, float] = {
    "COMPRESSION":       1.0,
    "EXHAUSTION":        0.6,
    "EXPANSION":         0.5,
    "NEUTRAL":           0.0,
    "CAPITULATION":      0.0,
    "INSUFFICIENT_DATA": 0.0,
}

_MAX_BOOST = 10.0


def _classify(snippet: str) -> str:
    """Return the corporate action type from a text snippet around a symbol."""
    s = snippet.upper()
    if "DIVIDEND" in s:
        return "DIVIDEND"
    if "BONUS" in s:
        return "BONUS"
    if "SPLIT" in s:
        return "SPLIT"
    if "RIGHTS" in s:
        return "RIGHTS"
    return "OTHER"


def fetch_active_events(watchlist: list[str]) -> dict:
    """
    Scan NSE announcements for current-year corporate actions on watchlist symbols.

    Returns:
        {symbol: {"event_type": str, "detected_at": ISO timestamp}}

    Only symbols with a recognised, time-sensitive event are included.
    Returns an empty dict on any network or parsing failure.
    """
    today  = datetime.now(EAT).date()
    events: dict = {}

    try:
        r = requests.get(
            _ANNOUNCEMENTS_URL,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/134.0 Safari/537.36"
                )
            },
            timeout=15,
        )
        r.raise_for_status()

        soup  = BeautifulSoup(r.text, 'html.parser')
        texts = soup.get_text(separator=" ").upper()

        # Time-sensitivity gate — only events from the current year qualify
        if str(today.year) not in texts:
            return {}

        for sym in watchlist:
            idx = texts.find(sym.upper())
            if idx == -1:
                continue

            # ±200-character context window around the symbol mention
            snippet    = texts[max(0, idx - 100): idx + 200]
            event_type = _classify(snippet)

            if event_type != "OTHER":
                events[sym] = {
                    "event_type":  event_type,
                    "detected_at": datetime.now(EAT).isoformat(),
                }

    except Exception:
        pass

    return events


def apply_event_boost(
    base_score: float,
    phase: str,
    event_info: dict | None,
) -> tuple[float, bool, str, int]:
    """
    Apply a phase-conditional, hard-capped boost to a raw signal score.

    Args:
        base_score  — raw score from score_signal()
        phase       — current market phase string
        event_info  — entry from fetch_active_events(), or None

    Returns:
        final_score      — score after boost (may equal base_score)
        event_flag       — True if a boost was applied
        event_type       — e.g. "DIVIDEND", empty string if no boost
        confidence_boost — integer points added (0 if no boost)
    """
    if not event_info:
        return base_score, False, "", 0

    event_type = event_info.get("event_type", "OTHER")
    phase_mult = _PHASE_MULTIPLIERS.get(phase, 0.0)

    if phase_mult == 0.0:
        return base_score, False, "", 0

    raw_boost    = _EVENT_BOOSTS.get(event_type, 0) * phase_mult
    capped_boost = round(min(raw_boost, _MAX_BOOST), 1)
    final_score  = base_score + capped_boost

    return final_score, True, event_type, int(round(capped_boost))

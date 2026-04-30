"""
Analytics Engine — turn the trade journal into structured insight.

Answers the two questions the calibration engine needs:
  - What is working?
  - What is failing?

All functions are pure: DataFrame in, dict/DataFrame out, no side effects.
This makes them safe to call any time, including interactively.

Pattern analyses exposed:
  by_phase      → which market conditions produce edge
  by_hold_time  → how long trades should stay open
  by_score      → whether signal score actually predicts outcome
"""

import pandas as pd


# ── Core metrics ───────────────────────────────────────────────────────────────

def _normalise_pnl_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise different journal schemas to a single column name used by analytics.
    Prefers the canonical trades-table column `pnl_pct`, but supports legacy CSV
    column `net_return_pct` as a fallback.
    """
    if df.empty:
        return df
    df = df.copy()
    if "pnl_pct" not in df.columns and "net_return_pct" in df.columns:
        df["pnl_pct"] = df["net_return_pct"]
    return df


def core_metrics(df: pd.DataFrame) -> dict:
    """
    Compute fundamental performance metrics from the trade journal.

    Returns a dict with:
        trade_count, win_rate, loss_rate,
        avg_win, avg_loss, expectancy,
        max_drawdown, profit_factor
    """
    df = _normalise_pnl_column(df)
    if df.empty or 'result' not in df.columns or 'pnl_pct' not in df.columns:
        return {}

    wins  = df[df['result'] == 'WIN']
    losses = df[df['result'] == 'LOSS']

    win_rate  = len(wins)   / len(df)
    loss_rate = len(losses) / len(df)
    avg_win   = float(wins['pnl_pct'].mean())   if not wins.empty   else 0.0
    avg_loss  = float(losses['pnl_pct'].mean()) if not losses.empty else 0.0

    # Standard expectancy formula: E = (P_win × avg_win) − (P_loss × |avg_loss|)
    # avg_loss is already negative so the sign works out correctly.
    expectancy = (win_rate * avg_win) + (loss_rate * avg_loss)

    # Max drawdown on the cumulative equity curve implied by the trade sequence
    cumulative  = (1 + df['pnl_pct']).cumprod()
    rolling_max = cumulative.cummax()
    max_drawdown = float(((cumulative - rolling_max) / rolling_max).min())

    # Profit factor: gross profit / gross loss
    gross_profit = wins['pnl_pct'].sum()   if not wins.empty   else 0.0
    gross_loss   = abs(losses['pnl_pct'].sum()) if not losses.empty else 0.0
    profit_factor = round(gross_profit / gross_loss, 3) if gross_loss > 0 else None

    return {
        "trade_count":   len(df),
        "win_rate":      round(win_rate, 4),
        "loss_rate":     round(loss_rate, 4),
        "avg_win":       round(avg_win, 4),
        "avg_loss":      round(avg_loss, 4),
        "expectancy":    round(expectancy, 4),
        "max_drawdown":  round(max_drawdown, 4),
        "profit_factor": profit_factor,
    }


# ── Pattern analyses ───────────────────────────────────────────────────────────

def by_phase(df: pd.DataFrame) -> pd.DataFrame:
    """
    Win rate and expectancy broken down by entry phase.
    This is where you find which market condition actually produces edge.
    If EXPANSION entries have 65% win rate and COMPRESSION entries have 30%,
    that's a parameter you can act on immediately.
    """
    df = _normalise_pnl_column(df)
    if df.empty or 'phase' not in df.columns:
        return pd.DataFrame()

    rows = []
    for phase, grp in df.groupby('phase'):
        wins  = grp[grp['result'] == 'WIN']
        losses = grp[grp['result'] == 'LOSS']
        wr    = len(wins) / len(grp)
        aw    = float(wins['pnl_pct'].mean())   if not wins.empty   else 0.0
        al    = float(losses['pnl_pct'].mean()) if not losses.empty else 0.0
        rows.append({
            "phase":      phase,
            "trades":     len(grp),
            "win_rate":   round(wr, 3),
            "avg_win":    round(aw, 4),
            "avg_loss":   round(al, 4),
            "expectancy": round((wr * aw) + ((1 - wr) * al), 4),
        })

    return (
        pd.DataFrame(rows)
        .sort_values("expectancy", ascending=False)
        .reset_index(drop=True)
    )


def by_hold_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performance segmented into holding-period buckets.
    Answers: are you cutting winners too early or holding losers too long?
    Requires holding_hours column in the journal (added in journal.py upgrade).
    """
    df = _normalise_pnl_column(df)
    if df.empty or 'holding_hours' not in df.columns:
        return pd.DataFrame()

    df    = df.copy()
    bins  = [0, 4, 8, 24, 72, float('inf')]
    lbls  = ['<4h', '4–8h', '8–24h', '24–72h', '>72h']
    df['hold_bucket'] = pd.cut(df['holding_hours'], bins=bins, labels=lbls)

    rows = []
    for bucket, grp in df.groupby('hold_bucket', observed=True):
        wins = grp[grp['result'] == 'WIN']
        rows.append({
            "hold_bucket":   str(bucket),
            "trades":        len(grp),
            "win_rate":      round(len(wins) / len(grp), 3) if len(grp) > 0 else 0.0,
            "avg_net_pct":   round(float(grp['pnl_pct'].mean()), 4),
            "avg_hold_hrs":  round(float(grp['holding_hours'].mean()), 1),
        })

    return pd.DataFrame(rows)


def by_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performance segmented by signal score bands.
    Answers: does the score actually predict outcome, or is it noise?
    If '75+' band consistently underperforms '50–75', the scoring logic needs work.
    Requires score column in the journal (added in journal.py upgrade).
    """
    df = _normalise_pnl_column(df)
    if df.empty or 'score' not in df.columns:
        return pd.DataFrame()

    df    = df.copy()
    bins  = [-100, 0, 50, 75, 100]
    lbls  = ['<0 (avoid)', '0–50', '50–75', '75+']
    df['score_band'] = pd.cut(df['score'], bins=bins, labels=lbls)

    rows = []
    for band, grp in df.groupby('score_band', observed=True):
        wins = grp[grp['result'] == 'WIN']
        rows.append({
            "score_band":  str(band),
            "trades":      len(grp),
            "win_rate":    round(len(wins) / len(grp), 3) if len(grp) > 0 else 0.0,
            "avg_net_pct": round(float(grp['pnl_pct'].mean()), 4),
        })

    return pd.DataFrame(rows)


# ── Full report ────────────────────────────────────────────────────────────────

def full_report(df: pd.DataFrame) -> dict:
    """
    Combine all analytics into a single structured report dict.
    Safe to call with an empty or partial journal — each section degrades gracefully.
    """
    return {
        "core":     core_metrics(df),
        "by_phase": by_phase(df).to_dict(orient='records'),
        "by_hold":  by_hold_time(df).to_dict(orient='records'),
        "by_score": by_score(df).to_dict(orient='records'),
    }

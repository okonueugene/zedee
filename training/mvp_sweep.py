"""
MVP backtest driver (Module 6) + validation (Module 7).

Threshold grid ONLY: 55, 60, 65, 70, 75.
Metrics: trades, win_rate, avg_win (net, winners), avg_loss, expectancy, avg_mfe (all trades),
avg_mfe_winners, avg_gross_win, mfe_capture_ratio (mean gross/mfe on wins), max_drawdown (compounded equity).

Usage:
  python training/mvp_sweep.py
  python training/mvp_sweep.py --labelled training/nse_labelled.parquet
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from core.mvp_engine import (
    compute_stop,
    exit_fill_price,
    new_position,
    should_enter,
    should_exit,
    update_excursions as mvp_update_excursions,
)
from training.features import DATE_COL, FEATURE_COLS, SYMBOL_COL, TEST_START

MODEL_PATH = _ROOT / "training" / "ziidi_signal_model.pkl"
OUT_DIR = _ROOT / "training" / "mvp_backtest"

# Module 6 — ONLY this grid
THRESHOLDS = (55, 60, 65, 70, 75)

# Match training/features.py fee constant
ROUND_TRIP_FEE = 0.03874


def _load_model():
    if not MODEL_PATH.exists():
        print(f"[error] Model not found: {MODEL_PATH}")
        sys.exit(1)
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def _prepare_df(labelled_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(labelled_path)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    test_df = df[df[DATE_COL] >= TEST_START].copy()
    if len(test_df) < 10:
        print("[error] Test set too small.")
        sys.exit(1)

    model = _load_model()
    X = test_df[FEATURE_COLS]
    test_df = test_df.copy()
    test_df["prob"] = model.predict_proba(X)[:, 1]
    test_df["ml_score"] = (test_df["prob"] * 100).round(1)

    if "high" not in test_df.columns:
        test_df["high"] = test_df["close"].astype(float)
    if "low" not in test_df.columns:
        test_df["low"] = test_df["close"].astype(float)
    if "close" not in test_df.columns:
        print("[error] Labelled data must include close (and preferably high/low).")
        sys.exit(1)

    test_df["high_5d"] = test_df.groupby(SYMBOL_COL)["high"].transform(
        lambda s: s.rolling(5, min_periods=1).max()
    )
    return test_df.sort_values([SYMBOL_COL, DATE_COL])


def _run_symbol(
    sym_df: pd.DataFrame,
    threshold: float,
    symbol: str,
) -> tuple[list[dict], int]:
    """
    One symbol, chronological daily bars.

    OHLC hygiene: some feeds have high < close or low > close; bound with close so
    MFE/MAE stay consistent with bar range.

    Returns (trades, mfe_violation_count).
    """
    trades: list[dict] = []
    pos: dict | None = None
    mfe_violations = 0

    for _, row in sym_df.iterrows():
        if pos is not None:
            pos["days"] += 1
            cl = float(row["close"])
            hi = max(float(row["high"]), cl)
            lo = min(float(row["low"]), cl)
            mvp_update_excursions(pos, hi, lo)
            compute_stop(pos, cl)
            reason = should_exit(pos, cl, low=lo)
            if reason is not None:
                fill = exit_fill_price(pos, lo, cl)
                ep = float(pos["entry_price"])
                gross = (fill - ep) / ep
                net = gross - ROUND_TRIP_FEE
                mfe_path = float(pos["mfe"])
                # Invariant: long gross exit return cannot exceed path MFE (high-based).
                if gross > mfe_path + 1e-5:
                    mfe_violations += 1
                    mfe_path = max(mfe_path, gross)
                trades.append({
                    "symbol": symbol,
                    "entry_price": ep,
                    "exit_price": fill,
                    "gross_ret": gross,
                    "net_ret": net,
                    "mfe": mfe_path,
                    "mae": float(pos["mae"]),
                    "reason": reason,
                })
                pos = None

        if pos is None:
            if should_enter(row, threshold):
                pos = new_position(float(row["close"]))

    return trades, mfe_violations


def _equity_max_drawdown(returns: np.ndarray) -> float:
    """
    Per-trade simple returns compounded: equity_0 = 1, equity_t = prod(1+r).
    Drawdown from running peak — always in [0, 1] (fraction of peak lost).
    """
    if len(returns) == 0:
        return 0.0
    growth = np.cumprod(1.0 + returns.astype(float))
    peak = np.maximum.accumulate(growth)
    dd = (peak - growth) / np.maximum(peak, 1e-12)
    return float(np.clip(dd.max(), 0.0, 1.0))


def _metrics(trades: list[dict]) -> dict:
    """
    avg_win / avg_loss use **net** return per trade (after fee).

    MFE diagnostics: compare winners only — avg_net_win must be <= avg_mfe_winners
    (since net < gross <= mfe for each winning trade).
    mfe_capture_ratio = mean(gross/mfe) on winning trades with mfe > 0.
    """
    nets = np.array([t["net_ret"] for t in trades], dtype=float)
    grosses = np.array([t["gross_ret"] for t in trades], dtype=float)
    mfes = np.array([t["mfe"] for t in trades], dtype=float)
    maes = np.array([t["mae"] for t in trades], dtype=float)
    n = len(nets)
    if n == 0:
        return {
            "trades": 0,
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "expectancy": 0.0,
            "avg_mfe": 0.0,
            "avg_mae": 0.0,
            "avg_mfe_winners": 0.0,
            "avg_gross_win": 0.0,
            "max_drawdown": 0.0,
            "mfe_capture_ratio": 0.0,
        }

    win_mask = nets > 0
    loss_mask = nets < 0

    max_dd = _equity_max_drawdown(nets)

    win_rate = float(win_mask.sum() / n)
    avg_win = float(nets[win_mask].mean()) if win_mask.any() else 0.0
    avg_loss_mag = float((-nets[loss_mask]).mean()) if loss_mask.any() else 0.0
    expectancy = float(nets.mean())
    avg_mfe = float(mfes.mean())
    avg_mae = float(maes.mean())

    gw = grosses[win_mask]
    mw = mfes[win_mask]
    avg_mfe_winners = float(mw.mean()) if win_mask.any() else 0.0
    avg_gross_win = float(gw.mean()) if win_mask.any() else 0.0
    if win_mask.any() and mw.max() > 1e-12:
        cap = gw / np.maximum(mw, 1e-12)
        mfe_cap = float(np.clip(cap.mean(), 0.0, 1.0))
    else:
        mfe_cap = 0.0

    return {
        "trades": n,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss_mag,
        "expectancy": expectancy,
        "avg_mfe": avg_mfe,
        "avg_mae": avg_mae,
        "avg_mfe_winners": avg_mfe_winners,
        "avg_gross_win": avg_gross_win,
        "max_drawdown": max_dd,
        "mfe_capture_ratio": mfe_cap,
    }


def run_sweep(labelled_path: Path) -> tuple[dict[int, dict], dict[int, list[dict]], int]:
    df = _prepare_df(labelled_path)
    symbols = df[SYMBOL_COL].unique()
    out: dict[int, dict] = {}
    by_th: dict[int, list[dict]] = {}
    total_mfe_violations = 0

    for th in THRESHOLDS:
        all_trades: list[dict] = []
        for sym in symbols:
            sym_df = df[df[SYMBOL_COL] == sym]
            chunk, viol = _run_symbol(sym_df, float(th), str(sym))
            total_mfe_violations += viol
            for t in chunk:
                t["threshold"] = int(th)
            all_trades.extend(chunk)
        by_th[int(th)] = all_trades
        out[int(th)] = _metrics(all_trades)

    return out, by_th, total_mfe_violations


def validate_mvp_sweep(results: dict[int, dict]) -> dict:
    """
    Module 7 — plateau acceptance (not “every threshold must pass”).

    ACCEPT if there exists a **contiguous** run of thresholds where **each** has:
      trades >= 30 and expectancy >= round-trip fee floor.

    Prefer a **plateau** (stable band), not a single spike — warn on isolated spikes.
    """
    lines: list[str] = []
    th_sorted = sorted(results.keys())
    FEE = ROUND_TRIP_FEE

    def band_ok(th: int) -> bool:
        m = results[th]
        return m["trades"] >= 30 and m["expectancy"] >= FEE

    runs: list[list[int]] = []
    i = 0
    while i < len(th_sorted):
        if not band_ok(th_sorted[i]):
            i += 1
            continue
        j = i + 1
        while j < len(th_sorted) and band_ok(th_sorted[j]):
            j += 1
        runs.append(th_sorted[i:j])
        i = j

    accept = any(len(r) >= 1 for r in runs)

    if runs:
        for r in runs:
            if len(r) >= 2:
                lines.append(
                    f"OK: contiguous plateau {r[0]}–{r[-1]} "
                    f"({len(r)} thresholds, each trades>=30 & expectancy>={FEE:.4f})"
                )
            else:
                lines.append(
                    f"OK: single threshold {r[0]} passes band (trades>=30 & expectancy>={FEE:.4f})"
                )
    else:
        lines.append(
            f"REJECT: no threshold has trades>=30 and expectancy>={FEE:.4f} (fee floor)"
        )
        accept = False

    for th in th_sorted:
        m = results[th]
        if m["trades"] > 0 and m["win_rate"] >= 0.999 and m["trades"] < 30:
            lines.append(f"WARN: threshold {th} near-100% win rate, small n={m['trades']}")

    exps = [results[th]["expectancy"] for th in th_sorted]
    if len(exps) >= 3:
        mx_i = int(np.argmax(exps))
        if 0 < mx_i < len(exps) - 1:
            if exps[mx_i] > exps[mx_i - 1] * 3 and exps[mx_i] > exps[mx_i + 1] * 3:
                lines.append(
                    f"WARN: isolated expectancy spike at {th_sorted[mx_i]} vs neighbours "
                    "(prefer a plateau)"
                )

    for th in th_sorted:
        m = results[th]
        if m["trades"] >= 20 and m.get("avg_mfe_winners", 0) > 1e-6:
            if m["avg_gross_win"] > m["avg_mfe_winners"] + 1e-5:
                lines.append(
                    f"WARN: threshold {th} avg_gross_win > avg_mfe_winners (metric bug or data)"
                )

    ratios = [results[th]["mfe_capture_ratio"] for th in th_sorted if results[th]["trades"] >= 20]
    if len(ratios) >= 2 and ratios[-1] < ratios[0]:
        lines.append("WARN: MFE capture (mean gross/mfe on wins) fell from low to high threshold")

    lines.append("ACCEPT" if accept else "REJECT")
    return {"accept": accept, "lines": lines}


def main() -> None:
    ap = argparse.ArgumentParser(description="MVP threshold sweep (frozen modules 1–5).")
    ap.add_argument(
        "--labelled",
        type=Path,
        default=_ROOT / "training" / "nse_labelled.parquet",
        help="Labelled parquet with OHLC + features",
    )
    args = ap.parse_args()

    if not args.labelled.exists():
        print(f"[error] Missing {args.labelled}")
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results, by_th, mfe_violations = run_sweep(args.labelled)

    rows = []
    for th in THRESHOLDS:
        m = results[th]
        rows.append({"threshold": th, **m})

    table = pd.DataFrame(rows)
    table_path = OUT_DIR / "mvp_sweep_metrics.csv"
    table.to_csv(table_path, index=False)
    print(table.to_string(index=False))
    print(f"\n[ok] Wrote {table_path}")

    all_rows: list[dict] = []
    for th in THRESHOLDS:
        all_rows.extend(by_th[int(th)])
    trades_path = OUT_DIR / "trades.csv"
    if all_rows:
        pd.DataFrame(all_rows).to_csv(trades_path, index=False)
        print(f"[ok] Wrote {trades_path} ({len(all_rows)} rows across thresholds)")
    if mfe_violations:
        print(
            f"[warn] {mfe_violations} trades had gross_ret > path MFE (OHLC fixed or mfe raised); "
            "check data quality"
        )

    val = validate_mvp_sweep(results)
    val_path = OUT_DIR / "mvp_validation.json"
    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "results": results,
                "validation": val,
                "mfe_gross_violations": mfe_violations,
            },
            f,
            indent=2,
            default=str,
        )
    print(f"[ok] Wrote {val_path}")
    for line in val["lines"]:
        print(line)


if __name__ == "__main__":
    main()

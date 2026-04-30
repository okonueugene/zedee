"""
Parameter Optimizer — grid search for optimal system_params.json starting values.

CRITICAL: searches on the VALIDATION set (2021–2022) only.
Never touches the test set — using test data for tuning would inflate backtest results.

Parameters searched
───────────────────
  score_threshold   :  [45, 50, 55, 60, 65, 70, 75, 80]
  vol_mult          :  [0.9, 1.0, 1.1, 1.2, 1.3]
  trail_stop_pct    :  [0.03, 0.04, 0.05, 0.06, 0.07]

Optimisation metric: expectancy on validation set after fees.
Secondary sort: win_rate (prefer higher win rate at equal expectancy).
Tertiary sort: trade count (prefer more trades at equal expectancy + win_rate — better stats).

Outputs
───────
  training/optimal_params.json   — best parameter set found
  training/grid_search_log.csv   — full grid results

Usage
─────
  python training/param_optimizer.py
  python training/param_optimizer.py --update-system-params   (writes to system_params.json)
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from training.features import (
    CLOSE_COL, DATE_COL, FEATURE_COLS, HOLD_DAYS,
    ROUND_TRIP_FEE, SYMBOL_COL,
    VAL_END, VAL_START,
)

MODEL_PATH     = _ROOT / "training" / "ziidi_signal_model.pkl"
OPT_PATH       = _ROOT / "training" / "optimal_params.json"
GRID_LOG_PATH  = _ROOT / "training" / "grid_search_log.csv"
PARAMS_FILE    = _ROOT / "system_params.json"

# Grid definition
_SCORE_THRESHOLDS  = [45, 50, 55, 60, 65, 70, 75, 80]
_VOL_MULTS         = [0.9, 1.0, 1.1, 1.2, 1.3]
_TRAIL_STOPS       = [0.03, 0.04, 0.05, 0.06, 0.07]


def _simulate(
    sym_df:          pd.DataFrame,
    proba_col:       str,
    score_threshold: float,
    vol_mult:        float,
    trail_stop_pct:  float,
    cooldown_bars:   int = 2,
) -> list[dict]:
    """
    Simulate trades for one symbol on the val set.
    Returns list of trade result dicts.
    """
    trades = []
    cooldown = 0

    for int_i in range(len(sym_df) - HOLD_DAYS):
        if cooldown > 0:
            cooldown -= 1
            continue

        row      = sym_df.iloc[int_i]
        score    = row["ml_score"]
        vol_sp   = row.get("vol_spike", np.nan)
        exp_trig = row.get("expansion_trigger", np.nan)
        entry_px = float(row[CLOSE_COL])

        if score < score_threshold:
            continue
        if not np.isnan(vol_sp) and vol_sp < vol_mult:
            continue
        if np.isnan(exp_trig) or exp_trig < 0.5:
            continue

        # Pullback gate
        if int_i + 1 >= len(sym_df):
            continue
        pullback_px = float(sym_df.iloc[int_i + 1][CLOSE_COL])
        if pullback_px > entry_px * 0.995:
            continue

        entry_price = pullback_px
        actual_i    = int_i + 1
        end_i       = min(actual_i + HOLD_DAYS, len(sym_df))
        hold_bars   = sym_df.iloc[actual_i:end_i]
        if hold_bars.empty:
            continue

        # Trailing stop: walk bars, exit early if price drops trail_stop_pct from highest
        highest     = entry_price
        exit_price  = float(hold_bars.iloc[-1][CLOSE_COL])
        for _, bar in hold_bars.iterrows():
            bar_px  = float(bar[CLOSE_COL])
            highest = max(highest, bar_px)
            if bar_px <= highest * (1 - trail_stop_pct):
                exit_price = bar_px
                break

        gross_ret = (exit_price - entry_price) / entry_price
        net_ret   = gross_ret - ROUND_TRIP_FEE

        trades.append({
            "net_ret": net_ret,
            "result":  "WIN" if net_ret > 0 else "LOSS",
        })
        cooldown = max(cooldown_bars, HOLD_DAYS - 1)

    return trades


def _expectancy(trades: list[dict]) -> tuple[float, float, int]:
    """Returns (expectancy, win_rate, n_trades)."""
    if not trades:
        return -999.0, 0.0, 0
    df     = pd.DataFrame(trades)
    wins   = df[df["result"] == "WIN"]["net_ret"]
    losses = df[df["result"] == "LOSS"]["net_ret"]
    n      = len(df)
    wr     = len(wins) / n
    aw     = float(wins.mean())   if len(wins)   > 0 else 0.0
    al     = float(-losses.mean()) if len(losses) > 0 else 0.0
    exp    = wr * aw - (1 - wr) * al
    return exp, wr, n


def run_grid_search(
    labelled_path: Path,
    cooldown_bars: int = 2,
) -> dict:
    """Run full grid search and return best parameter dict."""
    if not MODEL_PATH.exists():
        print(f"[error] Model not found: {MODEL_PATH}")
        print("        Run train_model.py first.")
        sys.exit(1)

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    print(f"[load] {labelled_path}")
    df = pd.read_parquet(labelled_path)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])

    val_df = df[(df[DATE_COL] >= VAL_START) & (df[DATE_COL] <= VAL_END)].copy()
    print(f"[val]  {len(val_df):,} rows  |  symbols: {val_df[SYMBOL_COL].nunique()}")

    if len(val_df) < 50:
        print("[error] Validation set too small.  Check date coverage in master data.")
        sys.exit(1)

    # ── Score all val rows once ───────────────────────────────────────────────
    X_val           = val_df[FEATURE_COLS]
    val_df["prob"]  = model.predict_proba(X_val)[:, 1]
    val_df["ml_score"] = (val_df["prob"] * 100).round(1)

    # Pre-group by symbol for faster inner loop
    sym_groups = {
        sym: grp.sort_values(DATE_COL).reset_index(drop=True)
        for sym, grp in val_df.groupby(SYMBOL_COL)
    }

    total_combos = len(_SCORE_THRESHOLDS) * len(_VOL_MULTS) * len(_TRAIL_STOPS)
    print(f"[grid] {total_combos} combinations to evaluate …")

    grid_rows = []
    best      = {"expectancy": -999.0, "win_rate": 0.0, "trades": 0}

    for sc_thresh, vol_m, trail in product(_SCORE_THRESHOLDS, _VOL_MULTS, _TRAIL_STOPS):
        all_trades: list[dict] = []

        for sym_df in sym_groups.values():
            all_trades.extend(_simulate(sym_df, "prob", sc_thresh, vol_m, trail, cooldown_bars))

        exp, wr, n = _expectancy(all_trades)

        grid_rows.append({
            "score_threshold": sc_thresh,
            "vol_mult":        vol_m,
            "trail_stop_pct":  trail,
            "expectancy":      round(exp, 5),
            "win_rate":        round(wr, 4),
            "trades":          n,
        })

        # Primary: expectancy ↑, secondary: win_rate ↑, tertiary: trades ↑
        if (exp, wr, n) > (best["expectancy"], best["win_rate"], best["trades"]):
            best = {
                "score_threshold":   sc_thresh,
                "vol_mult_morning":  round(vol_m, 2),
                "vol_mult_afternoon": round(max(0.9, vol_m - 0.1), 2),
                "trailing_stop_pct": trail,
                "expectancy":        round(exp, 5),
                "win_rate":          round(wr, 4),
                "trades":            n,
            }

    # ── Report ────────────────────────────────────────────────────────────────
    grid_df = pd.DataFrame(grid_rows).sort_values("expectancy", ascending=False)

    print(f"\n── Top 5 parameter sets (validation 2021–2022) ─────────────────")
    for _, r in grid_df.head(5).iterrows():
        print(
            f"   sc={r['score_threshold']:4.0f}  vol={r['vol_mult']:.1f}  "
            f"trail={r['trail_stop_pct']:.2f}  "
            f"E={r['expectancy']*100:.3f}%  wr={r['win_rate']*100:.1f}%  n={r['trades']}"
        )

    print(f"\n── Best: ───────────────────────────────────────────────────────")
    for k, v in best.items():
        print(f"   {k:<26} {v}")

    fee_pct = ROUND_TRIP_FEE * 100
    if best["expectancy"] <= ROUND_TRIP_FEE:
        print(f"\n   ⚠  Best expectancy ({best['expectancy']*100:.3f}%) ≤ fee floor ({fee_pct:.3f}%)")
        print("      The model may not have edge on this validation window.")
        print("      Consider: wider label window, larger data, more symbols.")
    else:
        print(f"\n   ✅ Best expectancy ({best['expectancy']*100:.3f}%) > fee floor ({fee_pct:.3f}%)")

    # ── Save ──────────────────────────────────────────────────────────────────
    OPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OPT_PATH.write_text(json.dumps(best, indent=2))
    print(f"\n[save] {OPT_PATH}")

    grid_df.to_csv(GRID_LOG_PATH, index=False)
    print(f"[save] {GRID_LOG_PATH}  ({len(grid_df)} rows)")

    return best


def update_system_params(best: dict) -> None:
    """Merge optimal params into system_params.json, preserving other keys."""
    if not PARAMS_FILE.exists():
        print(f"[warn] {PARAMS_FILE} not found — skipping update")
        return

    with open(PARAMS_FILE) as f:
        params = json.load(f)

    updatable_keys = ["score_threshold", "vol_mult_morning", "vol_mult_afternoon", "trailing_stop_pct"]
    changes = {}
    for k in updatable_keys:
        if k in best:
            old = params.get(k)
            params[k] = best[k]
            if old != best[k]:
                changes[k] = {"old": old, "new": best[k]}

    with open(PARAMS_FILE, "w") as f:
        json.dump(params, f, indent=2)

    if changes:
        print(f"\n[update] system_params.json updated:")
        for k, chg in changes.items():
            print(f"   {k}: {chg['old']} → {chg['new']}")
    else:
        print(f"\n[update] system_params.json — no changes needed")


def main() -> None:
    try:
        import xgboost  # noqa: F401
    except ImportError:
        print("[error] xgboost not installed.  Run:  pip install xgboost scikit-learn pyarrow")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Grid search for optimal Ziidi parameters")
    labelled_default = _ROOT / "training" / "nse_labelled.parquet"
    parser.add_argument("--labelled",            type=Path,  default=labelled_default)
    parser.add_argument("--cooldown",            type=int,   default=2,
                        help="Cooldown bars between trades in simulation")
    parser.add_argument("--update-system-params", action="store_true",
                        help="Write best params directly to system_params.json")
    args = parser.parse_args()

    if not args.labelled.exists():
        print(f"[error] Labelled dataset not found: {args.labelled}")
        print("        Run label_generator.py first.")
        sys.exit(1)

    best = run_grid_search(args.labelled, cooldown_bars=args.cooldown)

    if args.update_system_params:
        update_system_params(best)
    else:
        print(f"\nTo apply these params:  python training/param_optimizer.py --update-system-params")


if __name__ == "__main__":
    main()

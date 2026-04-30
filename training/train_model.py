"""
Train XGBoost binary classifier on the labelled NSE dataset.

Chronological split (no random shuffle — time series must not leak future into past):
    Train :  date  ≤ 2020-12-31
    Val   :  2021-01-01 → 2022-12-31
    Test  :  date  ≥ 2023-01-01

The model predicts P(5-day return > 3.874% fee floor) for each symbol/date.
Evaluation focuses on:
  • Precision/Recall on the positive class (costly to miss trades; costlier to take losers)
  • ROC-AUC (overall discrimination)
  • Simulated expectancy on the held-out test set after fees

Saves
─────
  training/ziidi_signal_model.pkl   — trained XGBoostClassifier
  training/model_report.json        — evaluation metrics + feature importance

Usage
─────
  python training/train_model.py
  python training/train_model.py --labelled training/nse_labelled.parquet --threshold 0.5
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

from training.features import (
    DATE_COL, FEATURE_COLS, HOLD_DAYS, LABEL_COL,
    ROUND_TRIP_FEE, SYMBOL_COL,
    TEST_START, TRAIN_END, VAL_END, VAL_START,
)

MODEL_PATH  = _ROOT / "training" / "ziidi_signal_model.pkl"
REPORT_PATH = _ROOT / "training" / "model_report.json"


def _check_xgboost() -> None:
    try:
        import xgboost  # noqa: F401
    except ImportError:
        print("[error] xgboost not installed.  Run:  pip install xgboost scikit-learn")
        sys.exit(1)


def chronological_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by date, never by random shuffle."""
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    train = df[df[DATE_COL] <= TRAIN_END]
    val   = df[(df[DATE_COL] >= VAL_START) & (df[DATE_COL] <= VAL_END)]
    test  = df[df[DATE_COL] >= TEST_START]
    return train, val, test


def _check_feature_coverage(df: pd.DataFrame, split_name: str) -> None:
    """Warn about features that are entirely NaN in a split."""
    all_nan = [c for c in FEATURE_COLS if df[c].isna().all()]
    if all_nan:
        print(f"  ⚠  [{split_name}] ALL-NaN features: {all_nan}")


def simulate_expectancy(
    proba: np.ndarray,
    fwd_returns: pd.Series,
    threshold: float,
) -> dict:
    """
    For rows where predicted probability ≥ threshold, compute:
        win_rate   — fraction of taken trades that are profitable (net > 0)
        avg_win    — mean net return on winners
        avg_loss   — mean net return on losers  (expressed as positive loss)
        expectancy — (win_rate × avg_win) − (loss_rate × avg_loss)

    Returns the same expectancy formula used by learning/analytics.py.
    """
    mask      = proba >= threshold
    if mask.sum() == 0:
        return {"trades": 0, "win_rate": 0.0, "expectancy": 0.0}

    net_rets  = fwd_returns[mask] - ROUND_TRIP_FEE
    wins      = net_rets[net_rets > 0]
    losses    = net_rets[net_rets <= 0]

    win_rate  = len(wins) / len(net_rets)
    avg_win   = float(wins.mean())  if len(wins)   > 0 else 0.0
    avg_loss  = float(-losses.mean()) if len(losses) > 0 else 0.0
    expectancy = win_rate * avg_win - (1 - win_rate) * avg_loss

    return {
        "trades":     int(mask.sum()),
        "win_rate":   round(win_rate, 4),
        "avg_win":    round(avg_win, 4),
        "avg_loss":   round(avg_loss, 4),
        "expectancy": round(expectancy, 4),
        "threshold":  threshold,
    }


def train(labelled_path: Path, threshold: float = 0.50) -> None:
    import xgboost as xgb
    from sklearn.metrics import (
        average_precision_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    print(f"[load] {labelled_path}")
    df = pd.read_parquet(labelled_path)

    # Drop rows with NaN in the target label
    df = df.dropna(subset=[LABEL_COL])

    # ── Chronological split ───────────────────────────────────────────────────
    train_df, val_df, test_df = chronological_split(df)
    print(f"[split] train={len(train_df):,}  val={len(val_df):,}  test={len(test_df):,}")

    if len(train_df) < 500:
        print("[error] Training set too small (<500 rows).  Ensure data covers pre-2021.")
        sys.exit(1)
    if len(val_df) < 100:
        print("[warn] Validation set small (<100 rows).  Early stopping may not be reliable.")
    if len(test_df) < 100:
        print("[warn] Test set small (<100 rows).  Expectancy estimate will have high variance.")

    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        _check_feature_coverage(split_df, split_name)

    X_train = train_df[FEATURE_COLS]
    y_train = train_df[LABEL_COL].astype(int)
    X_val   = val_df[FEATURE_COLS]
    y_val   = val_df[LABEL_COL].astype(int)
    X_test  = test_df[FEATURE_COLS]
    y_test  = test_df[LABEL_COL].astype(int)

    # Class imbalance weight: negative / positive
    pos = int(y_train.sum())
    neg = len(y_train) - pos
    if pos == 0:
        print("[error] No positive examples in training set.  Check label threshold.")
        sys.exit(1)
    scale_pos_weight = neg / pos
    print(f"[class] train pos={pos:,}  neg={neg:,}  scale_pos_weight={scale_pos_weight:.2f}")

    # ── XGBoost model ─────────────────────────────────────────────────────────
    model = xgb.XGBClassifier(
        n_estimators        = 1000,
        learning_rate       = 0.05,
        max_depth           = 6,
        subsample           = 0.80,
        colsample_bytree    = 0.80,
        min_child_weight    = 5,    # reduces overfitting on small positive class
        scale_pos_weight    = scale_pos_weight,
        eval_metric         = "logloss",
        early_stopping_rounds = 50,
        random_state        = 42,
        n_jobs              = -1,
    )

    print("[train] Fitting XGBoost …")
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )

    best_iter = model.best_iteration
    print(f"[train] Best iteration: {best_iter}")

    # ── Validation evaluation ─────────────────────────────────────────────────
    val_proba  = model.predict_proba(X_val)[:, 1]
    val_pred   = (val_proba >= threshold).astype(int)
    val_auc    = roc_auc_score(y_val, val_proba)
    val_ap     = average_precision_score(y_val, val_proba)
    val_prec   = precision_score(y_val, val_pred, zero_division=0)
    val_recall = recall_score(y_val, val_pred, zero_division=0)

    print(f"\n── Validation (2021–2022) ──────────────────────────────────────")
    print(f"   ROC-AUC: {val_auc:.4f}  |  Avg-Precision: {val_ap:.4f}")
    print(f"   Precision: {val_prec:.4f}  |  Recall: {val_recall:.4f}  (threshold={threshold})")

    val_exp = simulate_expectancy(val_proba, val_df["fwd_5d"], threshold)
    print(f"   Simulated expectancy: {val_exp['expectancy']*100:.2f}%  "
          f"(win_rate={val_exp['win_rate']*100:.1f}%  trades={val_exp['trades']})")

    # ── Test evaluation (out-of-sample 2023+) ─────────────────────────────────
    test_proba  = model.predict_proba(X_test)[:, 1]
    test_pred   = (test_proba >= threshold).astype(int)
    test_auc    = roc_auc_score(y_test, test_proba)
    test_ap     = average_precision_score(y_test, test_proba)
    test_prec   = precision_score(y_test, test_pred, zero_division=0)
    test_recall = recall_score(y_test, test_pred, zero_division=0)

    print(f"\n── Test (2023–2025, out-of-sample) ────────────────────────────")
    print(f"   ROC-AUC: {test_auc:.4f}  |  Avg-Precision: {test_ap:.4f}")
    print(f"   Precision: {test_prec:.4f}  |  Recall: {test_recall:.4f}  (threshold={threshold})")

    test_exp = simulate_expectancy(test_proba, test_df["fwd_5d"], threshold)
    print(f"   Simulated expectancy: {test_exp['expectancy']*100:.2f}%  "
          f"(win_rate={test_exp['win_rate']*100:.1f}%  trades={test_exp['trades']})")

    fee_floor_pct = ROUND_TRIP_FEE * 100
    if test_exp["expectancy"] > ROUND_TRIP_FEE:
        print(f"\n   ✅ Out-of-sample expectancy ({test_exp['expectancy']*100:.2f}%) "
              f"> fee floor ({fee_floor_pct:.3f}%) — model has genuine edge")
    else:
        print(f"\n   ⚠  Out-of-sample expectancy ({test_exp['expectancy']*100:.2f}%) "
              f"≤ fee floor ({fee_floor_pct:.3f}%)")
        print("   Likely causes (in order): class threshold too low (try --threshold 0.60),")
        print("   too few positives (try label_generator with wider window),")
        print("   or noise from illiquid symbols (tighten is_tradable filter).")

    # ── Feature importance ────────────────────────────────────────────────────
    importance = dict(zip(FEATURE_COLS, model.feature_importances_.tolist()))
    importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
    print("\n── Feature importance (gain) ───────────────────────────────────")
    for feat, imp in list(importance.items())[:10]:
        print(f"   {feat:<22} {imp:.4f}")

    # ── Save model ────────────────────────────────────────────────────────────
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\n[save] Model → {MODEL_PATH}")

    # ── Save report ───────────────────────────────────────────────────────────
    report = {
        "feature_cols":       FEATURE_COLS,
        "label_col":          LABEL_COL,
        "hold_days":          HOLD_DAYS,
        "round_trip_fee":     ROUND_TRIP_FEE,
        "decision_threshold": threshold,
        "best_iteration":     best_iter,
        "scale_pos_weight":   round(scale_pos_weight, 4),
        "split": {
            "train_rows": len(train_df),
            "val_rows":   len(val_df),
            "test_rows":  len(test_df),
            "train_end":  TRAIN_END,
            "val_start":  VAL_START, "val_end": VAL_END,
            "test_start": TEST_START,
        },
        "val_metrics":  {
            "roc_auc":   round(val_auc, 4),
            "avg_prec":  round(val_ap, 4),
            "precision": round(val_prec, 4),
            "recall":    round(val_recall, 4),
        },
        "val_expectancy":  val_exp,
        "test_metrics": {
            "roc_auc":   round(test_auc, 4),
            "avg_prec":  round(test_ap, 4),
            "precision": round(test_prec, 4),
            "recall":    round(test_recall, 4),
        },
        "test_expectancy": test_exp,
        "feature_importance": importance,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"[save] Report → {REPORT_PATH}")


def main() -> None:
    _check_xgboost()

    parser = argparse.ArgumentParser(description="Train Ziidi XGBoost signal model")
    parser.add_argument("--labelled", type=Path,
                        default=_ROOT / "training" / "nse_labelled.parquet")
    parser.add_argument("--threshold", type=float, default=0.50,
                        help="Decision threshold for precision/recall and expectancy simulation")
    args = parser.parse_args()

    if not args.labelled.exists():
        print(f"[error] Labelled dataset not found: {args.labelled}")
        print("        Run label_generator.py first.")
        sys.exit(1)

    train(args.labelled, args.threshold)


if __name__ == "__main__":
    main()

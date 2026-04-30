#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Ziidi ML Training Pipeline
# Runs in order: labels → train → backtest → param_optimize
#
# Usage (from project root, using Git Bash / WSL / Linux / macOS):
#   chmod +x training/05_run_pipeline.sh
#   ./training/05_run_pipeline.sh
#
# Optional flags forwarded to each step:
#   ./training/05_run_pipeline.sh --threshold 65 --update-system-params
#
# Windows PowerShell alternative:
#   python training/label_generator.py
#   python training/train_model.py
#   python training/backtest.py
#   python training/param_optimizer.py --update-system-params
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail    # exit on first error, treat unset vars as errors

# ── Resolve project root from script location ─────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# ── Colour helpers ────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
step() { echo -e "\n${GREEN}══ $1 ${NC}"; }
warn() { echo -e "${YELLOW}⚠  $1${NC}"; }
fail() { echo -e "${RED}✗  $1${NC}"; exit 1; }

# ── Parse forwarded args ──────────────────────────────────────────────────────
THRESHOLD="50"
UPDATE_PARAMS=""
EXTRA_BACKTEST_ARGS=""

for arg in "$@"; do
    case "$arg" in
        --threshold=*)  THRESHOLD="${arg#*=}" ;;
        --threshold)    shift; THRESHOLD="$1" ;;
        --update-system-params) UPDATE_PARAMS="--update-system-params" ;;
    esac
done

# ── Prerequisite check ────────────────────────────────────────────────────────
step "Prerequisites"
python -c "import xgboost, sklearn, pyarrow, pandas, numpy" 2>/dev/null || {
    warn "Missing dependencies.  Installing …"
    pip install xgboost scikit-learn pyarrow pandas numpy --quiet
}
echo "  Dependencies OK"

# Check nse_master data exists
if [ ! -d "nse_master" ] && [ ! -f "nse_master/nse_master.parquet" ]; then
    fail "nse_master/ directory not found.\n   Place NSE historical data there before running the pipeline."
fi

# ── Step 1: Label generation ──────────────────────────────────────────────────
step "Step 1/4 — Label Generator"
python training/label_generator.py
echo "  ✓ nse_labelled.parquet written"

# ── Step 2: Model training ────────────────────────────────────────────────────
step "Step 2/4 — Model Training  (threshold=${THRESHOLD})"
python training/train_model.py --threshold "$THRESHOLD"
echo "  ✓ ziidi_signal_model.pkl written"

# ── Step 3: Backtest (out-of-sample 2023+) ────────────────────────────────────
step "Step 3/4 — Backtest  (test set ≥ 2023)"
python training/backtest.py --threshold "$THRESHOLD"
echo "  ✓ backtest/results_summary.json written"

# ── Check backtest result before optimising params ───────────────────────────
EXPECTANCY=$(python -c "
import json, sys
try:
    d = json.load(open('training/backtest/results_summary.json'))
    print(d.get('expectancy', 0))
except Exception as e:
    print(0)
")
FEE=0.03874

HAS_EDGE=$(python -c "print('yes' if float('${EXPECTANCY}') > ${FEE} else 'no')")
if [ "$HAS_EDGE" = "yes" ]; then
    echo -e "  ${GREEN}✅ Out-of-sample expectancy ($(python -c "print(f'{float(\"${EXPECTANCY}\")*100:.3f}%')")) > fee floor — genuine edge confirmed${NC}"
else
    warn "Out-of-sample expectancy ($(python -c "print(f'{float(\"${EXPECTANCY}\")*100:.3f}%')")) ≤ fee floor ($(python -c "print(f'{${FEE}*100:.3f}%')"))"
    warn "Model may not have edge.  Proceeding with param optimisation anyway."
fi

# ── Step 4: Parameter optimisation ───────────────────────────────────────────
step "Step 4/4 — Parameter Optimizer"
python training/param_optimizer.py $UPDATE_PARAMS
echo "  ✓ optimal_params.json written"

if [ -n "$UPDATE_PARAMS" ]; then
    echo "  ✓ system_params.json updated with optimal values"
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}══ Pipeline complete ══════════════════════════════════════════════${NC}"
echo ""
echo "  Outputs:"
echo "    training/nse_labelled.parquet      ← labelled dataset"
echo "    training/ziidi_signal_model.pkl    ← trained model  (loaded by core/ml_scorer.py)"
echo "    training/model_report.json         ← evaluation metrics + feature importance"
echo "    training/backtest/results_summary.json  ← out-of-sample backtest results"
echo "    training/backtest/trades.csv       ← per-trade log"
echo "    training/backtest/equity_curves.csv"
echo "    training/optimal_params.json       ← best parameter set"
echo "    training/grid_search_log.csv       ← full grid search results"
echo ""
echo "  Next step: restart the live system — core/ml_scorer.py loads the model automatically."
echo "  If you passed --update-system-params, system_params.json now reflects optimal values."

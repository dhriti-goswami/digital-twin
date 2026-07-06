#!/usr/bin/env bash
# Full end-to-end pipeline: generate → train → evaluate
# Usage:
#   ./scripts/run_pipeline.sh               # default: 14 days, 100 epochs
#   ./scripts/run_pipeline.sh --epochs 50   # faster test run
#   ./scripts/run_pipeline.sh --shap        # include SHAP analysis

set -e
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
source .venv/bin/activate

EPOCHS=100
SHAP_FLAG=""
SKIP_GEN=0

for arg in "$@"; do
  case $arg in
    --epochs=*)  EPOCHS="${arg#*=}" ;;
    --epochs)    shift; EPOCHS="$1" ;;
    --shap)      SHAP_FLAG="--shap" ;;
    --skip-gen)  SKIP_GEN=1 ;;
  esac
done

echo "========================================================"
echo " DIABETES DIGITAL TWIN — FULL PIPELINE"
echo "========================================================"
echo " Epochs:   $EPOCHS"
echo " SHAP:     ${SHAP_FLAG:-off}"
echo ""

# 1. Generate training data (skip if already done)
if [ "$SKIP_GEN" -eq 0 ] && [ ! -f "data/raw/simulated/adult_001.csv" ]; then
  echo "[1/3] Generating ODE training data ..."
  python scripts/generate_training_data.py
else
  echo "[1/3] Data already generated (use --skip-gen=0 to regenerate)"
fi

# 2. Train model
echo ""
echo "[2/3] Training GlucoseTransformer (clinical penalty loss) ..."
python scripts/train_ode.py --epochs "$EPOCHS" --batch-size 128 $SHAP_FLAG

# 3. Evaluate
echo ""
echo "[3/3] Evaluating model ..."
python scripts/evaluate.py $SHAP_FLAG

echo ""
echo "========================================================"
echo " DONE — results in ./results/"
echo "========================================================"

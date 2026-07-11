#!/usr/bin/env bash
# Train the remaining models on the full 10-crop dataset, sequentially.
# XGBoost is already done. Order: fast → slow.
set -uo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE
cd "$(dirname "$0")/.."

for m in dlinear itransformer lstm patchtst tft; do
  echo "================ TRAIN $m ($(date +%H:%M:%S)) ================"
  rm -f "results/checkpoints/${m}.pt" "results/checkpoints/${m}.pipeline.json" 2>/dev/null
  python scripts/03_train.py --model "$m" 2>&1 | tail -4
done
echo "================ ALL MULTI-CROP TRAINING DONE ($(date +%H:%M:%S)) ================"

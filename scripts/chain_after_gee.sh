#!/usr/bin/env bash
# Overnight chain: wait for GEE download to finish, then run the entire
# downstream pipeline so you wake up to a complete LAI-enriched result set.
#
# Steps:
#   1. Wait for `data/raw/gee/gee_satellite.parquet` (the GEE driver's final
#      output) AND for no active `01_download_data --source gee` process.
#   2. Re-preprocess: rebuilds `data/processed/dataset.parquet` with LAI.
#   3. Re-train all 6 models on the LAI-enriched data (sequentially).
#   4. Re-run per-(model, crop) summary + per-horizon eval (refreshes the
#      In-Season page).
#   5. Per-district spatial eval (refreshes the Spatial page View 1).
#   6. LOO XGBoost sweep (lights up the Spatial page View 2).
#
# All wall-clock caps already enforced in the trainers; the deep models
# stop cleanly at 60 min so the whole chain is bounded.
set -uo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE
cd "$(dirname "$0")/.."
# use the project environment if it exists; otherwise the Python that is already active
[ -f "${VENV:-.venv}/bin/activate" ] && source "${VENV:-.venv}/bin/activate"

GEE_PARQUET="data/raw/gee/gee_satellite.parquet"
LOG="/tmp/chain_after_gee.log"

stamp() { date +"%H:%M:%S"; }

echo "[$(stamp)] chain started — waiting for GEE to finish"
while true; do
  if [ -f "$GEE_PARQUET" ] && ! pgrep -f "01_download_data.*gee" > /dev/null; then
    echo "[$(stamp)] GEE complete — parquet present, no GEE process"
    break
  fi
  sleep 120
done

echo "[$(stamp)] step 1 — preprocess"
python scripts/02_preprocess.py 2>&1 | tail -3
echo "[$(stamp)]   preprocess done"

# ── Retrain ──
echo "[$(stamp)] step 2a — retrain xgboost (fast)"
rm -f results/checkpoints/xgboost.ubj results/checkpoints/xgboost.pipeline.json
python scripts/03_train.py --model xgboost 2>&1 | tail -3

echo "[$(stamp)] step 2b — retrain deep models (sequential via train_all_multicrop)"
./scripts/train_all_multicrop.sh

echo "[$(stamp)] step 3 — per-crop summary + per-horizon eval"
rm -f results/metrics/multicrop_horizons.csv \
      results/metrics/multicrop_summary.csv \
      results/metrics/multicrop_leaderboard.csv
python scripts/05b_eval_multicrop.py 2>&1 | tail -5
python scripts/05c_eval_horizons.py 2>&1 | tail -5

echo "[$(stamp)] step 4 — per-district spatial eval"
rm -f results/metrics/spatial_per_district.csv
python scripts/06_spatial_eval.py 2>&1 | tail -3

echo "[$(stamp)] step 5 — LOO XGBoost (50 folds × 10 crops)"
rm -f results/metrics/loo_xgboost.csv
python scripts/06b_loo_xgboost.py 2>&1 | tail -5

echo "[$(stamp)] chain finished — all downstream artefacts refreshed with LAI"

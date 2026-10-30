#!/usr/bin/env bash
#SBATCH --job-name=cyp-rolling
#SBATCH --partition=h100
#SBATCH --qos=computervision
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --array=1-80%8
#SBATCH --output=slurm_logs/rolling-%A_%a.out
#SBATCH --error=slurm_logs/rolling-%A_%a.err
# Yearly rolling test. For harvest year Y: train <= Y-2, early-stop on Y-1,
# score on Y (scripts/30_eval_rolling.py). Arms differ only in drop_features:
#   drv   = with the ERA5-Land/fPAR drivers,  nodrv = without.
# Tasks  1-60: iTransformer, arm x year(2016-2025) x seed(42,43,44)
# Tasks 61-80: XGBoost,      arm x year(2016-2025), seed 42 (deterministic)
set -euo pipefail
source "${VENV:-.venv}/bin/activate"   # project environment (requirements.txt)
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CROP_YIELD_DEVICE=cuda
cd "${SLURM_SUBMIT_DIR:-.}"   # submit from the project root
i=$((SLURM_ARRAY_TASK_ID - 1))
ARMS=(drv nodrv)
if [ "$i" -lt 60 ]; then
  MODEL=itransformer; SEEDS=(42 43 44)
  ARM=${ARMS[$((i / 30))]}; YEAR=$((2016 + (i % 30) / 3)); SEED=${SEEDS[$((i % 3))]}
else
  j=$((i - 60)); MODEL=xgboost
  ARM=${ARMS[$((j / 10))]}; YEAR=$((2016 + j % 10)); SEED=42
fi
NAME="${MODEL}_roll${YEAR}_${ARM}_s${SEED}"
echo "=== $(date '+%F %T') START ${NAME} ==="; hostname; nvidia-smi -L | head -1
python -u scripts/03_train.py --model "${MODEL}" --config-name "${MODEL}_roll_${ARM}" \
    --suffix "_roll${YEAR}_${ARM}_s${SEED}" --test-year "${YEAR}" --seed "${SEED}"
test -s "results/checkpoints/${NAME}.pipeline.json"
echo "=== $(date '+%F %T') DONE ${NAME} ==="

#!/usr/bin/env bash
#SBATCH --job-name=cyp-forecast
#SBATCH --partition=h100
#SBATCH --qos=computervision
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --array=1-3
#SBATCH --output=slurm_logs/forecast-%A_%a.out
#SBATCH --error=slurm_logs/forecast-%A_%a.err
# Prospective forecast models (paper/preregistration_2026-09-26.md §7).
# Arm D, selected by the registered rule; same protocol as the rolling test:
# train <= TARGET-2, validate TARGET-1, 3 seeds. TARGET has no labels yet, so
# the test split is empty and nothing is scored here.
# Usage: sbatch --export=ALL,TARGET=2026 scripts/sbatch_forecast.sh
#        (CROP_YIELD_SMOKE=1 for a 1-epoch check)
set -euo pipefail
source "${VENV:-.venv}/bin/activate"   # project environment (requirements.txt)
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CROP_YIELD_DEVICE=cuda
cd "${SLURM_SUBMIT_DIR:-.}"   # submit from the project root
TARGET=${TARGET:?set TARGET=<harvest year>}
SEED=$((41 + SLURM_ARRAY_TASK_ID))
TAG=${CROP_YIELD_SMOKE:+_smoke}
NAME="itransformer_preD_fc${TARGET}_s${SEED}${TAG}"
echo "=== $(date '+%F %T') START ${NAME} ==="; hostname; nvidia-smi -L | head -1
python -u scripts/03_train.py --model itransformer --config-name itransformer_preD \
    --suffix "_preD_fc${TARGET}_s${SEED}${TAG}" --test-year "${TARGET}" --seed "${SEED}"
test -s "results/checkpoints/${NAME}.pt"
echo "=== $(date '+%F %T') DONE ${NAME} ==="

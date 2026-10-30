#!/usr/bin/env bash
#SBATCH --job-name=cyp-v4T
#SBATCH --partition=standard
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --array=1-30%10
#SBATCH --output=slurm_logs/v4T-%A_%a.out
#SBATCH --error=slurm_logs/v4T-%A_%a.err
# Pre-registration v4 (paper/preregistration_2026-09-29_v4.md) arm T: 10 years x 3 seeds.
# Task i (0-based): year = 2016 + i/3, seed = 42 + i%3. All on L40.
set -euo pipefail
source "${VENV:-.venv}/bin/activate"   # project environment (requirements.txt)
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CROP_YIELD_DEVICE=cuda
cd "${SLURM_SUBMIT_DIR:-.}"   # submit from the project root
i=$((SLURM_ARRAY_TASK_ID - 1))
ARM=T
YEAR=$((2016 + i / 3)); SEED=$((42 + i % 3))
TAG=${CROP_YIELD_SMOKE:+_smoke}
NAME="itransformer_v4${ARM}_roll${YEAR}_s${SEED}${TAG}"
echo "=== $(date '+%F %T') START ${NAME} ==="; hostname; nvidia-smi -L | head -1
python -u scripts/03_train.py --model itransformer --config-name "itransformer_v4${ARM}" \
    --suffix "_v4${ARM}_roll${YEAR}_s${SEED}${TAG}" --test-year "${YEAR}" --seed "${SEED}"
test -s "results/checkpoints/${NAME}.pt"
echo "=== $(date '+%F %T') DONE ${NAME} ==="

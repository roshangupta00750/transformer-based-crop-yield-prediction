#!/usr/bin/env bash
#SBATCH --job-name=cyp-prereg
#SBATCH --partition=h100
#SBATCH --qos=computervision
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --array=1-120%8
#SBATCH --output=slurm_logs/prereg-%A_%a.out
#SBATCH --error=slurm_logs/prereg-%A_%a.err
# Pre-registered experiment (paper/preregistration_2026-09-26.md).
# 4 arms x 10 harvest years (2016-2025) x 3 seeds = 120 runs.
# Task i (0-based): arm = ABCD[i/30], year = 2016 + (i%30)/3, seed = 42+(i%3)
set -euo pipefail
source "${VENV:-.venv}/bin/activate"   # project environment (requirements.txt)
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CROP_YIELD_DEVICE=cuda
cd "${SLURM_SUBMIT_DIR:-.}"   # submit from the project root
i=$((SLURM_ARRAY_TASK_ID - 1))
ARMS=(A B C D); ARM=${ARMS[$((i / 30))]}
YEAR=$((2016 + (i % 30) / 3)); SEED=$((42 + i % 3))
NAME="itransformer_pre${ARM}_roll${YEAR}_s${SEED}"
echo "=== $(date '+%F %T') START ${NAME} ==="; hostname; nvidia-smi -L | head -1
python -u scripts/03_train.py --model itransformer --config-name "itransformer_pre${ARM}" \
    --suffix "_pre${ARM}_roll${YEAR}_s${SEED}" --test-year "${YEAR}" --seed "${SEED}"
test -s "results/checkpoints/${NAME}.pt"
echo "=== $(date '+%F %T') DONE ${NAME} ==="

#!/usr/bin/env bash
#SBATCH --job-name=cyp-v4F
#SBATCH --partition=standard
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=05:00:00
#SBATCH --array=1-30%10
#SBATCH --output=slurm_logs/v4F-%A_%a.out
#SBATCH --error=slurm_logs/v4F-%A_%a.err
# Pre-registration v4 §2.2 arm F: per-crop fine-tuning of arm T, one fold per task. All on L40.
set -euo pipefail
source "${VENV:-.venv}/bin/activate"   # project environment (requirements.txt)
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CROP_YIELD_DEVICE=cuda
cd "${SLURM_SUBMIT_DIR:-.}"   # submit from the project root
i=$((SLURM_ARRAY_TASK_ID - 1)); YEAR=$((2016 + i / 3)); SEED=$((42 + i % 3))
echo "=== $(date '+%F %T') START v4F roll${YEAR} s${SEED} ==="; hostname; nvidia-smi -L | head -1
python -u scripts/47_finetune_per_crop.py --test-year "${YEAR}" --seed "${SEED}"
echo "=== $(date '+%F %T') DONE ==="

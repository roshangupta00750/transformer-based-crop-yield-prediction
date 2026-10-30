#!/usr/bin/env bash
#SBATCH --job-name=cyp-v16-gbm
#SBATCH --partition=standard
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --array=1-16
#SBATCH --output=slurm_logs/v16-%x-%A_%a.out
#SBATCH --error=slurm_logs/v16-%x-%A_%a.err
# Pre-registration v16 §2 arm G16 (gradient boosting on Q12's inputs), folds 2010-2025. L40.

set -euo pipefail
source "${VENV:-.venv}/bin/activate"   # project environment (requirements.txt)
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=16
cd "${SLURM_SUBMIT_DIR:-.}"   # submit from the project root
YEAR=$((2009 + SLURM_ARRAY_TASK_ID))
echo "=== $(date '+%F %T') G16 ${YEAR} ==="; hostname; nvidia-smi -L | head -1
python -u scripts/68_gbm_ablation.py --test-year "${YEAR}" --device cuda
test -s "results/metrics/v16_gbm/G16_${YEAR}.parquet"
echo "=== $(date '+%F %T') DONE ==="

#!/usr/bin/env bash
#SBATCH --job-name=tft-percrop
#SBATCH --partition=h100
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:15:00
#SBATCH --array=1-10
#SBATCH --output=slurm_logs/tft-percrop-%A_%a.out
#SBATCH --error=slurm_logs/tft-percrop-%A_%a.err

# Option C: per-crop TFT retrain. One task per crop, one H100 each.
# Saves as results/checkpoints/tft_<crop>.{pt,pipeline.json}. The dashboard
# _predict_test_set_tft path already prefers tft_<crop> over shared tft.pt.
#
# Wall-clock ceiling: 2h per crop (typically completes in 20-60 min since each
# per-crop dataset is ~10% of the 10-crop shared dataset). Requires nothing
# more than 03_train.py's --crop + --suffix flags — no code changes upstream.

set -euo pipefail
mkdir -p slurm_logs
source "${VENV:-.venv}/bin/activate"   # project environment (requirements.txt)

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export CROP_YIELD_DEVICE=cuda

cd "${SLURM_SUBMIT_DIR:-.}"   # submit from the project root

CROPS=(winter_wheat winter_barley rye triticale winter_rapeseed
       summer_barley oats silage_maize potatoes sugar_beet)
CROP="${CROPS[$((SLURM_ARRAY_TASK_ID - 1))]}"

echo "=== $(date '+%F %H:%M:%S') START crop=${CROP} task=${SLURM_ARRAY_TASK_ID} ==="
hostname
nvidia-smi -L | head -1

python -u scripts/03_train.py --model tft --crop "${CROP}" --suffix "_${CROP}"

echo "=== $(date '+%F %H:%M:%S') DONE  crop=${CROP} ==="
ls -lh "results/checkpoints/tft_${CROP}.pipeline.json"
ls -lh "results/checkpoints/tft_${CROP}.pt" 2>/dev/null || \
  echo "no .pt (unexpected for TFT)"
test -s "results/checkpoints/tft_${CROP}.pipeline.json"

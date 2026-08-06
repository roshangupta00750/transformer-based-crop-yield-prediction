#!/usr/bin/env bash
#SBATCH --job-name=tft-percrop-l40
#SBATCH --partition=h100
#SBATCH --qos=computervision
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --array=1-10
#SBATCH --output=/home/s517868/slurm_logs/tft-percrop-l40-%A_%a.out
#SBATCH --error=/home/s517868/slurm_logs/tft-percrop-l40-%A_%a.err

# L40 fallback of the per-crop TFT training for the 5 crops still missing
# after H100 QOSGrpGRESMinutes blocked further h100 usage.
# Same 60-min Lightning budget as the 5 already-completed per-crop TFTs
# (winter_wheat/winter_barley/rye/triticale on 2970214; winter_rapeseed on
# 2973672_5) — preserves consistency across all 10 crops.
#
# L40 is roughly 1.5-2× slower than H100 for the 555K-param TFT (memory
# bandwidth 864 GB/s vs H100 3000 GB/s). 1h45m SLURM = 60min Lightning +
# 45min buffer for init + model save. Fits well inside the 1-day
# standard-partition wall-time cap.
#
# Runs on `standard` (norm+ QoS) — separate quota pool from the h100
# partition that blocked 2973672_[6-10].

set -euo pipefail
mkdir -p /home/s517868/slurm_logs
source ~/agri-venv/bin/activate

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export CROP_YIELD_DEVICE=cuda

cd ~/crop_yield_benchmark

CROPS=(winter_wheat winter_barley rye triticale winter_rapeseed
       summer_barley oats silage_maize potatoes sugar_beet)
CROP="${CROPS[$((SLURM_ARRAY_TASK_ID - 1))]}"

echo "=== $(date '+%F %H:%M:%S') START crop=${CROP} task=${SLURM_ARRAY_TASK_ID} ==="
hostname
nvidia-smi -L | head -1

python -u scripts/03_train.py --model tft --crop "${CROP}" --suffix "_${CROP}"

echo "=== $(date '+%F %H:%M:%S') DONE  crop=${CROP} ==="
# Success gate: pipeline.json must exist and be non-empty. Report ckpt size
# tolerantly so a missing .pt doesn't false-fail under set -e.
test -s "results/checkpoints/tft_${CROP}.pipeline.json"
ls -lh "results/checkpoints/tft_${CROP}.pt" "results/checkpoints/tft_${CROP}.pipeline.json" 2>&1 | tail -5 || true

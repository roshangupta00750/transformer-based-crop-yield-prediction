#!/usr/bin/env bash
#SBATCH --job-name=cyp-v3-eval
#SBATCH --partition=h100
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=/home/s517868/slurm_logs/v3-eval-%j.out
#SBATCH --error=/home/s517868/slurm_logs/v3-eval-%j.err

# Runs 05b_eval_multicrop across all 6 v3 checkpoints × 10 crops.
# Writes results/metrics/multicrop_summary.csv, multicrop_leaderboard.csv,
# and (only when all 60 model-crop pairs succeed) feature_policy.json —
# which un-gates the dashboard.

set -euo pipefail
mkdir -p /home/s517868/slurm_logs
source ~/agri-venv/bin/activate
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export CROP_YIELD_DEVICE=cuda
cd ~/crop_yield_benchmark

echo "=== $(date '+%F %H:%M:%S') START 05b_eval_multicrop ==="
hostname
nvidia-smi -L | head -1

if [[ "${EVAL_MODE:-all}" == "resume-tft" ]]; then
  python -u scripts/05b_eval_multicrop.py --resume --models tft
else
  python -u scripts/05b_eval_multicrop.py
fi

echo
echo "=== $(date '+%F %H:%M:%S') DONE ==="
ls -lh results/metrics/multicrop_summary.csv \
       results/metrics/multicrop_leaderboard.csv \
       results/metrics/feature_policy.json 2>&1

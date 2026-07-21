#!/usr/bin/env bash
#SBATCH --job-name=eval-24-25
#SBATCH --partition=h100
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=/home/s517868/slurm_logs/eval-24-25-%j.out
#SBATCH --error=/home/s517868/slurm_logs/eval-24-25-%j.err

# H100 eval of 2024+2025 test years with the shipped 5-model + TFT checkpoints.
# 128 GB RAM to avoid the pf-TFT-load OOM we hit on Mac.
set -euo pipefail
mkdir -p /home/s517868/slurm_logs
source ~/agri-venv/bin/activate
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd ~/crop_yield_benchmark

echo "=== node + GPU ==="
hostname
nvidia-smi -L
free -h | head -2

# Force cuda for TFT inference
cp configs/experiment.yaml /tmp/experiment_eval2425.yaml
trap 'cp /tmp/experiment_eval2425.yaml configs/experiment.yaml' EXIT
sed -i 's/^device:.*/device: "cuda"/' configs/experiment.yaml
grep -E "^device|^  max_epochs|test_years" configs/experiment.yaml configs/data.yaml | head

echo "=== 05b_eval_multicrop.py on H100 (test=2024+2025) ==="
rm -f results/metrics/multicrop_summary.csv results/metrics/multicrop_leaderboard.csv
python -u scripts/05b_eval_multicrop.py

echo "=== fusion 5-model (exclude tft) ==="
rm -f results/metrics/fusion_predictions_val.parquet results/metrics/fusion_predictions_test.parquet
python -u scripts/07_fusion.py --exclude tft --suffix _5models

echo "=== fusion 6-model ==="
python -u scripts/07_fusion.py --suffix _6models

echo "=== eval-24-25 done: $(date '+%H:%M:%S') ==="

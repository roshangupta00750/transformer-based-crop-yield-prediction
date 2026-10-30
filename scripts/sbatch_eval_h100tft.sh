#!/usr/bin/env bash
#SBATCH --job-name=eval-h100tft
#SBATCH --partition=standard
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=slurm_logs/eval-%j.out
#SBATCH --error=slurm_logs/eval-%j.err

set -euo pipefail

mkdir -p slurm_logs
source "${VENV:-.venv}/bin/activate"   # project environment (requirements.txt)
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

cd "${SLURM_SUBMIT_DIR:-.}"   # submit from the project root

echo "=== node + GPU ==="
hostname
nvidia-smi -L
echo

# Point tft.pt at the FULL H100 auto-checkpoint (has all pf keys) — not my
# stripped 2.4M version (which was for the Mac load-hack attempt).
cp results/logs/tft/version_0/checkpoints/epoch=1-step=100380.ckpt \
   results/checkpoints/tft.pt
ls -lh results/checkpoints/tft.pt

# force cuda device (this node has GPU)
sed -i 's/^device:.*/device: "cuda"    # eval on GPU/' configs/experiment.yaml

echo "=== full multicrop eval (all 6 models, H100-trained tft.pt) ==="
python -u scripts/05b_eval_multicrop.py 2>&1 | tail -80

echo
echo "=== NEW multicrop_leaderboard.csv ==="
cat results/metrics/multicrop_leaderboard.csv

echo
echo "=== NEW TFT per-crop ==="
grep "^tft," results/metrics/multicrop_summary.csv

echo
echo "=== eval done: $(date '+%H:%M:%S') ==="

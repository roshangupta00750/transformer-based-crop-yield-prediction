#!/usr/bin/env bash
#SBATCH --job-name=tft-preflight
#SBATCH --partition=standard
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=/home/s517868/slurm_logs/preflight-%j.out
#SBATCH --error=/home/s517868/slurm_logs/preflight-%j.err

# Preflight: verify the whole H100 → data-load → TFT-instantiate → 1-epoch
# train → checkpoint-save pipeline works end-to-end, without burning 4h.
# Bracket-sed on configs with EXIT trap so any crash restores originals.
set -euo pipefail

mkdir -p /home/s517868/slurm_logs
source ~/agri-venv/bin/activate
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

cd ~/crop_yield_benchmark

echo "=== node + GPU ==="
hostname
nvidia-smi -L
free -h | head -2
echo

# Snapshot originals and register restore-on-exit
cp configs/models/tft.yaml /tmp/tft_orig.yaml
cp configs/experiment.yaml /tmp/experiment_orig.yaml
trap 'cp /tmp/tft_orig.yaml configs/models/tft.yaml; cp /tmp/experiment_orig.yaml configs/experiment.yaml; echo "  restored configs on EXIT"' EXIT

# Short-cap overrides for preflight only
sed -i 's/^max_epochs:.*/max_epochs: 1               # preflight/' configs/models/tft.yaml
sed -i 's/^early_stopping_patience:.*/early_stopping_patience: 1          # preflight/' configs/models/tft.yaml
sed -i 's/^  max_minutes:.*/  max_minutes: 5           # preflight — 5-min Lightning cap/' configs/experiment.yaml
sed -i 's/^device:.*/device: "cuda"    # preflight/' configs/experiment.yaml

echo "  preflight config overrides applied:"
grep -E "^max_epochs|^early_stopping|^  max_minutes|^device" configs/models/tft.yaml configs/experiment.yaml | sed 's/^/    /'

echo
echo "=== preflight training (max_epochs=1, max_minutes=5) start: $(date '+%H:%M:%S') ==="
rm -f results/checkpoints/tft.pt results/checkpoints/tft.pipeline.json
python -u scripts/03_train.py --model tft

echo
echo "=== checkpoint output ==="
ls -lh results/checkpoints/tft.pt results/checkpoints/tft.pipeline.json 2>&1

echo
echo "=== preflight done: $(date '+%H:%M:%S') ==="
echo "  If you see 'restored configs on EXIT' below, the trap fired cleanly."

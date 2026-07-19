#!/usr/bin/env bash
#SBATCH --job-name=tft-cropyield
#SBATCH --partition=h100
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/home/s517868/slurm_logs/tft-%j.out
#SBATCH --error=/home/s517868/slurm_logs/tft-%j.err

set -euo pipefail

mkdir -p /home/s517868/slurm_logs
source ~/agri-venv/bin/activate
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

cd ~/crop_yield_benchmark

echo "=== node + GPU ==="
hostname
nvidia-smi -L
free -h
echo

# Override experiment.yaml to cluster regime for this run (does not touch Mac)
sed -i 's/^device:.*/device: "cuda"    # H100 cluster/' configs/experiment.yaml
sed -i 's/^  max_minutes:.*/  max_minutes: 480    # H100 lets us fit ~6-8 epochs in 8 h/' configs/experiment.yaml
grep -E "^device|max_epochs|early_stopping_patience|max_minutes" configs/experiment.yaml | sed 's/^/  /'

echo
echo "=== TFT training start: $(date '+%H:%M:%S') ==="
rm -f results/checkpoints/tft.pt results/checkpoints/tft.pipeline.json
python -u scripts/03_train.py --model tft

echo
echo "=== checkpoint ==="
ls -lh results/checkpoints/tft.pt results/checkpoints/tft.pipeline.json 2>&1
echo "=== TFT done: $(date '+%H:%M:%S') ==="

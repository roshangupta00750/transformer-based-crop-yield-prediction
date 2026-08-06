#!/usr/bin/env bash
#SBATCH --job-name=abl-axis
#SBATCH --partition=h100
#SBATCH --qos=computervision
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=1-2
#SBATCH --output=/home/s517868/slurm_logs/abl-axis-%A_%a.out
#SBATCH --error=/home/s517868/slurm_logs/abl-axis-%A_%a.err

# Attention-axis ablation for the paper's central claim: for MULTIVARIATE
# crop series, does attention over VARIATES beat attention over TIME?
#
# Both arms are the same architecture at the same parameter count (428,801,
# encoder identical); only iTransformer's defining transpose is removed.
set -euo pipefail
mkdir -p /home/s517868/slurm_logs
source ~/agri-venv/bin/activate
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export CROP_YIELD_DEVICE=cuda
cd ~/crop_yield_benchmark

AXES=(variate temporal)
AXIS="${AXES[$((SLURM_ARRAY_TASK_ID - 1))]}"

echo "=== $(date '+%F %H:%M:%S') attention_axis=${AXIS} ==="
hostname; nvidia-smi -L | head -1

python -u scripts/03_train.py --model itransformer \
  --config-name "itransformer_${AXIS}" --suffix "_axis_${AXIS}"

echo "=== $(date '+%F %H:%M:%S') done ${AXIS} ==="
ls -lh "results/checkpoints/itransformer_axis_${AXIS}.pt"

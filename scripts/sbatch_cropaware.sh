#!/usr/bin/env bash
#SBATCH --job-name=cyp-cropaware
#SBATCH --partition=h100
#SBATCH --qos=computervision
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=1-2
#SBATCH --output=slurm_logs/cropaware-%A_%a.out
#SBATCH --error=slurm_logs/cropaware-%A_%a.err

# Crop-aware iTransformer experiment. Two arms, identical except for the
# static attention tokens (configs/models/itransformer_crop{base,aware}.yaml):
#   1 = cropbase   : today's iTransformer, crop-blind
#   2 = cropaware  : + crop, soil and terrain as attention tokens
# Same GPU type, seed, data and budget for both arms.
set -euo pipefail
mkdir -p slurm_logs
source "${VENV:-.venv}/bin/activate"   # project environment (requirements.txt)
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CROP_YIELD_DEVICE=cuda
cd "${SLURM_SUBMIT_DIR:-.}"   # submit from the project root

ARMS=(cropbase cropaware)
ARM="${ARMS[$((SLURM_ARRAY_TASK_ID - 1))]}"
echo "=== $(date '+%F %T') START arm=${ARM} ==="; hostname; nvidia-smi -L | head -1

python -u scripts/03_train.py --model itransformer \
    --config-name "itransformer_${ARM}" --suffix "_${ARM}"

test -s "results/checkpoints/itransformer_${ARM}.pt"
test -s "results/checkpoints/itransformer_${ARM}.pipeline.json"
echo "=== $(date '+%F %T') DONE arm=${ARM} ==="

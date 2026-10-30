#!/usr/bin/env bash
#SBATCH --job-name=cyp-cropaware-seeds
#SBATCH --partition=h100
#SBATCH --qos=computervision
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=1-4
#SBATCH --output=slurm_logs/cropaware-seeds-%A_%a.out
#SBATCH --error=slurm_logs/cropaware-seeds-%A_%a.err
# Repeat runs of both crop-aware arms with seeds 43 and 44 (seed 42 = the
# original run). Gives run-to-run spread and a 3-seed average per arm.
set -euo pipefail
source "${VENV:-.venv}/bin/activate"   # project environment (requirements.txt)
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CROP_YIELD_DEVICE=cuda
cd "${SLURM_SUBMIT_DIR:-.}"   # submit from the project root
ARMS=(cropbase cropbase cropaware cropaware)
SEEDS=(43 44 43 44)
i=$((SLURM_ARRAY_TASK_ID - 1)); ARM=${ARMS[$i]}; SEED=${SEEDS[$i]}
echo "=== $(date '+%F %T') START arm=${ARM} seed=${SEED} ==="; hostname; nvidia-smi -L | head -1
python -u scripts/03_train.py --model itransformer \
    --config-name "itransformer_${ARM}" --suffix "_${ARM}_s${SEED}" --seed "${SEED}"
test -s "results/checkpoints/itransformer_${ARM}_s${SEED}.pt"
echo "=== $(date '+%F %T') DONE arm=${ARM} seed=${SEED} ==="

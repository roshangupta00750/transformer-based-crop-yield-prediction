#!/usr/bin/env bash
#SBATCH --job-name=cyp-v3-retrain
#SBATCH --partition=standard
#SBATCH --qos=normal
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --array=1-5   # TFT trains per-crop via sbatch_v3_train_tft_percrop_l40.sh
#SBATCH --output=/home/s517868/slurm_logs/v3-retrain-%A_%a.out
#SBATCH --error=/home/s517868/slurm_logs/v3-retrain-%A_%a.err

# Fleet retrain of the 6 model families under feature-policy v3.
# One array task per model, one H100 each, ~10 crops trained sequentially
# within the same task (each model's --crop-agnostic training already spans
# all crops, so this is 6 total training runs, not 60).
#
# Wall-clock: TFT dominates at ~3-8 h; the other 5 finish in ~1-3 h.
# Fleet ends when the slowest array task ends (~8 h ceiling).
#
# Device selection uses the env override added in src/utils.py::pick_device
# so we do not race on configs/experiment.yaml between concurrent tasks.

set -euo pipefail
mkdir -p /home/s517868/slurm_logs
source ~/agri-venv/bin/activate

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export CROP_YIELD_DEVICE=cuda   # honored by src/utils.py::pick_device

cd ~/crop_yield_benchmark

MODELS=(xgboost dlinear lstm patchtst itransformer tft)
MODEL="${MODELS[$((SLURM_ARRAY_TASK_ID - 1))]}"

echo "=== $(date '+%F %H:%M:%S') START model=${MODEL} task=${SLURM_ARRAY_TASK_ID} ==="
hostname
nvidia-smi -L | head -1
free -h | head -2

python -u scripts/03_train.py --model "${MODEL}"

echo "=== $(date '+%F %H:%M:%S') DONE  model=${MODEL} ==="
# XGBoost saves .ubj instead of .pt, so tolerate either — the pipeline.json
# is the required artefact and its presence is asserted below.
ls -lh results/checkpoints/"${MODEL}".pipeline.json
ls -lh results/checkpoints/"${MODEL}".pt results/checkpoints/"${MODEL}".ubj 2>/dev/null | head -5 || true
test -s results/checkpoints/"${MODEL}".pipeline.json  # actual success gate

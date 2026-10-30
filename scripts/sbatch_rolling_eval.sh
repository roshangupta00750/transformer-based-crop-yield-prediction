#!/usr/bin/env bash
#SBATCH --job-name=cyp-rolling-eval
#SBATCH --partition=h100
#SBATCH --qos=computervision
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=slurm_logs/rolling-eval-%j.out
#SBATCH --error=slurm_logs/rolling-eval-%j.err
set -euo pipefail
source "${VENV:-.venv}/bin/activate"   # project environment (requirements.txt)
export PYTHONUNBUFFERED=1 CROP_YIELD_DEVICE=cuda
cd "${SLURM_SUBMIT_DIR:-.}"   # submit from the project root
python -u scripts/30_eval_rolling.py

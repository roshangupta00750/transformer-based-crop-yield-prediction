#!/usr/bin/env bash
#SBATCH --job-name=cyp-cmp-v7-eval
#SBATCH --partition=h100
#SBATCH --qos=computervision
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/cmp-v7-eval-%j.out
#SBATCH --error=slurm_logs/cmp-v7-eval-%j.err
set -euo pipefail
source "${VENV:-.venv}/bin/activate"   # project environment (requirements.txt)
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 CROP_YIELD_DEVICE=cuda
cd "${SLURM_SUBMIT_DIR:-.}"   # submit from the project root
CMP_ARM=v7RF python -u scripts/39_eval_compare.py

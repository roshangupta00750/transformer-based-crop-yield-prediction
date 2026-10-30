#!/usr/bin/env bash
#SBATCH --job-name=cyp-cmp-v15-eval
#SBATCH --partition=standard
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/cmp-v15-eval-%j.out
#SBATCH --error=slurm_logs/cmp-v15-eval-%j.err
set -euo pipefail
source "${VENV:-.venv}/bin/activate"   # project environment (requirements.txt)
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 CROP_YIELD_DEVICE=cuda
cd "${SLURM_SUBMIT_DIR:-.}"   # submit from the project root
CMP_ARM=v15WF CMP_YIELD=data/interim/v5/yield.parquet python -u scripts/39_eval_compare.py

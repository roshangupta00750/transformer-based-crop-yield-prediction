#!/usr/bin/env bash
#SBATCH --job-name=cyp-v9-eval
#SBATCH --partition=standard
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=08:00:00
#SBATCH --output=slurm_logs/v9-eval-%j.out
#SBATCH --error=slurm_logs/v9-eval-%j.err
set -euo pipefail
source "${VENV:-.venv}/bin/activate"   # project environment (requirements.txt)
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 CROP_YIELD_DEVICE=cuda
cd "${SLURM_SUBMIT_DIR:-.}"   # submit from the project root
python -u scripts/54_eval_v9.py

#!/usr/bin/env bash
#SBATCH --job-name=cyp-cropaware-eval
#SBATCH --partition=h100
#SBATCH --qos=computervision
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --output=slurm_logs/cropaware-eval-%j.out
#SBATCH --error=slurm_logs/cropaware-eval-%j.err
# Scores both arms on 2024-25 against the historical average (dt/ha + skill).
set -euo pipefail
source "${VENV:-.venv}/bin/activate"   # project environment (requirements.txt)
export PYTHONUNBUFFERED=1
cd "${SLURM_SUBMIT_DIR:-.}"   # submit from the project root
python -u scripts/24_eval_cropaware.py --arms cropbase,cropaware --seeds "${SEEDS:-}"

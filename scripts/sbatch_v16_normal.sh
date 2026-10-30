#!/usr/bin/env bash
#SBATCH --job-name=cyp-v16-normal
#SBATCH --partition=standard
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/v16-%x-%j.out
#SBATCH --error=slurm_logs/v16-%x-%j.err
# Pre-registration v16 §4.2: fold-2027 district normal (no network). 2025 first as a check
# against the rolling fold-2025 normal.
set -euo pipefail
source "${VENV:-.venv}/bin/activate"   # project environment (requirements.txt)
export PYTHONUNBUFFERED=1
cd "${SLURM_SUBMIT_DIR:-.}"   # submit from the project root
for Y in 2025 2027; do python -u scripts/71_forecast_yieldformer.py --stage normal --year $Y; done

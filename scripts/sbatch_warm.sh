#!/usr/bin/env bash
#SBATCH --job-name=cyp-warm
#SBATCH --partition=cpu_standard
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/warm-%j.out
#SBATCH --error=slurm_logs/warm-%j.err
set -euo pipefail
source "${VENV:-.venv}/bin/activate"   # project environment (requirements.txt)
cd "${SLURM_SUBMIT_DIR:-.}"   # submit from the project root
python -u scripts/warm_frame_cache.py "$1"

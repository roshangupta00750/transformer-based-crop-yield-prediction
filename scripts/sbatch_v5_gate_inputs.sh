#!/usr/bin/env bash
#SBATCH --job-name=cyp-v5-gate
#SBATCH --partition=cpu_standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/v5-gate-%j.out
#SBATCH --error=slurm_logs/v5-gate-%j.err
# Pre-registration v5 §2.1: national indicators + per-fold gate decisions (training years only).
set -euo pipefail
source "${VENV:-.venv}/bin/activate"   # project environment (requirements.txt)
export PYTHONUNBUFFERED=1
cd "${SLURM_SUBMIT_DIR:-.}"   # submit from the project root
python -u scripts/49_national_gate_inputs.py

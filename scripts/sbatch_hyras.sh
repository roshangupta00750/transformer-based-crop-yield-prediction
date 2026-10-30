#!/usr/bin/env bash
#SBATCH --job-name=cyp-hyras
#SBATCH --partition=cpu_standard
#SBATCH --qos=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=slurm_logs/hyras-%j.out
#SBATCH --error=slurm_logs/hyras-%j.err
# Pre-registration v14 §2: DWD HYRAS-DE v6-1 1 km -> district cropland means, 2000-2025.
set -euo pipefail
source "${VENV:-.venv}/bin/activate"   # project environment (requirements.txt)
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
cd "${SLURM_SUBMIT_DIR:-.}"   # submit from the project root
python -u scripts/61_hyras_weather.py --years 2000-2025 --vars tasmax,tasmin,pr,hurs

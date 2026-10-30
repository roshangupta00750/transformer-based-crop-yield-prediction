#!/usr/bin/env bash
#SBATCH --job-name=cyp-v10-build
#SBATCH --partition=cpu_standard
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=06:00:00
#SBATCH --output=slurm_logs/v10-build-%j.out
#SBATCH --error=slurm_logs/v10-build-%j.err
# Pre-registration v10 §2: build dataset_v5 (corrected yields + ERA5-Land weather), then warm the frame cache.
# on 2003-2025, then warm the crop-year frame cache the 150 training runs share.
set -euo pipefail
source "${VENV:-.venv}/bin/activate"   # project environment (requirements.txt)
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=16
cd "${SLURM_SUBMIT_DIR:-.}"   # submit from the project root
python -u scripts/55_build_dataset_v5.py
python -u - <<'PY'
from pathlib import Path
from src.data.dataset import load_modeling_frame
from src.utils import load_data_config
f = load_modeling_frame(Path("data/processed/dataset_v5.parquet"), load_data_config(),
                        window="crop_year", common_sample=True, cache_dir=Path("data/processed/cache"))
print("frame", len(f), f.groupby(["landkreis", "crop", "year"]).ngroups, "crop-years",
      f.year.min(), f.year.max())
PY

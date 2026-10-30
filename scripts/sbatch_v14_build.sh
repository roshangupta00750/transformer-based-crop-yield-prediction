#!/usr/bin/env bash
#SBATCH --job-name=cyp-v14-build
#SBATCH --partition=standard
#SBATCH --qos=normal
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=slurm_logs/v14-build-%j.out
#SBATCH --error=slurm_logs/v14-build-%j.err
# Pre-registration v14: dataset with DWD HYRAS-DE weather, then warm the frame cache.
# Runs on the GPU partition only because the CPU queue is ~18 h deep; the GPU is idle.
set -euo pipefail
source "${VENV:-.venv}/bin/activate"   # project environment (requirements.txt)
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=16
cd "${SLURM_SUBMIT_DIR:-.}"   # submit from the project root
OUT=data/processed/dataset_v5_hyrasH.parquet
python -u scripts/55_build_dataset_v5.py --weather hyras --out "$OUT"
python -u - "$OUT" <<'PY'
import sys
from pathlib import Path
from src.data.dataset import load_modeling_frame
from src.utils import load_data_config, load_model_config
c = load_model_config("itransformer_v14H")
f = load_modeling_frame(Path(sys.argv[1]), load_data_config(), window="crop_year", common_sample=True,
                        cache_dir=Path("data/processed/cache"), phenology=c["phenology"],
                        phenology_version=c["phenology_version"], stage_features=bool(c["stage_features"]))
print("frame", len(f), f.groupby(["landkreis", "crop", "year"]).ngroups, "crop-years")
PY

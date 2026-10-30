#!/usr/bin/env bash
#SBATCH --job-name=cyp-v11-build
#SBATCH --partition=cpu_standard
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --array=1-3
#SBATCH --output=slurm_logs/v11-build-%A_%a.out
#SBATCH --error=slurm_logs/v11-build-%A_%a.err
# Pre-registration v11: leave-one-out datasets (1 = hybrid weather, 2 = MODIS start dating, 3 = name-matched yields)
set -euo pipefail
source "${VENV:-.venv}/bin/activate"   # project environment (requirements.txt)
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=16
cd "${SLURM_SUBMIT_DIR:-.}"   # submit from the project root
case $SLURM_ARRAY_TASK_ID in
  1) A="--weather hybrid --out data/processed/dataset_v5_hybridW.parquet" ;;
  2) A="--modis-start-stamp --out data/processed/dataset_v5_startM.parquet" ;;
  3) A="--yields name --out data/processed/dataset_v5_nameY.parquet" ;;
esac
python -u scripts/55_build_dataset_v5.py $A
OUT=$(echo $A | sed 's/.*--out //')
python -u - "$OUT" <<'PY'
import sys
from pathlib import Path
from src.data.dataset import load_modeling_frame
from src.utils import load_data_config
f = load_modeling_frame(Path(sys.argv[1]), load_data_config(), window="crop_year", common_sample=True,
                        cache_dir=Path("data/processed/cache"))
print("frame", len(f), f.groupby(["landkreis", "crop", "year"]).ngroups, "crop-years")
PY

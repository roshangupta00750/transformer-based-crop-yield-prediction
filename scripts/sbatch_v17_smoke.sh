#!/usr/bin/env bash
#SBATCH --job-name=cyp-v17-smoke
#SBATCH --partition=test,standard,gpu_computervision
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=00:59:00
#SBATCH --output=slurm_logs/v17-smoke-%j.out
#SBATCH --error=slurm_logs/v17-smoke-%j.err
# Pre-registration v17 §4.1: 1-epoch smoke test of both arms on fold 2015, seed 42:
# train -> per-crop fine-tune (2 crops) -> prediction. No result is used.
set -euo pipefail
source "${VENV:-.venv}/bin/activate"   # project environment (requirements.txt)
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CROP_YIELD_DEVICE=cuda CROP_YIELD_SMOKE=1
cd "${SLURM_SUBMIT_DIR:-.}"   # submit from the project root
for A in "T tftnet tftnet_v17T v17T tftnet_v17TF" "P patchtst17 patchtst17_v17P v17P patchtst17_v17PF"; do
  set -- $A
  echo "=== $(date '+%T') smoke $2 ==="
  python -u scripts/03_train.py --model "$2" --config-name "$3" --suffix "_$4_roll2015_s42_smoke" --test-year 2015 --seed 42
  python -u scripts/47_finetune_per_crop.py --test-year 2015 --seed 42 --base "$3" --out "$5"
  python -u scripts/73_eval_v17.py --stage predict --arm "$1" --years 2015 --smoke
done
echo "=== $(date '+%T') SMOKE OK ==="

#!/usr/bin/env bash
#SBATCH --job-name=cyp-prereg-smoke
#SBATCH --partition=h100
#SBATCH --qos=computervision
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --output=slurm_logs/prereg-smoke-%j.out
#SBATCH --error=slurm_logs/prereg-smoke-%j.err
# Gate for the 120 runs: builds both cached frames (calendar and crop-year),
# trains 1 epoch of arm A and arm D for 2016, runs the full evaluator, then
# removes the smoke outputs. The array only starts if this exits 0.
set -euo pipefail
source "${VENV:-.venv}/bin/activate"   # project environment (requirements.txt)
export PYTHONUNBUFFERED=1 CROP_YIELD_DEVICE=cuda
cd "${SLURM_SUBMIT_DIR:-.}"   # submit from the project root
for ARM in A D; do
  CROP_YIELD_SMOKE=1 python -u scripts/03_train.py --model itransformer \
      --config-name "itransformer_pre${ARM}" --suffix "_pre${ARM}_roll2016_s42" \
      --test-year 2016 --seed 42
done
python -u scripts/34_eval_prereg.py
rm -f results/checkpoints/itransformer_pre?_roll2016_s42.* results/metrics/itransformer_pre?_roll2016_s42_meta.json \
      results/metrics/prereg_*.csv results/metrics/prereg_*.json results/metrics/prereg_*.parquet
echo "SMOKE PASSED"

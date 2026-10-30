#!/usr/bin/env bash
#SBATCH --job-name=cyp-rolling-smoke
#SBATCH --partition=h100
#SBATCH --qos=computervision
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=slurm_logs/rolling-smoke-%j.out
#SBATCH --error=slurm_logs/rolling-smoke-%j.err
# Gate: 1-epoch iTransformer (with drivers) + XGBoost (without) for 2016,
# then score them. The 80-task array only starts if this exits 0.
set -euo pipefail
source "${VENV:-.venv}/bin/activate"   # project environment (requirements.txt)
export PYTHONUNBUFFERED=1 CROP_YIELD_DEVICE=cuda CROP_YIELD_SMOKE=1
cd "${SLURM_SUBMIT_DIR:-.}"   # submit from the project root
python -u scripts/03_train.py --model itransformer --config-name itransformer_roll_drv \
    --suffix _roll2016_drv_s42 --test-year 2016 --seed 42
python -u scripts/03_train.py --model xgboost --config-name xgboost_roll_nodrv \
    --suffix _roll2016_nodrv_s42 --test-year 2016 --seed 42
unset CROP_YIELD_SMOKE
python -u scripts/30_eval_rolling.py
rm -f results/checkpoints/*_roll2016_*_s42.* results/metrics/rolling_eval.csv results/metrics/rolling_predictions.parquet
echo "SMOKE PASSED"

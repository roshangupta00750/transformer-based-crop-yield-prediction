#!/usr/bin/env bash
#SBATCH --job-name=cyp-compare
#SBATCH --partition=standard
#SBATCH --qos=computervision
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --array=1-15%8
#SBATCH --output=slurm_logs/compare-%A_%a.out
#SBATCH --error=slurm_logs/compare-%A_%a.err
# Other papers' protocols with arm D (paper/preregistration_2026-09-26.md change log, 2026-09-27).
# Ran on the standard partition (L40/L40S; H100 was full), job 3384140.
# Tasks 1-3 AgroForecast-DE, 4-6 CropFusionNet, 7-15 Srivastava 2022 (2017/18/19 x 3 seeds).
set -euo pipefail
source "${VENV:-.venv}/bin/activate"   # project environment (requirements.txt)
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CROP_YIELD_DEVICE=cuda
cd "${SLURM_SUBMIT_DIR:-.}"   # submit from the project root
i=$((SLURM_ARRAY_TASK_ID - 1)); SEED=$((42 + i % 3))
if   [ $i -lt 3 ]; then TAG=AFDE; SPLIT="--train-end 2019 --val-years 2020,2021,2022 --test-years 2023,2024"
elif [ $i -lt 6 ]; then TAG=CFN;  SPLIT="--train-end 2018 --val-years 2019,2020,2021 --test-years 2022,2023,2024"
else Y=$((2017 + (i - 6) / 3)); TAG=SRV$Y
     SPLIT="--train-end $((Y - 1)) --test-years $Y --val-random-frac 0.15"; fi
SMOKE=${CROP_YIELD_SMOKE:+_smoke}
NAME="itransformer_preD_cmp${TAG}_s${SEED}${SMOKE}"
echo "=== $(date '+%F %T') START ${NAME} (${SPLIT}) ==="; hostname
python -u scripts/03_train.py --model itransformer --config-name itransformer_preD \
    --suffix "_preD_cmp${TAG}_s${SEED}${SMOKE}" --seed "${SEED}" ${SPLIT}
test -s "results/checkpoints/${NAME}.pt"
echo "=== $(date '+%F %T') DONE ${NAME} ==="

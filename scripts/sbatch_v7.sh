#!/usr/bin/env bash
#SBATCH --job-name=cyp-v7
#SBATCH --partition=standard
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=05:00:00
#SBATCH --array=1-66%12
#SBATCH --output=slurm_logs/v7-%x-%A_%a.out
#SBATCH --error=slurm_logs/v7-%x-%A_%a.err
# Pre-registration v7. STAGE=train: all-crop model; STAGE=finetune: per-crop fine-tuning (as v4 F).
# Tasks 1-18: v4 arm T/F hindcast 2010-2015; tasks 19-66: v7 R7, 2010-2025. year/seed from the index. L40.
set -euo pipefail
source "${VENV:-.venv}/bin/activate"   # project environment (requirements.txt)
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CROP_YIELD_DEVICE=cuda
cd "${SLURM_SUBMIT_DIR:-.}"   # submit from the project root
i=$((SLURM_ARRAY_TASK_ID - 1))
if [ $i -lt 18 ]; then CFG=itransformer_v4T; SUF=v4T; OUT=itransformer_v4F; YEAR=$((2010 + i / 3))
else j=$((i - 18)); CFG=itransformer_v7R; SUF=v7R; OUT=itransformer_v7RF; YEAR=$((2010 + j / 3)); fi
SEED=$((42 + i % 3)); TAG=${CROP_YIELD_SMOKE:+_smoke}
echo "=== $(date '+%F %T') ${STAGE} ${CFG} ${YEAR} s${SEED} ==="; hostname; nvidia-smi -L | head -1
if [ "${STAGE}" = train ]; then
  python -u scripts/03_train.py --model itransformer --config-name "${CFG}" \
      --suffix "_${SUF}_roll${YEAR}_s${SEED}${TAG}" --test-year "${YEAR}" --seed "${SEED}"
  test -s "results/checkpoints/${CFG}_roll${YEAR}_s${SEED}${TAG}.pt"
else
  python -u scripts/47_finetune_per_crop.py --test-year "${YEAR}" --seed "${SEED}" --base "${CFG}" --out "${OUT}"
fi
echo "=== $(date '+%F %T') DONE ==="

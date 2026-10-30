#!/usr/bin/env bash
#SBATCH --job-name=cyp-v16-fc
#SBATCH --partition=standard
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=05:00:00
#SBATCH --array=1-3
#SBATCH --output=slurm_logs/v16-%x-%A_%a.out
#SBATCH --error=slurm_logs/v16-%x-%A_%a.err
# Pre-registration v16 §4.1: YieldFormer-DE network for forecast year YEAR (default 2026),
# fold exactly as v12 Q12's rolling folds (train <= YEAR-2, validate YEAR-1, 3 seeds, per-crop
# fine-tuning). YEAR has no labels, so nothing is scored. STAGE=train|finetune. L40.

set -euo pipefail
source "${VENV:-.venv}/bin/activate"   # project environment (requirements.txt)
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CROP_YIELD_DEVICE=cuda
cd "${SLURM_SUBMIT_DIR:-.}"   # submit from the project root
CFG=itransformer_v12Q; SUF=v12Q; OUT=itransformer_v12QF; YEAR=${YEAR:-2026}
SEED=$((41 + SLURM_ARRAY_TASK_ID)); TAG=${CROP_YIELD_SMOKE:+_smoke}
echo "=== $(date '+%F %T') ${STAGE} ${CFG} ${YEAR} s${SEED} ==="; hostname; nvidia-smi -L | head -1
if [ "${STAGE}" = train ]; then
  python -u scripts/03_train.py --model itransformer --config-name "${CFG}" \
      --suffix "_${SUF}_roll${YEAR}_s${SEED}${TAG}" --test-year "${YEAR}" --seed "${SEED}"
  test -s "results/checkpoints/${CFG}_roll${YEAR}_s${SEED}${TAG}.pt"
else
  python -u scripts/47_finetune_per_crop.py --test-year "${YEAR}" --seed "${SEED}" --base "${CFG}" --out "${OUT}"
fi
echo "=== $(date '+%F %T') DONE ==="

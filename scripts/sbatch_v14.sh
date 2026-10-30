#!/usr/bin/env bash
#SBATCH --job-name=cyp-v14
#SBATCH --partition=standard
#SBATCH --qos=normal
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=05:00:00
#SBATCH --array=1-48%12
#SBATCH --output=slurm_logs/v14-%x-%A_%a.out
#SBATCH --error=slurm_logs/v14-%x-%A_%a.err
# Pre-registration v14 arm H14. STAGE=train|finetune. 2010-2025 x 3 seeds. L40, QoS normal.

set -euo pipefail
source "${VENV:-.venv}/bin/activate"   # project environment (requirements.txt)
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CROP_YIELD_DEVICE=cuda
cd "${SLURM_SUBMIT_DIR:-.}"   # submit from the project root
i=$((SLURM_ARRAY_TASK_ID - 1))
CFG=itransformer_v14H; SUF=v14H; OUT=itransformer_v14HF; YEAR=$((2010 + i / 3))
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

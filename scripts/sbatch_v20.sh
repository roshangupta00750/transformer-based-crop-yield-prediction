#!/usr/bin/env bash
#SBATCH --job-name=cyp-v20
#SBATCH --partition=standard,gpu_computervision
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=05:00:00
#SBATCH --array=1-48
#SBATCH --output=slurm_logs/v20-%x-%A_%a.out
#SBATCH --error=slurm_logs/v20-%x-%A_%a.err
# Pre-registration v20 §2: ARM=Q (iTransformer) | T (TFT) | P (PatchTST) + stress inputs; STAGE=train|finetune; 2010-2025 x 3 seeds,
# exactly as v12 Q12 (scripts/sbatch_v12.sh). QoS: account default (computervision).

set -euo pipefail
source "${VENV:-.venv}/bin/activate"   # project environment (requirements.txt)
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CROP_YIELD_DEVICE=cuda
cd "${SLURM_SUBMIT_DIR:-.}"   # submit from the project root
i=$((SLURM_ARRAY_TASK_ID - 1))
case "${ARM}" in
  Q) MODEL=itransformer; CFG=itransformer_v20Q; SUF=v20Q; OUT=itransformer_v20QF ;;
  T) MODEL=tftnet;     CFG=tftnet_v20T;     SUF=v20T; OUT=tftnet_v20TF ;;
  P) MODEL=patchtst17; CFG=patchtst17_v20P; SUF=v20P; OUT=patchtst17_v20PF ;;
esac
YEAR=$((2010 + i / 3)); SEED=$((42 + i % 3)); TAG=${CROP_YIELD_SMOKE:+_smoke}
echo "=== $(date '+%F %T') ${STAGE} ${CFG} ${YEAR} s${SEED} ==="; hostname; nvidia-smi -L | head -1
if [ "${STAGE}" = train ]; then
  python -u scripts/03_train.py --model "${MODEL}" --config-name "${CFG}" \
      --suffix "_${SUF}_roll${YEAR}_s${SEED}${TAG}" --test-year "${YEAR}" --seed "${SEED}"
  test -s "results/checkpoints/${CFG}_roll${YEAR}_s${SEED}${TAG}.pt"
else
  python -u scripts/47_finetune_per_crop.py --test-year "${YEAR}" --seed "${SEED}" --base "${CFG}" --out "${OUT}"
fi
echo "=== $(date '+%F %T') DONE ==="

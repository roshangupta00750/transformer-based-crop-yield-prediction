#!/usr/bin/env bash
#SBATCH --job-name=cyp-v11
#SBATCH --partition=standard
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=05:00:00
#SBATCH --array=1-72%12
#SBATCH --output=slurm_logs/v11-%x-%A_%a.out
#SBATCH --error=slurm_logs/v11-%x-%A_%a.err
# Pre-registration v11 leave-one-out. STAGE=train|finetune. Tasks: 4 arms (W,T,M,Y) x 2010-2015 x 3 seeds. L40.
set -euo pipefail
source "${VENV:-.venv}/bin/activate"   # project environment (requirements.txt)
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CROP_YIELD_DEVICE=cuda
cd "${SLURM_SUBMIT_DIR:-.}"   # submit from the project root
i=$((SLURM_ARRAY_TASK_ID - 1)); ARMS=(W T M Y); ARM=${ARMS[$((i / 18))]}
YEAR=$((2010 + (i % 18) / 3)); SEED=$((42 + i % 3))
CFG=itransformer_v11${ARM}; OUT=itransformer_v11${ARM}F
echo "=== $(date '+%F %T') ${STAGE} ${CFG} ${YEAR} s${SEED} ==="; hostname
if [ "${STAGE}" = train ]; then
  python -u scripts/03_train.py --model itransformer --config-name "${CFG}" \
      --suffix "_v11${ARM}_roll${YEAR}_s${SEED}" --test-year "${YEAR}" --seed "${SEED}"
  test -s "results/checkpoints/${CFG}_roll${YEAR}_s${SEED}.pt"
else
  python -u scripts/47_finetune_per_crop.py --test-year "${YEAR}" --seed "${SEED}" --base "${CFG}" --out "${OUT}"
fi
echo "=== $(date '+%F %T') DONE ==="

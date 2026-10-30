#!/usr/bin/env bash
#SBATCH --job-name=cyp-cmp-v17
#SBATCH --partition=standard,gpu_computervision
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --array=1-15
#SBATCH --output=slurm_logs/compare-v17-%A_%a.out
#SBATCH --error=slurm_logs/compare-v17-%A_%a.err
# Pre-registration v18 (paper protocols): TFT (ARM=T) / PatchTST (ARM=P) on the other papers' splits,
# exactly as scripts/sbatch_compare_v12.sh does for the iTransformer. QoS: account default (computervision).
# Tasks 1-3 AgroForecast-DE, 4-6 CropFusionNet, 7-15 Srivastava 2022 (2017/18/19 x 3 seeds).
set -euo pipefail
source "${VENV:-.venv}/bin/activate"   # project environment (requirements.txt)
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CROP_YIELD_DEVICE=cuda
cd "${SLURM_SUBMIT_DIR:-.}"   # submit from the project root
case "${ARM}" in
  T) MODEL=tftnet;     CFG=tftnet_v17T;     SUF=v17T; OUT=tftnet_v17TF ;;
  P) MODEL=patchtst17; CFG=patchtst17_v17P; SUF=v17P; OUT=patchtst17_v17PF ;;
esac
i=$((SLURM_ARRAY_TASK_ID - 1)); SEED=$((42 + i % 3))
if   [ $i -lt 3 ]; then TAG=AFDE; END=2019; VAL=2020,2021,2022; SPLIT="--train-end 2019 --val-years 2020,2021,2022 --test-years 2023,2024"
elif [ $i -lt 6 ]; then TAG=CFN;  END=2018; VAL=2019,2020,2021; SPLIT="--train-end 2018 --val-years 2019,2020,2021 --test-years 2022,2023,2024"
else Y=$((2017 + (i - 6) / 3)); TAG=SRV$Y; END=$((Y - 1)); VAL=""
     SPLIT="--train-end $((Y - 1)) --test-years $Y --val-random-frac 0.15"; fi
NAME="${CFG}_cmp${TAG}_s${SEED}"
echo "=== $(date '+%F %T') START ${NAME} (${SPLIT}) ==="; hostname; nvidia-smi -L | head -1
python -u scripts/03_train.py --model "${MODEL}" --config-name "${CFG}" \
    --suffix "_${SUF}_cmp${TAG}_s${SEED}" --seed "${SEED}" ${SPLIT}
test -s "results/checkpoints/${NAME}.pt"
echo "=== $(date '+%F %T') FINE-TUNE ${TAG} s${SEED} ==="
python -u scripts/47_finetune_per_crop.py --tag "cmp${TAG}" --train-end "${END}" \
    ${VAL:+--val-years ${VAL}} --seed "${SEED}" --base "${CFG}" --out "${OUT}"
echo "=== $(date '+%F %T') DONE ==="

#!/usr/bin/env bash
# Retrain all 6 model families on the new 15-feature-augmented dataset.parquet.
# Julia2 L40 for the 5 deep models in parallel (H100 nodes are ecc-drained,
# L40 is ~= H100 throughput for our data-bound TFT workload).
# XGBoost stays on Mac CPU.

set -euo pipefail

REPO=/Users/roshangupta/Documents/Claude/Projects/Agriculture/crop_yield_benchmark
CW=/home/s517868/crop_yield_benchmark
PY=/Users/roshangupta/miniconda3/envs/agri/bin/python

cd "$REPO"

banner() {
    printf "\n═══════════════════════════════════════════════════════════════════\n"
    printf " %s  [%s]\n" "$1" "$(date '+%H:%M:%S')"
    printf "═══════════════════════════════════════════════════════════════════\n"
}

# ───────────────────────────────────────────────────────────────────────
# 0/N  Sync latest code + dataset to Julia2
# ───────────────────────────────────────────────────────────────────────
banner "0/N SYNC to julia2"
rsync -avh --exclude=__pycache__ src/ julia2:$CW/src/ 2>&1 | tail -2
rsync -avh --exclude=__pycache__ configs/ julia2:$CW/configs/ 2>&1 | tail -2
rsync -avh --exclude=__pycache__ scripts/03_train.py scripts/sbatch_*.sh julia2:$CW/scripts/ 2>&1 | tail -2
# The critical bit — the newly merged dataset with 15 land-features
rsync -avh data/processed/dataset.parquet julia2:$CW/data/processed/dataset.parquet 2>&1 | tail -2

# ───────────────────────────────────────────────────────────────────────
# Helper — generate + submit an sbatch for one model
# ───────────────────────────────────────────────────────────────────────
submit_model() {
    local MODEL=$1
    local TIME=$2       # e.g. "08:30:00" (SLURM wall — > Lightning cap + buffer)
    local MAX_MIN=$3    # Lightning max_time in minutes

    banner "SUBMIT $MODEL (--time=$TIME  max_min=$MAX_MIN)"
    ssh julia2 "cat > /tmp/sbatch_${MODEL}.sh" <<SBATCH
#!/usr/bin/env bash
#SBATCH --job-name=retrain-${MODEL}
#SBATCH --partition=standard
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=${TIME}
#SBATCH --output=/home/s517868/slurm_logs/retrain-${MODEL}-%j.out
#SBATCH --error=/home/s517868/slurm_logs/retrain-${MODEL}-%j.err

set -euo pipefail
mkdir -p /home/s517868/slurm_logs
source ~/agri-venv/bin/activate
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd ~/crop_yield_benchmark

echo "=== ${MODEL} on \$(hostname) ==="
nvidia-smi -L

# Bracket-sed the cluster training regime on experiment.yaml — restore on EXIT
cp configs/experiment.yaml /tmp/experiment_${MODEL}_orig.yaml
trap 'cp /tmp/experiment_${MODEL}_orig.yaml configs/experiment.yaml' EXIT
sed -i 's/^device:.*/device: "cuda"/' configs/experiment.yaml
sed -i "s/^  max_minutes:.*/  max_minutes: ${MAX_MIN}/" configs/experiment.yaml
grep -E "^device|^  max_epochs|^  early_stopping|^  max_minutes" configs/experiment.yaml | sed 's/^/  /'

echo "=== training start: \$(date '+%H:%M:%S') ==="
rm -f results/checkpoints/${MODEL}.pt results/checkpoints/${MODEL}.pipeline.json
python -u scripts/03_train.py --model ${MODEL}

echo "=== checkpoint output ==="
ls -lh results/checkpoints/${MODEL}.pt results/checkpoints/${MODEL}.pipeline.json 2>&1
echo "=== ${MODEL} done: \$(date '+%H:%M:%S') ==="
SBATCH

    ssh julia2 "sbatch /tmp/sbatch_${MODEL}.sh" 2>&1
}

# ───────────────────────────────────────────────────────────────────────
# 1/N  XGBoost on Mac (parallel with the samal / julia2 jobs)
# ───────────────────────────────────────────────────────────────────────
banner "XGBoost (Mac, ~5 min) — background"
(
    rm -f results/checkpoints/xgboost.ubj results/checkpoints/xgboost.pipeline.json
    env OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE PYTHONUNBUFFERED=1 \
        "$PY" -u scripts/03_train.py --model xgboost 2>&1
    ls -lh results/checkpoints/xgboost.ubj
    echo "  XGBoost done: $(date '+%H:%M:%S')"
) > /tmp/xgboost_retrain.log 2>&1 &
XGB_PID=$!

# ───────────────────────────────────────────────────────────────────────
# 2/N  Submit 5 deep-model sbatches to julia2 L40 in parallel
# ───────────────────────────────────────────────────────────────────────
#
# Times: SLURM --time > Lightning max_minutes + 30 min for safe checkpoint save
#
submit_model dlinear      "00:30:00"  20
submit_model itransformer "00:45:00"  30
submit_model lstm         "01:00:00"  45
submit_model patchtst     "01:30:00"  90
submit_model tft          "08:30:00"  480

banner "ALL 5 SBATCH JOBS SUBMITTED — polling julia2 queue"
ssh julia2 'squeue -u $USER -o "%.12i %.15j %.12P %.2t %.10M %.10L %.20R"'

# ───────────────────────────────────────────────────────────────────────
# 3/N  Wait for all 5 cluster jobs + XGBoost
# ───────────────────────────────────────────────────────────────────────
banner "WAITING for 5 julia2 jobs to complete"
while true; do
    N=$(ssh -o ConnectTimeout=8 julia2 "squeue -u \$USER -h -n retrain-dlinear,retrain-lstm,retrain-patchtst,retrain-itransformer,retrain-tft -o '%A' 2>/dev/null | wc -l" 2>&1 | xargs)
    echo "  [$(date '+%H:%M:%S')] $N of 5 retrain jobs still in queue"
    [ "$N" = "0" ] && break
    sleep 60
done

banner "XGBoost result"
wait $XGB_PID
tail -15 /tmp/xgboost_retrain.log

# ───────────────────────────────────────────────────────────────────────
# 4/N  Pull all 5 checkpoints back to Mac
# ───────────────────────────────────────────────────────────────────────
banner "PULL checkpoints back to Mac"
for M in dlinear lstm patchtst itransformer tft; do
    scp julia2:$CW/results/checkpoints/${M}.pt results/checkpoints/${M}.pt 2>&1 | tail -1
    scp julia2:$CW/results/checkpoints/${M}.pipeline.json results/checkpoints/${M}.pipeline.json 2>&1 | tail -1 || true
done
ls -lh results/checkpoints/*.pt results/checkpoints/*.ubj 2>&1

# ───────────────────────────────────────────────────────────────────────
# 5/N  IF TFT ckpt fails to load on Mac (CPU-only torch + pf CUDA quirk),
#      grab the auto-saved Lightning ckpt as a fallback (stripped)
# ───────────────────────────────────────────────────────────────────────
banner "TFT: verify Mac load works, else fall back to Lightning auto-ckpt on julia2"
"$PY" - <<'PY' 2>&1 || true
import warnings; warnings.filterwarnings("ignore")
import torch
from pytorch_forecasting import TemporalFusionTransformer
try:
    m = TemporalFusionTransformer.load_from_checkpoint(
        "results/checkpoints/tft.pt", map_location="cpu")
    print(f"  TFT load OK  params={sum(p.numel() for p in m.parameters()):,}")
except Exception as e:
    print(f"  TFT Mac load FAILED: {repr(e)[:150]}")
    print("  Fall back manually via strip_ckpt if needed.")
PY

banner "RETRAIN ORCHESTRATOR COMPLETE"
echo "  Fresh checkpoints for all 6 models on Mac. Next: step 11 evaluate."

#!/usr/bin/env bash
#SBATCH --job-name=cyp-cropaware-smoke
#SBATCH --partition=h100
#SBATCH --qos=computervision
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=slurm_logs/cropaware-smoke-%j.out
#SBATCH --error=slurm_logs/cropaware-smoke-%j.err
# Gate for the crop-aware experiment: one real epoch of the crop-aware arm
# through train -> checkpoint -> reload -> scoring. The two full arms only
# start if this exits 0 (afterok), so a broken build never burns GPU hours.
set -euo pipefail
source "${VENV:-.venv}/bin/activate"   # project environment (requirements.txt)
export PYTHONUNBUFFERED=1 CROP_YIELD_DEVICE=cuda CROP_YIELD_SMOKE=1
cd "${SLURM_SUBMIT_DIR:-.}"   # submit from the project root
python -u scripts/03_train.py --model itransformer \
    --config-name itransformer_cropaware --suffix _smoke
test -s results/checkpoints/itransformer_smoke.pt
python - <<'PY'
from src.models.itransformer import ITransformerModel
m = ITransformerModel({}).load("results/checkpoints/itransformer_smoke.pt")
assert len(m.net.static_token_idx) == 3, m.net.static_token_idx
print("SMOKE reload OK, static token groups:", [len(i) for i in m.net.static_token_idx])
PY
rm -f results/checkpoints/itransformer_smoke.* results/metrics/itransformer_smoke_meta.json
echo "SMOKE PASSED"

#!/usr/bin/env bash
#SBATCH --job-name=hw-parity
#SBATCH --qos=computervision
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/home/s517868/slurm_logs/hwpar-%A_%a.out
#SBATCH --error=/home/s517868/slurm_logs/hwpar-%A_%a.err

# Does the GPU affect the metrics? Same commit, same seed, same config —
# only the hardware differs. Submit twice, once per partition:
#   sbatch --partition=h100     --gres=gpu:H100:1 --export=ALL,TAG=h100 ...
#   sbatch --partition=standard --gres=gpu:L40:1  --export=ALL,TAG=l40  ...
#
# Motivation: metrics fell across a run that ALSO moved L40 -> H100, so
# hardware and code changed together. iTransformer moved only 0.470 -> 0.468
# across that boundary, which argues against a hardware effect, but
# torch.set_float32_matmul_precision("high") enables TF32 matmuls whose
# mantissa width differs by device. That is a real mechanism and it has not
# been tested. If the two arms disagree, every cross-run comparison in this
# project is unsafe.
set -euo pipefail
mkdir -p /home/s517868/slurm_logs
source ~/agri-venv/bin/activate
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export CROP_YIELD_DEVICE=cuda
cd ~/crop_yield_benchmark

echo "=== $(date '+%F %H:%M:%S') hardware-parity arm: ${TAG} ==="
hostname; nvidia-smi -L | head -1
python -c "import torch; print('  torch', torch.__version__, '| tf32 matmul:', torch.backends.cuda.matmul.allow_tf32)"

python -u scripts/03_train.py --model itransformer --suffix "_hw_${TAG}"

echo "=== $(date '+%F %H:%M:%S') done ${TAG} ==="
ls -lh "results/checkpoints/itransformer_hw_${TAG}.pt"

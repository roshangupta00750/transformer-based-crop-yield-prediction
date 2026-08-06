#!/usr/bin/env bash
#SBATCH --job-name=cyp-tax-fusion
#SBATCH --partition=h100
#SBATCH --qos=computervision
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=/home/s517868/slurm_logs/tax-fusion-%j.out
#SBATCH --error=/home/s517868/slurm_logs/tax-fusion-%j.err

# Regenerates the two artefacts still dated Jul 21, i.e. still derived from
# leakage-contaminated checkpoints:
#
#   results/metrics/taxonomy_validation.csv        <- the paper's headline claim
#   results/metrics/taxonomy_rank_correlation.csv
#   results/metrics/fusion_summary_6models.csv
#   results/metrics/fusion_coefficients_6models.csv
#
# Must run AFTER the eval, because 14_taxonomy_validate.py reads
# multicrop_summary.csv and 07_fusion.py re-predicts from the checkpoints.
# Submit with --dependency=afterany:<eval job>.

set -euo pipefail
mkdir -p /home/s517868/slurm_logs
source ~/agri-venv/bin/activate
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export CROP_YIELD_DEVICE=cuda
cd ~/crop_yield_benchmark

echo "=== $(date '+%F %H:%M:%S') inputs ==="
ls -l results/metrics/multicrop_summary.csv
echo "pairs: $(($(wc -l < results/metrics/multicrop_summary.csv) - 1))/60"

echo
echo "═══ taxonomy validation ═══"
python -u scripts/14_taxonomy_validate.py

echo
echo "═══ fusion (6 models) ═══"
python -u scripts/07_fusion.py --suffix _6models

echo
echo "=== $(date '+%F %H:%M:%S') outputs ==="
ls -lh results/metrics/taxonomy_validation.csv \
       results/metrics/taxonomy_rank_correlation.csv \
       results/metrics/fusion_summary_6models.csv 2>&1

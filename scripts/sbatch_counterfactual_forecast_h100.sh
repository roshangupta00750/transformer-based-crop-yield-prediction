#!/usr/bin/env bash
#SBATCH --job-name=cf-fc-24
#SBATCH --partition=h100
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=slurm_logs/cf-fc-24-%j.out
#SBATCH --error=slurm_logs/cf-fc-24-%j.err

# H100 rerun of counterfactual (13_counterfactual_daily.py) and forecast_2024
# (20_forecast_2024.py). Both consume the shipped 6 checkpoints and produce
# metric CSVs that the Streamlit Climate + Home pages read.
set -euo pipefail
mkdir -p slurm_logs
source "${VENV:-.venv}/bin/activate"   # project environment (requirements.txt)
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "${SLURM_SUBMIT_DIR:-.}"   # submit from the project root

echo "=== node + GPU ==="
hostname
nvidia-smi -L
free -h | head -2

# Force cuda for TFT inference during counterfactual (has 6-model loop)
cp configs/experiment.yaml /tmp/experiment_cf_fc.yaml
trap 'cp /tmp/experiment_cf_fc.yaml configs/experiment.yaml' EXIT
sed -i 's/^device:.*/device: "cuda"/' configs/experiment.yaml

echo
echo "═══ counterfactual (~30-45 min) ═══"
rm -f results/metrics/counterfactual_daily_summary.csv results/metrics/counterfactual_daily_district.csv
python -u scripts/13_counterfactual_daily.py

echo
echo "═══ forecast 2024 (~60-90 min) ═══"
rm -f results/metrics/forecast_2024.csv results/metrics/forecast_2024_metrics.csv results/metrics/forecast_2024_leaderboard.csv
python -u scripts/20_forecast_2024.py

echo
echo "═══ output summary ═══"
ls -lh results/metrics/counterfactual_daily_*.csv results/metrics/forecast_2024*.csv
echo "=== cf-fc-24 done: $(date '+%H:%M:%S') ==="

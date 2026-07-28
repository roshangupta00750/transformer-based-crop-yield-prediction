#!/usr/bin/env bash
#SBATCH --job-name=cf-fc-v3
#SBATCH --partition=standard
#SBATCH --qos=normal
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=/home/s517868/slurm_logs/cf-fc-v3-%j.out
#SBATCH --error=/home/s517868/slurm_logs/cf-fc-v3-%j.err

# Regenerates the two downstream metric sets that still reflect the old
# shared multi-crop TFT checkpoint:
#   results/metrics/counterfactual_daily_{summary,district}.csv
#   results/metrics/forecast_2024{,_metrics,_leaderboard}.csv
#
# Both scripts call dashboard/_model_template.py::_predict_test_set, which now
# prefers the per-crop tft_<crop>.pt checkpoints and rebuilds pf's reference
# dataset with each checkpoint's recorded encoder window, so no change is
# needed inside either script.
#
# CAVEAT worth carrying into the paper: 13_counterfactual_daily.py perturbs
# tasmax, tasmin, pr and rsds, but feature policy v3 excludes radiation_global
# (90.5% missing in the source weather table). The rsds channel is therefore
# inert — these counterfactuals measure the yield response to warming and
# precipitation change only. Satellite indices are likewise held at their
# observed values, so vegetation feedback is not simulated. Both are inherent
# limits of the delta method, not regressions.
#
# standard/L40 under the normal QoS: the h100 partition's computervision QoS
# group cap is exhausted. Wall-time is generous (8h vs the 3h the H100 run
# used) because the L40 is slower and this sweep loads TFT many times.

set -euo pipefail
mkdir -p /home/s517868/slurm_logs
source ~/agri-venv/bin/activate

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
# Preferred over rewriting configs/experiment.yaml in place — concurrent jobs
# used to race on that file.
export CROP_YIELD_DEVICE=cuda

cd ~/crop_yield_benchmark

echo "=== node ==="
hostname
nvidia-smi -L | head -1
free -h | head -2

echo
echo "=== per-crop TFT checkpoints visible to the sweep ==="
ls -1 results/checkpoints/tft_*.pt 2>/dev/null | grep -v abs | wc -l

echo
echo "═══ $(date '+%F %H:%M:%S') counterfactual ═══"
rm -f results/metrics/counterfactual_daily_summary.csv \
      results/metrics/counterfactual_daily_district.csv
python -u scripts/13_counterfactual_daily.py

echo
echo "═══ $(date '+%F %H:%M:%S') forecast 2024 ═══"
rm -f results/metrics/forecast_2024.csv \
      results/metrics/forecast_2024_metrics.csv \
      results/metrics/forecast_2024_leaderboard.csv
python -u scripts/20_forecast_2024.py

echo
echo "═══ $(date '+%F %H:%M:%S') done ═══"
ls -lh results/metrics/counterfactual_daily_*.csv results/metrics/forecast_2024*.csv

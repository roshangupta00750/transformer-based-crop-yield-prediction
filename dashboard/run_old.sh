#!/usr/bin/env bash
# Backup launcher: the earlier multi-page dashboard (Report / analytics / Background pages), port 8503.
#
# Pins OpenMP/MKL to a single thread and tells Intel MKL it's okay if
# multiple libomp copies end up loaded (which happens when torch + xgboost
# + pytorch-forecasting are imported into the same Python process on
# macOS Intel). Without these the dashboard segfaults the moment any
# model page touches a checkpoint.
set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export KMP_DUPLICATE_LIB_OK=TRUE

cd "$(dirname "$0")/.."   # cd into crop_yield_benchmark project root
# use the project environment (requirements.txt) if it exists; set VENV to use another one
if [ -f "${VENV:-.venv}/bin/activate" ]; then source "${VENV:-.venv}/bin/activate"; fi
python scripts/63_dashboard_data.py >/dev/null   # refresh results/dashboard/ from the latest evaluations
# localhost only: without --server.address, Streamlit listens on every network interface,
# so anyone on the same Wi-Fi/VPN could open the dashboard. Pass --server.address 0.0.0.0 to share.
exec streamlit run dashboard/nav.py --server.port 8503 --server.address 127.0.0.1 "$@"

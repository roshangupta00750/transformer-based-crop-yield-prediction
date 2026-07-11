#!/usr/bin/env bash
# Launcher for the crop-yield dashboard.
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
exec streamlit run dashboard/Home.py "$@"

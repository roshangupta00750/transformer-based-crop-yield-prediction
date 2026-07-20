"""2024 out-of-training-window yield forecast — a real validation of the
shipped 6-model benchmark on a year the models have NEVER seen.

Pipeline
--------
For each (model, crop):
  1. Load the shipped pipeline.json + checkpoint (Jul 19 21-static schema).
  2. Filter dataset.parquet to year=2024 (freshly rebuilt with 2024 DWD +
     GEE + Destatis-if-available).
  3. Apply the saved feature pipeline (no re-fit — pure inference).
  4. Predict per (landkreis, year=2024) → one number per district-crop.
  5. Aggregate to a per-crop, per-model forecast_2024.csv.

If Destatis 2024 ground truth is present in the dataset (yield_dt_per_ha
non-null on 2024 rows), also emit forecast_2024_metrics.csv with per-
(model, crop) R² / RMSE / MAE and a leaderboard.

Silent-bug prevention
---------------------
  * Uses the existing dashboard._model_template._predict_test_set helper —
    the same code path that produced the 2022-23 test-set numbers, so any
    2024 collapse points to data-drift, not evaluation-code drift.
  * Passes years=[2024] to force the year override (defaults to
    configs/data.yaml splits.temporal.test_years otherwise).
  * TFT's _predict_test_set_tft previously dropped rows where y_true is
    NaN — patched to drop only on y_pred (see dashboard/_model_template.py).
  * If dataset.parquet has NO 2024 rows we exit with a clear error before
    firing any per-model inference (waste-prevention).
"""
from __future__ import annotations

import gc
import logging
import os
import sys
import types
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "dashboard"))

# Stub streamlit so the dashboard helper imports headlessly.
st = types.ModuleType("streamlit")
st.cache_data = lambda **kw: (lambda f: f)
st.cache_resource = lambda **kw: (lambda f: f)
st.set_page_config = lambda **kw: None
sys.modules["streamlit"] = st

import importlib.util
spec = importlib.util.spec_from_file_location(
    "mt", str(REPO / "dashboard" / "_model_template.py"))
mt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mt)

from src.utils import load_data_config

FORECAST_YEAR = 2024
MODELS = ["xgboost", "dlinear", "itransformer", "lstm", "patchtst", "tft"]
OUT_DIR = REPO / "results" / "metrics"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FORECAST_CSV = OUT_DIR / f"forecast_{FORECAST_YEAR}.csv"
METRICS_CSV = OUT_DIR / f"forecast_{FORECAST_YEAR}_metrics.csv"
LEADERBOARD_CSV = OUT_DIR / f"forecast_{FORECAST_YEAR}_leaderboard.csv"


def main() -> None:
    cfg = load_data_config()
    crops = cfg["crops"]

    # Sanity pre-check: does dataset.parquet actually have 2024 rows?
    ds_path = REPO / "data" / "processed" / "dataset.parquet"
    ds = pd.read_parquet(ds_path, columns=["year", "yield_dt_per_ha"])
    n_2024 = (ds["year"] == FORECAST_YEAR).sum()
    if n_2024 == 0:
        raise SystemExit(
            f"[abort] dataset.parquet has 0 rows for year={FORECAST_YEAR}. "
            "Run the S1-S9 rebuild first (dataset must include the target year)."
        )
    truth_2024 = ds[ds["year"] == FORECAST_YEAR]["yield_dt_per_ha"]
    has_truth = truth_2024.notna().sum()
    print(f"  dataset.parquet: {n_2024:,} rows for {FORECAST_YEAR}; "
          f"{has_truth:,} have non-null yield_dt_per_ha (Destatis truth)")
    del ds

    # Incremental write: append after each (model, crop) so a mid-run OOM
    # (previous version lost all progress) still leaves a partial CSV.
    header_written = False
    if FORECAST_CSV.exists():
        FORECAST_CSV.unlink()
    n_written = 0

    metric_rows: list[dict] = []
    for model in MODELS:
        for crop in crops:
            try:
                df, err = mt._predict_test_set(model, crop, years=[FORECAST_YEAR])
            except Exception as e:  # noqa: BLE001
                print(f"  {model:13s} {crop:16s} → EXCEPTION: "
                      f"{type(e).__name__}: {str(e)[:80]}")
                continue
            if err or df.empty:
                print(f"  {model:13s} {crop:16s} → SKIP "
                      f"({(err or {}).get('error', 'empty')[:60]})")
                continue

            df["model"] = model
            df["crop"] = crop
            out_df = df[["model", "crop", "landkreis", "year", "y_pred", "y_true"]]
            # Append to CSV
            out_df.to_csv(FORECAST_CSV, mode="a", index=False,
                          header=not header_written)
            header_written = True
            n_written += len(out_df)

            # Metrics if truth exists
            valid = df.dropna(subset=["y_true"])
            if len(valid) >= 3:
                yt = valid["y_true"].to_numpy()
                yp = valid["y_pred"].to_numpy()
                metric_rows.append({
                    "model": model, "crop": crop,
                    "r2": float(r2_score(yt, yp)),
                    "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
                    "mae": float(mean_absolute_error(yt, yp)),
                    "n": int(len(valid)),
                })
                print(f"  {model:13s} {crop:16s} → "
                      f"R²={metric_rows[-1]['r2']:6.3f}  "
                      f"RMSE={metric_rows[-1]['rmse']:7.2f}  n={len(valid)}")
            else:
                print(f"  {model:13s} {crop:16s} → forecast-only "
                      f"(no truth for R²), n_pred={len(df)}")

            # Free memory before the next pair — OOM protection
            del df, out_df, valid
            gc.collect()

    if n_written == 0:
        raise SystemExit("[abort] No predictions produced for any (model, crop).")
    print(f"\n  wrote {FORECAST_CSV.relative_to(REPO)}  ({n_written:,} rows total)")

    # ---- If truth exists, write per-(model, crop) metrics + leaderboard ----
    if metric_rows:
        metrics = pd.DataFrame(metric_rows)
        metrics.to_csv(METRICS_CSV, index=False)
        print(f"  wrote {METRICS_CSV.relative_to(REPO)}  ({len(metrics)} rows)")

        lb = (metrics.groupby("model")
              .agg(mean_r2=("r2", "mean"),
                   median_r2=("r2", "median"),
                   min_r2=("r2", "min"),
                   max_r2=("r2", "max"),
                   n_crops=("crop", "nunique"))
              .sort_values("mean_r2", ascending=False)
              .round(3))
        lb.to_csv(LEADERBOARD_CSV)
        print(f"  wrote {LEADERBOARD_CSV.relative_to(REPO)}")
        print()
        print(f"=== {FORECAST_YEAR} FORECAST LEADERBOARD ===")
        print(lb.to_string())
    else:
        print()
        print(f"  no Destatis truth available for {FORECAST_YEAR} — "
              "forecast-only mode. Run pull + re-run for validation.")


if __name__ == "__main__":
    main()

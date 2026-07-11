"""Proper leave-one-Landkreis-out (LOO) sweep for XGBoost.

XGBoost is fast enough (~30-60 s per fold) to run a real LOO methodology:
for each of N sampled districts, train on all-others-only and evaluate on
the held-out district's 2022-2023 test set.

This is the methodology Slide 8 of the PPT explicitly promises ("Spatial
holdout: leave-one-Landkreis-out"). The deep models would each cost
~50-150 hours and are out of scope for a CPU-only Praktikum compute
budget; see :mod:`scripts.06_spatial_eval` for the per-district view
without retraining.

Output: ``results/metrics/loo_xgboost.csv`` with columns
``crop, landkreis, r2, rmse, mae, n``.

Usage:
    python scripts/06b_loo_xgboost.py [--sample-size 50] [--crop winter_wheat]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.dataset import (CropYieldDataset, FeaturePipeline, FeatureSpec,
                              apply_phenology_window, collate_pad)
from src.models.xgboost_model import _flatten_batch, loader_to_xy
from src.utils import load_data_config, load_model_config


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample-size", type=int, default=50,
                    help="Number of districts to hold out (random sample).")
    ap.add_argument("--crop", default=None,
                    help="Restrict to one crop (default: all configured crops).")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data_cfg = load_data_config()
    model_cfg = load_model_config("xgboost")
    crops = [args.crop] if args.crop else data_cfg["crops"]

    df_full = pd.read_parquet(ROOT / data_cfg["paths"]["unified_dataset"])
    df_full = apply_phenology_window(df_full, data_cfg["phenology"])

    train_end = data_cfg["splits"]["temporal"]["train_end_year"]
    test_years = data_cfg["splits"]["temporal"]["test_years"]

    # Pick the held-out districts up front so the sample is consistent across crops.
    all_districts = sorted(df_full["landkreis"].unique().tolist())
    rng = np.random.default_rng(args.seed)
    held_out = rng.choice(all_districts,
                          size=min(args.sample_size, len(all_districts)),
                          replace=False).tolist()

    rows: list[dict] = []
    for crop in crops:
        df_c = df_full[df_full["crop"] == crop].copy()
        if df_c.empty:
            print(f"skip {crop}: no data")
            continue

        for i, lk in enumerate(held_out, 1):
            df_train = df_c[(df_c["year"] <= train_end)
                            & (df_c["landkreis"] != lk)]
            df_test = df_c[(df_c["year"].isin(test_years))
                           & (df_c["landkreis"] == lk)]
            if df_train.empty or df_test.empty:
                print(f"  {crop:16s} {i:>3}/{args.sample_size}  {lk:6s}"
                      f" -> SKIP (no train or test data)")
                continue

            # Feature spec + pipeline fitted on the leave-one-out training set.
            spec = FeatureSpec(
                past_observed=[c for c in df_train.columns
                               if c.startswith(("temperature_", "precip",
                                                "radiation", "humid",
                                                "ndvi", "evi", "lai", "ndwi"))],
                static=[c for c in df_train.columns
                        if c in ("sand", "clay", "soc", "ph", "bulk_density",
                                 "landkreis_id")],
                known_future=[c for c in df_train.columns
                              if c in ("sin_doy", "cos_doy")],
            )
            pipe = FeaturePipeline(spec).fit(df_train)
            df_train_t = pipe.transform_features(df_train)
            df_test_t = pipe.transform_features(df_test)

            ds_tr = CropYieldDataset(df_train_t, spec, pipe)
            ds_te = CropYieldDataset(df_test_t, spec, pipe)
            if len(ds_tr) == 0 or len(ds_te) == 0:
                print(f"  {crop:16s} {i:>3}/{args.sample_size}  {lk:6s}"
                      f" -> SKIP (empty dataset)")
                continue

            tr_loader = DataLoader(ds_tr, batch_size=128,
                                   collate_fn=collate_pad)
            te_loader = DataLoader(ds_te, batch_size=128,
                                   collate_fn=collate_pad)
            Xtr, ytr = loader_to_xy(tr_loader)
            Xte, _ = loader_to_xy(te_loader)

            booster = xgb.XGBRegressor(
                n_estimators=model_cfg["n_estimators"],
                max_depth=model_cfg["max_depth"],
                learning_rate=model_cfg["learning_rate"],
                subsample=model_cfg["subsample"],
                colsample_bytree=model_cfg["colsample_bytree"],
                objective=model_cfg["objective"],
                tree_method="hist", n_jobs=model_cfg.get("n_jobs", 4),
            )
            booster.fit(Xtr, ytr, verbose=False)
            z_pred = booster.predict(Xte)

            lk_arr = np.array([k[0] for k in ds_te.keys])
            cr_arr = np.array([k[1] for k in ds_te.keys])
            y_pred = pipe.inverse_transform_array(lk_arr, cr_arr, z_pred)
            y_true = np.array([pipe.inverse_transform_target(*k[:2], z)
                               for k, z in zip(ds_te.keys, ds_te._targets_z)])

            if np.var(y_true) < 1e-9 or len(y_true) < 2:
                # Single test point or zero variance -> R² undefined; record
                # the absolute error instead.
                rows.append({"crop": crop, "landkreis": lk,
                             "r2": None,
                             "rmse": float(np.sqrt(((y_true - y_pred) ** 2).mean())),
                             "mae": float(np.abs(y_true - y_pred).mean()),
                             "n": int(len(y_true))})
            else:
                rows.append({"crop": crop, "landkreis": lk,
                             "r2": float(r2_score(y_true, y_pred)),
                             "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
                             "mae": float(mean_absolute_error(y_true, y_pred)),
                             "n": int(len(y_true))})
            print(f"  {crop:16s} {i:>3}/{args.sample_size}  {lk:6s}"
                  f" -> R²={rows[-1]['r2']!s:>7.7s}  RMSE={rows[-1]['rmse']:6.2f}  n={rows[-1]['n']}")

    out = pd.DataFrame(rows)
    out_path = ROOT / "results" / "metrics" / "loo_xgboost.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"\nWrote {len(out)} rows to {out_path}")


if __name__ == "__main__":
    main()

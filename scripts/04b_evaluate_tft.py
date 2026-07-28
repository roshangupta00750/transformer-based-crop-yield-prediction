"""TFT-specific evaluator (refactored architecture).

Series_id = ``landkreis__crop`` spans all years. For each in-season horizon
``h`` and each test year, slice the (district, crop) series so it ends at
``harvest_doy − h`` of that year, then ask pf to predict the last day.
This gives one prediction per (landkreis, year) per horizon.

Output goes to ``results/metrics/tft_<crop>_horizons.json`` so the
dashboard leaderboard picks it up automatically.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import (
    FeaturePipeline, prepare_modeling_frame, require_current_dataset_policy,
)
from src.evaluation.metrics import all_metrics
from src.training.tft_adapter import (DEFAULT_MAX_ENCODER_LENGTH,
                                      build_timeseries_dataset,
                                      prepare_tft_dataframe,
                                      transform_tft_features)
from src.utils import (get_logger, load_data_config, load_experiment_config,
                       PROJECT_ROOT)

log = get_logger("evaluate_tft")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--crop", default="winter_wheat")
    args = ap.parse_args()

    from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet

    data_cfg = load_data_config()
    exp_cfg = load_experiment_config()

    unified = PROJECT_ROOT / data_cfg["paths"]["unified_dataset"]
    require_current_dataset_policy(unified)
    df_all = pd.read_parquet(unified)
    df_all = prepare_modeling_frame(df_all, data_cfg)

    train_end = data_cfg["splits"]["temporal"]["train_end_year"]
    test_years = data_cfg["splits"]["temporal"]["test_years"]
    df_train = df_all[df_all["year"] <= train_end]
    if df_train.empty:
        log.error("No training data."); sys.exit(1)

    ckpt = PROJECT_ROOT / exp_cfg["log"]["checkpoints_dir"] / "tft.pt"
    if not ckpt.exists():
        log.error("Missing tft.pt — train first."); sys.exit(1)
    pipeline_path = ckpt.with_name("tft.pipeline.json")
    if not pipeline_path.exists():
        log.error("Missing tft.pipeline.json — retrain under the safe policy.")
        sys.exit(1)
    pipeline = FeaturePipeline.load(pipeline_path)
    pipeline.require_current_feature_policy()
    df_all = transform_tft_features(df_all, pipeline)
    df_train = df_all[df_all["year"] <= train_end]
    df = df_all[df_all["crop"] == args.crop].copy()
    soil_cols = list(pipeline.spec.static)
    weather_sat_cols = list(pipeline.spec.past_observed)

    # Rebuild the reference dataset on the same all-crop population used by
    # training so categorical encoders and target normalizers match.
    df_tr, cols = prepare_tft_dataframe(
        df_train, soil_cols=soil_cols, weather_sat_cols=weather_sat_cols,
    )
    # Match the encoder window the checkpoint was trained with. Per-crop TFTs
    # on short-season crops use a smaller window than the 180 default; feeding
    # the model a longer context than it learned on shifts its predictions.
    # Pre-existing checkpoints have no recorded value and all used 180.
    _enc = getattr(pipeline, "tft_max_encoder_length_", None) or \
        DEFAULT_MAX_ENCODER_LENGTH
    train_ds = build_timeseries_dataset(
        df_tr, cols, max_encoder_length=int(_enc), max_prediction_length=1,
    )
    model = TemporalFusionTransformer.load_from_checkpoint(str(ckpt))
    model.eval()

    harvest_doy = data_cfg["phenology"][args.crop]["harvest_doy"]
    horizons = data_cfg["in_season_horizons_days"]

    results: dict[str, dict] = {}
    for h in horizons:
        cutoff = max(harvest_doy - h, 1)
        rows = []
        for yr in test_years:
            # Include all history up to and including this year, but clip
            # each series at cutoff_doy within the test year.
            df_slice = df[
                (df["year"] < yr) | ((df["year"] == yr) & (df["doy"] <= cutoff))
            ].copy()
            if df_slice.empty:
                continue
            df_slice_p, _ = prepare_tft_dataframe(
                df_slice, soil_cols=soil_cols, weather_sat_cols=weather_sat_cols,
            )
            try:
                test_ds = TimeSeriesDataSet.from_dataset(
                    train_ds, df_slice_p,
                    stop_randomization=True, predict=True,
                )
            except Exception as e:  # noqa: BLE001
                log.warning("Skip h=%d yr=%d: %s", h, yr, e)
                continue

            loader = test_ds.to_dataloader(
                train=False, batch_size=max(len(test_ds), 1), num_workers=0,
            )
            preds = model.predict(loader, mode="prediction",
                                  return_y=True, return_index=True)
            y_pred = preds.output.cpu().numpy().squeeze()
            if y_pred.ndim == 2:
                y_pred = y_pred[:, y_pred.shape[1] // 2]
            y_true = (preds.y[0].cpu().numpy().squeeze()
                      if hasattr(preds, "y") and preds.y is not None else None)
            idx = preds.index if hasattr(preds, "index") else None
            if idx is None or y_true is None:
                continue
            for i, sid in enumerate(idx["series_id"].astype(str).tolist()):
                rows.append({"landkreis": sid.split("__", 1)[0],
                             "year": int(yr),
                             "y_pred": float(y_pred[i]),
                             "y_true": float(y_true[i])})

        if not rows:
            log.warning("No predictions at h-%dd", h)
            continue
        d = pd.DataFrame(rows)
        results[f"h-{h}d"] = all_metrics(d["y_true"].to_numpy(),
                                         d["y_pred"].to_numpy())
        log.info("h-%dd: R²=%.3f  RMSE=%.2f  MAE=%.2f  n=%d",
                 h, results[f"h-{h}d"]["r2"], results[f"h-{h}d"]["rmse"],
                 results[f"h-{h}d"]["mae"], results[f"h-{h}d"]["n"])

    out = (PROJECT_ROOT / exp_cfg["log"]["metrics_dir"]
           / f"tft_{args.crop}_horizons.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    log.info("Saved %s", out)


if __name__ == "__main__":
    main()

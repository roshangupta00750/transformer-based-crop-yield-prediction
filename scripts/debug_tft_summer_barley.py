"""Debug TFT prediction scale on summer_barley — one-shot diagnostic."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import (
    FeaturePipeline, prepare_modeling_frame, require_current_dataset_policy,
)
from src.training.tft_adapter import (
    build_timeseries_dataset, prepare_tft_dataframe, transform_tft_features,
)
from src.utils import load_data_config, load_experiment_config, PROJECT_ROOT

from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet

CROP = "summer_barley"
YR = 2024

data_cfg = load_data_config()
exp_cfg = load_experiment_config()

unified = PROJECT_ROOT / data_cfg["paths"]["unified_dataset"]
require_current_dataset_policy(unified)
df_all = pd.read_parquet(unified)
df_all = prepare_modeling_frame(df_all, data_cfg)

train_end = data_cfg["splits"]["temporal"]["train_end_year"]
ckpt = PROJECT_ROOT / exp_cfg["log"]["checkpoints_dir"] / "tft.pt"
pipeline = FeaturePipeline.load(ckpt.with_name("tft.pipeline.json"))
pipeline.require_current_feature_policy()

df_all = transform_tft_features(df_all, pipeline)
soil = list(pipeline.spec.static); wsat = list(pipeline.spec.past_observed)

# Train dataset for encoder reference
df_train = df_all[df_all["year"] <= train_end]
df_tr, cols = prepare_tft_dataframe(df_train, soil_cols=soil, weather_sat_cols=wsat)
train_ds = build_timeseries_dataset(df_tr, cols, max_encoder_length=180, max_prediction_length=1)

model = TemporalFusionTransformer.load_from_checkpoint(str(ckpt))
model.eval()

# Predict yr, apply the same pre-filter as the fix
df = df_all[df_all["crop"] == CROP].copy()
valid_lk = (df[(df["year"] == YR) & (df["yield_dt_per_ha"].notna())][["landkreis"]]
              .drop_duplicates())
df_slice = df[df["year"] <= YR]
df_slice = df_slice.merge(valid_lk, on="landkreis", how="inner")
df_slice_p, _ = prepare_tft_dataframe(df_slice, soil_cols=soil, weather_sat_cols=wsat)
test_ds = TimeSeriesDataSet.from_dataset(train_ds, df_slice_p,
                                         stop_randomization=True, predict=True)
loader = test_ds.to_dataloader(train=False, batch_size=max(len(test_ds), 1), num_workers=0)

# --- Try BOTH modes and compare ---
for mode in ("prediction", "quantiles", "raw"):
    print(f"\n=== mode='{mode}' ===")
    try:
        preds = model.predict(loader, mode=mode, return_y=True, return_index=True)
    except Exception as e:
        print(f"  failed: {e}")
        continue
    raw = preds.output
    if isinstance(raw, tuple):
        print(f"  output is tuple of {len(raw)}; taking [0]")
        raw = raw[0]
    y_pred = raw.cpu().numpy().squeeze() if hasattr(raw, "cpu") else np.asarray(raw).squeeze()
    print(f"  y_pred shape={y_pred.shape}  range=[{y_pred.min():.3f}, {y_pred.max():.3f}]  mean={y_pred.mean():.3f}")
    if y_pred.ndim == 2:
        y_pred = y_pred[:, y_pred.shape[1] // 2]
        print(f"  after taking median column: range=[{y_pred.min():.3f}, {y_pred.max():.3f}]  mean={y_pred.mean():.3f}")
    y_true = preds.y[0].cpu().numpy().squeeze() if hasattr(preds, "y") else None
    if y_true is not None:
        print(f"  y_true shape={y_true.shape}  range=[{y_true.min():.3f}, {y_true.max():.3f}]  mean={y_true.mean():.3f}")
        # sample 5
        idx = preds.index
        for i in range(min(5, len(y_pred))):
            sid = str(idx["series_id"].iloc[i])
            print(f"    {sid[:30]:30s}  pred={float(y_pred[i]):8.2f}  true={float(y_true[i]):8.2f}  resid={float(y_pred[i]-y_true[i]):+8.2f}")

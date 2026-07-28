"""Interpretability extractors for the dashboard.

One function per model family. Each returns a small pandas object the
``pages/11_Interpret.py`` view can render directly.

We don't try to define a single uniform schema — TFT's per-variable + per-
timestep importance is genuinely different from XGBoost's gain-based score,
and the comparative value comes from showing them side by side.
"""
from __future__ import annotations

import os
import sys
from functools import partial
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from dashboard.utils import CHECKPOINTS_DIR  # noqa: E402


# --------------------------------------------------------------------------- #
# TFT — variable selection + attention via pytorch_forecasting
# --------------------------------------------------------------------------- #
def tft_interpretation(crop: str, max_batches: int = 4) -> dict | None:
    """Return per-variable importance + per-timestep attention for TFT.

    Output dict (all values are pandas):
      - static_importance      : Series, importance per static feature
      - encoder_importance     : Series, mean importance per past feature
      - decoder_importance     : Series, mean importance per known-future feat
      - encoder_attention_mean : Series, mean attention weight per encoder step
      - n_samples              : int, how many batches were averaged

    Returns None if the model / dataset can't be loaded.
    """
    try:
        from pytorch_forecasting import (TemporalFusionTransformer,
                                         TimeSeriesDataSet)
        from src.data.dataset import (
            FeaturePipeline, prepare_modeling_frame,
        )
        from src.training.tft_adapter import (DEFAULT_MAX_ENCODER_LENGTH,
                                              build_timeseries_dataset,
                                              prepare_tft_dataframe,
                                              transform_tft_features)
        from src.utils import load_data_config
    except Exception:  # noqa: BLE001
        return None

    ckpt = None
    for cand in (CHECKPOINTS_DIR / "tft.pt", CHECKPOINTS_DIR / "tft_abs.pt"):
        if cand.exists():
            ckpt = cand
            break
    if ckpt is None:
        return None
    pipeline_path = CHECKPOINTS_DIR / "tft.pipeline.json"
    if not pipeline_path.exists():
        return None
    try:
        safe_pipeline = FeaturePipeline.load(pipeline_path)
        safe_pipeline.require_current_feature_policy()
    except RuntimeError:
        return None

    cfg = load_data_config()
    parquet = ROOT / cfg["paths"]["unified_dataset"]
    if not parquet.exists():
        return None
    df_all = pd.read_parquet(parquet)
    df_all = prepare_modeling_frame(df_all, cfg)
    try:
        df_all = transform_tft_features(df_all, safe_pipeline)
    except (KeyError, ValueError, RuntimeError):
        return None

    train_end = cfg["splits"]["temporal"]["train_end_year"]
    test_years = cfg["splits"]["temporal"]["test_years"]

    soil_cols = list(safe_pipeline.spec.static)
    weather_sat_cols = list(safe_pipeline.spec.past_observed)

    df_train_all = df_all[df_all["year"] <= train_end]
    df_tr, cols = prepare_tft_dataframe(
        df_train_all, soil_cols=soil_cols, weather_sat_cols=weather_sat_cols,
    )
    # Match the encoder window the checkpoint was trained with. Per-crop TFTs
    # on short-season crops use a smaller window than the 180 default; feeding
    # the model a longer context than it learned on shifts its predictions.
    # Pre-existing checkpoints have no recorded value and all used 180.
    _enc = getattr(safe_pipeline, "tft_max_encoder_length_", None) or \
        DEFAULT_MAX_ENCODER_LENGTH
    train_ds = build_timeseries_dataset(
        df_tr, cols, max_encoder_length=int(_enc), max_prediction_length=1,
    )

    model = TemporalFusionTransformer.load_from_checkpoint(str(ckpt))
    model.eval()

    # Build a test slice for the requested crop ending at the latest test year
    # — gives us valid prediction points to interpret.
    df_pred = df_all[(df_all["crop"] == crop)
                     & (df_all["year"] <= max(test_years))].copy()
    df_pred_p, _ = prepare_tft_dataframe(
        df_pred, soil_cols=soil_cols, weather_sat_cols=weather_sat_cols,
    )
    test_ds = TimeSeriesDataSet.from_dataset(
        train_ds, df_pred_p, stop_randomization=True, predict=True,
    )
    if len(test_ds) == 0:
        return None
    loader = test_ds.to_dataloader(
        train=False, batch_size=min(64, len(test_ds)), num_workers=0,
    )

    raw = model.predict(loader, mode="raw", return_x=True, return_y=False)
    interp = model.interpret_output(raw.output, reduction="sum")

    # pf stacks static categoricals + reals in a single importance vector;
    # truncate to whatever length pf produced.
    static = pd.Series(dtype=float)
    static_v = interp.get("static_variables")
    if static_v is not None:
        v = static_v.detach().cpu().float().numpy().squeeze()
        v = np.atleast_1d(v)
        static_names = list(train_ds.static_categoricals) + list(train_ds.static_reals)
        n = min(len(v), len(static_names))
        if n > 0:
            static = pd.Series(v[:n].tolist(),
                               index=static_names[:n]) \
                       .sort_values(ascending=False)

    encoder = pd.Series(dtype=float)
    enc_v = interp.get("encoder_variables")
    if enc_v is not None:
        enc_names = list(train_ds.time_varying_known_reals) + \
                    list(train_ds.time_varying_unknown_reals)
        v = enc_v.detach().cpu().float().numpy().squeeze()
        encoder = pd.Series(v.tolist(), index=enc_names[:v.size]) \
                    .sort_values(ascending=False)

    decoder = pd.Series(dtype=float)
    dec_v = interp.get("decoder_variables")
    if dec_v is not None:
        dec_names = list(train_ds.time_varying_known_reals)
        v = dec_v.detach().cpu().float().numpy().squeeze()
        decoder = pd.Series(v.tolist(), index=dec_names[:v.size]) \
                    .sort_values(ascending=False)

    attention = pd.Series(dtype=float)
    att_v = interp.get("attention")
    if att_v is not None:
        a = att_v.detach().cpu().float().numpy()
        a = a.mean(axis=tuple(range(a.ndim - 1)))
        attention = pd.Series(a.tolist(),
                              index=[f"t-{len(a) - 1 - i}" for i in range(len(a))])

    return {
        "static_importance": static,
        "encoder_importance": encoder,
        "decoder_importance": decoder,
        "encoder_attention_mean": attention,
        "n_samples": int(len(test_ds)),
    }


# --------------------------------------------------------------------------- #
# PatchTST + iTransformer — attention extraction via monkey-patched _sa_block
# --------------------------------------------------------------------------- #
def _capture_attention_run(model_name: str, crop: str,
                           max_batches: int = 2) -> dict | None:
    """Run a small batch through the trained model and capture each transformer
    layer's self-attention weights. The trick: temporarily replace
    ``nn.TransformerEncoderLayer._sa_block`` with a version that asks
    ``need_weights=True`` and stashes the resulting tensor.

    Returns a dict with ``attn_maps`` (list of tensors, one per encoder layer)
    plus the variable / patch labels needed to make a readable bar chart.
    """
    try:
        import torch
        import torch.nn as nn
        from src.data.dataset import (
            CropYieldDataset, FeaturePipeline, collate_pad,
            prepare_modeling_frame,
        )
        from src.models import build_model
        from src.utils import load_data_config, load_model_config
        from torch.utils.data import DataLoader
    except Exception:  # noqa: BLE001
        return None

    ckpt = CHECKPOINTS_DIR / f"{model_name}.pt"
    pipeline_path = CHECKPOINTS_DIR / f"{model_name}.pipeline.json"
    if not ckpt.exists() or not pipeline_path.exists():
        return None
    pipe = FeaturePipeline.load(pipeline_path)
    try:
        pipe.require_current_feature_policy()
    except RuntimeError:
        return None

    data_cfg = load_data_config()
    model_cfg = load_model_config(model_name)
    df = pd.read_parquet(ROOT / data_cfg["paths"]["unified_dataset"])
    df = df[df["crop"] == crop]
    df = prepare_modeling_frame(df, data_cfg)
    df_test = df[df["year"].isin(data_cfg["splits"]["temporal"]["test_years"])]
    if df_test.empty:
        return None
    df_test_t = pipe.transform_features(df_test)
    ds = CropYieldDataset(df_test_t, pipe.spec, pipe)
    if len(ds) == 0:
        return None
    sequence_length = pipe.sequence_length_ or max(
        p["harvest_doy"] if p["sowing_doy"] > p["harvest_doy"]
        else p["harvest_doy"] - p["sowing_doy"] + 1
        for p in data_cfg["phenology"].values()
    )
    loader = DataLoader(
        ds,
        batch_size=min(32, len(ds)),
        collate_fn=partial(collate_pad, pad_to=sequence_length),
    )

    mdl = build_model(model_name, model_cfg)
    mdl.load(ckpt)
    mdl.device = "cpu"
    mdl.net.eval()

    # Torch's TransformerEncoderLayer takes a fused fast-path at eval-time
    # that bypasses ``_sa_block`` and never exposes attention weights. We
    # patch *both* ``_sa_block`` (to grab weights) and ``forward`` (to force
    # the slow path that actually calls ``_sa_block``). Both restored on exit.
    captured: list = []
    orig_sa_block = nn.TransformerEncoderLayer._sa_block
    orig_forward = nn.TransformerEncoderLayer.forward

    def patched_sa(self, x, attn_mask, key_padding_mask, is_causal=False):
        out, weights = self.self_attn(
            x, x, x, attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=True, average_attn_weights=True,
            is_causal=is_causal,
        )
        captured.append(weights.detach().cpu())
        return self.dropout1(out) if hasattr(self, "dropout1") else out

    def slow_forward(self, src, src_mask=None, src_key_padding_mask=None,
                     is_causal=False):
        # Force the documented slow-path so our patched _sa_block runs.
        if self.norm_first:
            x = src + self._sa_block(self.norm1(src), src_mask,
                                     src_key_padding_mask, is_causal)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(src + self._sa_block(src, src_mask,
                                                src_key_padding_mask, is_causal))
            x = self.norm2(x + self._ff_block(x))
        return x

    nn.TransformerEncoderLayer._sa_block = patched_sa
    nn.TransformerEncoderLayer.forward = slow_forward
    try:
        with torch.no_grad():
            for i, batch in enumerate(loader):
                if i >= max_batches:
                    break
                x = torch.cat([batch["x_past"], batch["x_future"]], dim=-1)
                _ = mdl.net(x)
    finally:
        nn.TransformerEncoderLayer._sa_block = orig_sa_block
        nn.TransformerEncoderLayer.forward = orig_forward

    if not captured:
        return None

    # Aggregate: across batches, across heads, across queries → per-key score.
    # captured[i] shape: (B*?, Nq, Nk) for each layer.
    per_layer_scores = []
    for w in captured:
        s = w.mean(dim=0).mean(dim=0)  # avg over (B*C or B) then over queries
        per_layer_scores.append(s.numpy())

    return {
        "per_layer_attention": per_layer_scores,
        "past_features": list(pipe.spec.past_observed),
        "future_features": list(pipe.spec.known_future),
        "n_samples": int(len(ds)),
    }


def patchtst_attention(crop: str) -> dict | None:
    """PatchTST attends over patches (time chunks). Returns per-patch attention
    averaged across heads/layers, plus an interpretable axis label.
    """
    res = _capture_attention_run("patchtst", crop)
    if res is None:
        return None
    # Average across all layers → one curve.
    mat = np.stack(res["per_layer_attention"], axis=0)  # (n_layers, N, ...) but
    # for PatchTST encoder input was (B*C, N, D), so weights are (B*C, N, N).
    # After our mean-over-batch+mean-over-queries, each row is (N,).
    avg = mat.mean(axis=0)
    series = pd.Series(avg, index=[f"patch_{i + 1}" for i in range(len(avg))])
    return {"attention": series, "n_samples": res["n_samples"]}


def itransformer_attention(crop: str) -> dict | None:
    """iTransformer attends over variables (features) — so the attention map
    is one weight per input channel. We label them with the actual feature
    names so the bar chart reads as 'which variables drove the prediction'.
    """
    res = _capture_attention_run("itransformer", crop)
    if res is None:
        return None
    mat = np.stack(res["per_layer_attention"], axis=0)
    avg = mat.mean(axis=0)
    # iTransformer's channels = past + known_future, in concat order.
    labels = list(res["past_features"]) + list(res["future_features"])
    # Truncate to whatever length we actually got.
    n = min(len(avg), len(labels))
    series = pd.Series(avg[:n], index=labels[:n]).sort_values(ascending=False)
    return {"attention": series, "n_samples": res["n_samples"]}


# --------------------------------------------------------------------------- #
# XGBoost — built-in gain-based importance, aggregated back to raw features
# --------------------------------------------------------------------------- #
def xgboost_importance() -> pd.DataFrame | None:
    """Return per-(raw-feature, summary) importance for the multi-crop XGBoost.

    The model trains on a 12-month aggregation: each raw past feature is
    expanded into 12 buckets × 4 stats (mean/std/min/max), then concatenated
    with static features. We map the booster's internal feature names back to
    (raw_feature, month_bucket, stat) so the chart is interpretable.
    """
    try:
        import xgboost as xgb
        import json
        from dashboard.utils import load_pipeline
    except Exception:  # noqa: BLE001
        return None

    pipe = load_pipeline("xgboost")
    if pipe is None:
        return None
    ckpt = CHECKPOINTS_DIR / "xgboost.ubj"
    if not ckpt.exists():
        return None
    booster = xgb.Booster()
    booster.load_model(str(ckpt))
    raw_scores = booster.get_score(importance_type="gain")
    if not raw_scores:
        return None

    # Booster names features f0, f1, …. Reconstruct what each fN is.
    n_past = len(pipe.spec.past_observed)
    n_future = len(pipe.spec.known_future)
    n_static = len(pipe.spec.static)
    # Flatten order from _flatten_batch: for each of 12 chunks, [mean,std,min,max]
    # of all past+future channels stacked, then static features appended.
    past_future = pipe.spec.past_observed + pipe.spec.known_future
    stats = ["mean", "std", "min", "max"]
    feature_index: list[tuple[str, str, str]] = []  # (raw, bucket, stat)
    for chunk in range(12):
        for stat in stats:
            for feat in past_future:
                feature_index.append((feat, f"m{chunk + 1:02d}", stat))
    for s in pipe.spec.static:
        feature_index.append((s, "static", "value"))

    rows = []
    for fname, gain in raw_scores.items():
        idx = int(fname.lstrip("f"))
        if idx >= len(feature_index):
            continue
        raw, bucket, stat = feature_index[idx]
        rows.append({"feature": raw, "bucket": bucket, "stat": stat, "gain": gain})
    return pd.DataFrame(rows)

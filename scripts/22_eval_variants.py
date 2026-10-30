"""Score checkpoint VARIANTS that scripts/05b_eval_multicrop.py will not touch.

05b validates model names against a fixed six-model list, by design — it is
the headline benchmark and should not silently pick up stray checkpoints. But
the attention-axis ablation and the hardware-parity control both produce
checkpoints that reuse an existing builder under a different name:

    itransformer_axis_variate    itransformer, tokens are variates
    itransformer_axis_temporal   itransformer, tokens are timesteps
    itransformer_hw_h100         identical config, trained on an H100
    itransformer_hw_l40          identical config, trained on an L40

This scores them on exactly the population 05b scores, so the numbers are
directly comparable to results/metrics/multicrop_summary.csv, and writes
results/metrics/variant_summary.csv.

Usage:
    python scripts/22_eval_variants.py --variants itransformer_axis_variate,...
                                       --builder itransformer
"""
from __future__ import annotations

import argparse
import sys
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import make_temporal_split
from src.data.dataset import (CropYieldDataset, FeaturePipeline, collate_pad,
                              prepare_modeling_frame,
                              require_current_dataset_policy)
from src.evaluation.metrics import all_metrics
from src.models import build_model
from src.utils import (get_logger, load_data_config, load_experiment_config,
                       load_model_config, PROJECT_ROOT)

log = get_logger("eval_variants")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variants", required=True,
                    help="Comma-separated checkpoint stems under results/checkpoints/")
    ap.add_argument("--builder", default="itransformer",
                    help="Registered model name whose builder these variants use")
    ap.add_argument("--out", default="variant_summary.csv")
    args = ap.parse_args()

    data_cfg = load_data_config()
    exp_cfg = load_experiment_config()
    ckpt_dir = PROJECT_ROOT / exp_cfg["log"]["checkpoints_dir"]

    unified = PROJECT_ROOT / data_cfg["paths"]["unified_dataset"]
    require_current_dataset_policy(unified)
    df_all = prepare_modeling_frame(pd.read_parquet(unified), data_cfg)

    split = make_temporal_split(
        df_all,
        train_end_year=data_cfg["splits"]["temporal"]["train_end_year"],
        val_years=data_cfg["splits"]["temporal"]["val_years"],
        test_years=data_cfg["splits"]["temporal"]["test_years"],
    )
    df_test_all = df_all.loc[split.test]

    rows: list[dict] = []
    for variant in [v.strip() for v in args.variants.split(",") if v.strip()]:
        pipe_path = ckpt_dir / f"{variant}.pipeline.json"
        ck_path = ckpt_dir / f"{variant}.pt"
        if not (pipe_path.exists() and ck_path.exists()):
            log.warning("%s: checkpoint or pipeline missing, skipping", variant)
            continue

        pipeline = FeaturePipeline.load(pipe_path)
        pipeline.require_current_feature_policy()
        spec = pipeline.spec

        # The variant's own config drives the architecture, so an ablation arm
        # rebuilds with the axis it was trained with rather than the default.
        try:
            model_cfg = load_model_config(variant)
        except Exception:
            model_cfg = load_model_config(args.builder)
        model = build_model(args.builder, model_cfg)
        model.load(ck_path)

        seq_len = pipeline.sequence_length_
        for crop in data_cfg["crops"]:
            df_c = df_test_all[df_test_all["crop"] == crop]
            if df_c.empty:
                continue
            df_c_t = pipeline.transform_features(df_c)
            ds = CropYieldDataset(df_c_t, spec, pipeline)
            if len(ds) == 0:
                continue
            loader = DataLoader(ds, batch_size=128,
                                collate_fn=partial(collate_pad, pad_to=seq_len))
            z = np.asarray(model.predict(loader)).ravel()
            lk = np.array([k[0] for k in ds.keys])
            cr = np.array([k[1] for k in ds.keys])
            yr = np.array([k[2] for k in ds.keys])
            y_pred = pipeline.inverse_transform_array(lk, cr, z, yr)
            y_true = pipeline.inverse_transform_observed_array(
                lk, cr, np.asarray(ds._targets_z), yr)
            m = all_metrics(y_true, y_pred)
            rows.append({"variant": variant, "crop": crop, **m})
            log.info("%-30s %-16s R2=%+.3f n=%d", variant, crop, m["r2"], m["n"])

    if not rows:
        log.error("no variants scored"); sys.exit(1)

    out = pd.DataFrame(rows)
    out_path = PROJECT_ROOT / exp_cfg["log"]["metrics_dir"] / args.out
    out.to_csv(out_path, index=False)
    print(f"\nWrote {out_path} ({len(out)} rows)\n")
    print(out.groupby("variant")["r2"]
             .agg(["mean", "median", "min", "max", "count"]).round(3).to_string())


if __name__ == "__main__":
    main()

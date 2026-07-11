"""Train one model on the unified dataset.

Pipeline order:
  1. Read unified parquet, filter to crop(s).
  2. Apply phenology window from data.yaml (per-crop growing season).
  3. Make temporal split: train ≤ train_end_year, val, test.
  4. Fit FeaturePipeline on TRAIN ONLY (z-score features, anomaly-frame target).
  5. Transform features for all splits, build CropYieldDataset.
  6. Train; save model checkpoint + pipeline JSON.

Usage:
    python scripts/03_train.py --model lstm
    python scripts/03_train.py --model patchtst --crop winter_wheat
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import make_temporal_split
from src.data.dataset import (
    CropYieldDataset, FeaturePipeline, FeatureSpec, apply_phenology_window,
    collate_pad,
)
from src.models import build_model
from src.training import Trainer
from src.utils import (get_logger, load_data_config, load_experiment_config,
                       load_model_config, pick_device, set_seed, PROJECT_ROOT)

log = get_logger("train")


CANDIDATE_PAST = [
    "temperature_air_max_2m", "temperature_air_min_2m",
    "precipitation_height", "radiation_global", "humidity",
    "ndvi", "evi", "lai", "ndwi",
]
CANDIDATE_FUTURE = ["sin_doy", "cos_doy"]
CANDIDATE_STATIC = ["sand", "clay", "soc", "ph", "bulk_density", "landkreis_id"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--crop", default=None)
    ap.add_argument("--no-anomaly", action="store_true",
                    help="Predict absolute yield z-scored globally (not within-"
                         "district anomaly). Pair with landkreis_id as a static "
                         "feature so the model can recover district baselines.")
    ap.add_argument("--suffix", default="",
                    help="Appended to checkpoint filename, e.g. '_abs' for the "
                         "absolute-yield track.")
    args = ap.parse_args()

    data_cfg = load_data_config()
    exp_cfg = load_experiment_config()
    model_cfg = load_model_config(args.model)
    set_seed(exp_cfg["seed"])
    device = pick_device(exp_cfg.get("device", "auto"))

    unified = PROJECT_ROOT / data_cfg["paths"]["unified_dataset"]
    if not unified.exists():
        log.error("Unified dataset missing at %s — run scripts/02_preprocess.py first.",
                  unified)
        sys.exit(1)

    df = pd.read_parquet(unified)
    if args.crop:
        df = df[df["crop"] == args.crop]
    log.info("Loaded %d rows (%d districts, %d crops, %d years)",
             len(df), df["landkreis"].nunique(), df["crop"].nunique(),
             df["year"].nunique())

    df = apply_phenology_window(df, data_cfg["phenology"])
    log.info("After phenology windowing: %d rows", len(df))

    split = make_temporal_split(
        df,
        train_end_year=data_cfg["splits"]["temporal"]["train_end_year"],
        val_years=data_cfg["splits"]["temporal"]["val_years"],
        test_years=data_cfg["splits"]["temporal"]["test_years"],
    )
    df_train = df.loc[split.train]
    df_val = df.loc[split.val]
    log.info("Split sizes: train=%d, val=%d, test=%d",
             len(df_train), len(df_val), len(split.test))

    spec = FeatureSpec(
        past_observed=[c for c in CANDIDATE_PAST if c in df.columns],
        static=[c for c in CANDIDATE_STATIC if c in df.columns],
        known_future=[c for c in CANDIDATE_FUTURE if c in df.columns],
    )
    log.info("Features: past=%s, static=%s, future=%s",
             spec.past_observed, spec.static, spec.known_future)

    pipeline = FeaturePipeline(spec, anomaly_framing=not args.no_anomaly).fit(df_train)
    log.info("Pipeline mode: %s", "absolute-yield" if args.no_anomaly else "anomaly-framed")

    # In absolute-yield mode, give the model the per-(landkreis, crop) train
    # mean as an explicit static feature ("target encoding"). Without this,
    # the model has to learn 281 separate baselines from the categorical id,
    # which is impossible with ~16 training samples per district.
    if args.no_anomaly:
        baseline_map = pipeline.target_mean_
        global_mean = pipeline.target_global_mean_
        for f in (df_train, df_val):
            f["district_baseline"] = [
                baseline_map.get((lk, cr), global_mean)
                for lk, cr in zip(f["landkreis"], f["crop"])
            ]
        if "district_baseline" not in spec.static:
            spec.static.append("district_baseline")
        pipeline = FeaturePipeline(spec, anomaly_framing=False).fit(df_train)
        log.info("Added district_baseline static feature.")
    df_train_t = pipeline.transform_features(df_train)
    df_val_t = pipeline.transform_features(df_val)

    bs = model_cfg.get("batch_size", 64)
    train_loader = DataLoader(
        CropYieldDataset(df_train_t, spec, pipeline),
        batch_size=bs, shuffle=True, collate_fn=collate_pad, num_workers=0,
    )
    val_loader = DataLoader(
        CropYieldDataset(df_val_t, spec, pipeline),
        batch_size=bs, shuffle=False, collate_fn=collate_pad, num_workers=0,
    )
    log.info("DataLoaders ready: train=%d samples, val=%d samples",
             len(train_loader.dataset), len(val_loader.dataset))

    model = build_model(args.model, model_cfg)
    trainer = Trainer(
        model=model, device=device,
        max_epochs=exp_cfg["budget"]["max_epochs"],
        early_stopping_patience=exp_cfg["budget"]["early_stopping_patience"],
        max_minutes=exp_cfg["budget"]["max_minutes"],
    )
    result = trainer.fit(train_loader, val_loader)

    out_dir = PROJECT_ROOT / exp_cfg["log"]["metrics_dir"]
    trainer.save_metadata(result, out_dir)

    ckpt_dir = PROJECT_ROOT / exp_cfg["log"]["checkpoints_dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    name = f"{args.model}{args.suffix}"
    try:
        model.save(ckpt_dir / f"{name}.pt")
    except NotImplementedError:
        log.info("Model %s has no save() — skipping checkpoint.", args.model)
    pipeline.save(ckpt_dir / f"{name}.pipeline.json")

    log.info("Done. Checkpoint + pipeline in %s", ckpt_dir)


if __name__ == "__main__":
    main()

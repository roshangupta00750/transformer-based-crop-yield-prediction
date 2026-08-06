"""Train one model on the unified dataset.

Pipeline order:
  1. Read unified parquet, filter to crop(s).
  2. Apply phenology window and minimum sequence-coverage policy.
  3. Make temporal split: train ≤ train_end_year, val, test.
  4. Select leakage-safe features from data.yaml.
  5. Fit FeaturePipeline on TRAIN ONLY (z-score features, anomaly-frame target).
  6. Transform features for all splits, build CropYieldDataset.
  7. Train; save model checkpoint + pipeline JSON.

Usage:
    python scripts/03_train.py --model lstm
    python scripts/03_train.py --model patchtst --crop winter_wheat
"""
from __future__ import annotations

import argparse
import sys
from functools import partial
from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import make_temporal_split
from src.data.dataset import (
    CropYieldDataset, FeaturePipeline, build_feature_spec, collate_pad,
    prepare_modeling_frame, require_current_dataset_policy,
)
from src.models import build_model
from src.training import Trainer
from src.utils import (get_logger, load_data_config, load_experiment_config,
                       load_model_config, pick_device, set_seed, PROJECT_ROOT)

log = get_logger("train")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--crop", default=None)
    ap.add_argument("--no-anomaly", action="store_true",
                    help="Predict absolute yield z-scored globally (not within-"
                         "district anomaly). Pair with landkreis_id as a static "
                         "feature so the model can recover district baselines.")
    ap.add_argument("--config-name", default=None,
                    help="Model config file to load (default: --model). Lets an "
                         "ablation reuse a builder with a different config, e.g. "
                         "--model itransformer --config-name itransformer_temporal.")
    ap.add_argument("--suffix", default="",
                    help="Appended to checkpoint filename, e.g. '_abs' for the "
                         "absolute-yield track.")
    args = ap.parse_args()

    data_cfg = load_data_config()
    exp_cfg = load_experiment_config()
    model_cfg = load_model_config(args.config_name or args.model)
    set_seed(exp_cfg["seed"])
    device = pick_device(exp_cfg.get("device", "auto"))

    unified = PROJECT_ROOT / data_cfg["paths"]["unified_dataset"]
    if not unified.exists():
        log.error("Unified dataset missing at %s — run scripts/02_preprocess.py first.",
                  unified)
        sys.exit(1)
    try:
        require_current_dataset_policy(unified)
    except RuntimeError as e:
        log.error("%s", e)
        sys.exit(1)

    df = pd.read_parquet(unified)
    if args.crop:
        df = df[df["crop"] == args.crop]
    log.info("Loaded %d rows (%d districts, %d crops, %d years)",
             len(df), df["landkreis"].nunique(), df["crop"].nunique(),
             df["year"].nunique())

    rows_before_quality = len(df)
    df = prepare_modeling_frame(df, data_cfg)
    log.info(
        "After crop window + %.0f%% sequence coverage filter: %d rows "
        "(dropped %d)",
        100 * data_cfg["quality"]["min_sequence_coverage"],
        len(df),
        rows_before_quality - len(df),
    )

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

    # Per-model static overrides are still passed through the global exclusion
    # policy, so a model cannot accidentally re-enable a leaking feature.
    spec = build_feature_spec(
        df,
        data_cfg,
        static_features=model_cfg.get("static_features"),
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
    sequence_length = int(
        df.groupby(["landkreis", "crop", "year"])["doy"].nunique().max()
    )
    pipeline.sequence_length_ = sequence_length
    collate = partial(collate_pad, pad_to=sequence_length)
    log.info("Fixed sequence padding length: %d", sequence_length)
    df_train_t = pipeline.transform_features(df_train)
    df_val_t = pipeline.transform_features(df_val)

    bs = model_cfg.get("batch_size", 64)
    train_loader = DataLoader(
        CropYieldDataset(df_train_t, spec, pipeline),
        batch_size=bs, shuffle=True, collate_fn=collate, num_workers=0,
    )
    val_loader = DataLoader(
        CropYieldDataset(df_val_t, spec, pipeline),
        batch_size=bs, shuffle=False, collate_fn=collate, num_workers=0,
    )
    log.info("DataLoaders ready: train=%d samples, val=%d samples",
             len(train_loader.dataset), len(val_loader.dataset))

    model = build_model(args.model, model_cfg)
    # Per-model budget overrides (model_cfg wins, else experiment.yaml default)
    _budget = exp_cfg["budget"]
    trainer = Trainer(
        model=model, device=device,
        max_epochs=model_cfg.get("max_epochs", _budget["max_epochs"]),
        early_stopping_patience=model_cfg.get(
            "early_stopping_patience", _budget["early_stopping_patience"]),
        max_minutes=model_cfg.get("max_minutes", _budget["max_minutes"]),
    )
    result = trainer.fit(train_loader, val_loader)

    # A per-crop run MUST NOT write to the all-crop artefact names. --crop
    # only filtered rows; naming came from --suffix alone, so any caller that
    # passed --crop without also passing a matching --suffix silently
    # overwrote results/checkpoints/<model>.pt and <model>_meta.json — the
    # all-crop artefacts every dashboard page reads. The dashboard's Train
    # page did exactly that. Deriving the suffix here means no caller can
    # repeat it; an explicit --suffix still wins, which is what the sbatch
    # arrays pass.
    suffix = args.suffix
    if not suffix and args.crop:
        suffix = f"_{args.crop}"
    name = f"{args.model}{suffix}"

    out_dir = PROJECT_ROOT / exp_cfg["log"]["metrics_dir"]
    trainer.save_metadata(result, out_dir, name=name)

    ckpt_dir = PROJECT_ROOT / exp_cfg["log"]["checkpoints_dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    try:
        model.save(ckpt_dir / f"{name}.pt")
    except NotImplementedError:
        log.info("Model %s has no save() — skipping checkpoint.", args.model)
    # TFT resolves its encoder window from the data (short-season crops cannot
    # use the 180-step default). Record it so evaluation rebuilds the reference
    # TimeSeriesDataSet with the same geometry the checkpoint learned on.
    _tft_enc = getattr(model, "max_encoder_length_", None)
    if _tft_enc is not None:
        pipeline.tft_max_encoder_length_ = int(_tft_enc)
        log.info("TFT max_encoder_length: %d", _tft_enc)
    pipeline.save(ckpt_dir / f"{name}.pipeline.json")

    log.info("Done. Checkpoint + pipeline in %s", ckpt_dir)


if __name__ == "__main__":
    main()

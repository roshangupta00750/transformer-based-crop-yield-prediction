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
import os
import sys
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import make_temporal_split
from src.data.dataset import (load_modeling_frame, phenology_columns, PHENOLOGY_SLOTS,
    PHENOLOGY_SLOTS_V3, STAGE_COLUMNS, fit_district_climatology, add_district_anomalies,
    anomaly_columns, ANOMALY_BASE, reliable_series, FLOWER_COLUMNS, STRESS_COLUMNS, build_yield_history, add_yield_lags, YIELD_LAG_COLUMNS, 
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
    ap.add_argument("--test-year", type=int, default=None,
                    help="Rolling-origin split for one harvest year Y: train <= Y-2, "
                         "validate Y-1, test Y (overrides data.yaml splits).")
    ap.add_argument("--train-end", type=int, default=None,
                    help="Explicit split (replicating another paper's protocol): "
                         "train <= this year. Use with --val-years/--test-years.")
    ap.add_argument("--val-years", default=None,
                    help="Comma-separated validation years for --train-end.")
    ap.add_argument("--test-years", default=None,
                    help="Comma-separated test years for --train-end.")
    ap.add_argument("--val-random-frac", type=float, default=None,
                    help="Instead of validation years: hold out this fraction of "
                         "the training crop-years at random (not year-blocked), "
                         "as papers using random CV do. Use with --train-end.")
    ap.add_argument("--seed", type=int, default=None,
                    help="Override experiment.yaml seed (for repeat runs).")
    ap.add_argument("--suffix", default="",
                    help="Appended to checkpoint filename, e.g. '_abs' for the "
                         "absolute-yield track.")
    args = ap.parse_args()

    data_cfg = load_data_config()
    exp_cfg = load_experiment_config()
    model_cfg = load_model_config(args.config_name or args.model)
    set_seed(args.seed if args.seed is not None else exp_cfg["seed"])
    device = pick_device(exp_cfg.get("device", "auto"))

    # model config may point at another dataset build (e.g. dataset_v4, 2000-2025)
    unified = PROJECT_ROOT / (model_cfg.get("dataset") or data_cfg["paths"]["unified_dataset"])
    phenology = model_cfg.get("phenology")          # None | "full" | "rt"
    national = bool(model_cfg.get("national", False))
    phenology_version = model_cfg.get("phenology_version", "v2")  # v3: + harvest slot
    stage_features = bool(model_cfg.get("stage_features", False))
    district_anomaly = bool(model_cfg.get("district_anomaly", False))
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
    window = model_cfg.get("window", "calendar")
    common_sample = bool(model_cfg.get("common_sample", False))
    if args.crop:
        df = prepare_modeling_frame(df, data_cfg, window=window,
                                    common_sample=common_sample)
    else:
        # cached per (dataset, window, sample rule): the crop-year frame takes
        # ~13 min to build and every rolling run needs the same one
        df = load_modeling_frame(unified, data_cfg, window=window,
                                 common_sample=common_sample,
                                 cache_dir=PROJECT_ROOT / "data/processed/cache",
                                 phenology=phenology,
                                 phenology_version=phenology_version,
                                 stage_features=stage_features,
                                 flowering_index=bool(model_cfg.get("flowering_index", False)),
                                 crop_specific_sat=bool(model_cfg.get("crop_specific_sat", False)),
                                 stress_features=bool(model_cfg.get("stress_features", False)))
    if model_cfg.get("crops"):
        # preregistration v5 §2.2: one model per crop group (autumn-/spring-sown)
        df = df[df["crop"].isin(model_cfg["crops"])]
        log.info("Crop subset: %s", sorted(df["crop"].unique()))
    log.info("Season window: %s | common sample: %s", window, common_sample)
    log.info(
        "After crop window + %.0f%% sequence coverage filter: %d rows "
        "(dropped %d)",
        100 * data_cfg["quality"]["min_sequence_coverage"],
        len(df),
        rows_before_quality - len(df),
    )

    if args.train_end is not None:
        _years = lambda v: [int(x) for x in v.split(",")] if v else []
        split = make_temporal_split(df, train_end_year=args.train_end,
                                    val_years=_years(args.val_years),
                                    test_years=_years(args.test_years))
    else:
        split = make_temporal_split(
            df,
            train_end_year=(args.test_year - 2 if args.test_year
                            else data_cfg["splits"]["temporal"]["train_end_year"]),
            val_years=([args.test_year - 1] if args.test_year
                       else data_cfg["splits"]["temporal"]["val_years"]),
            test_years=([args.test_year] if args.test_year
                        else data_cfg["splits"]["temporal"]["test_years"]),
        )
    df_train = df.loc[split.train]
    df_val = df.loc[split.val]
    held = None
    if args.val_random_frac:
        # whole crop-year sequences, never split inside a sequence
        keys = df_train[["landkreis", "crop", "year"]].drop_duplicates()
        held = keys.sample(frac=args.val_random_frac,
                           random_state=args.seed if args.seed is not None else exp_cfg["seed"])
        is_val = (df_train.set_index(["landkreis", "crop", "year"]).index
                  .isin(pd.MultiIndex.from_frame(held)))
        df_val, df_train = df_train[is_val], df_train[~is_val]
        log.info("Random (not year-blocked) validation: %d of %d crop-years",
                 len(held), len(keys))
    df_train_fit = df_train          # baselines, scaling, district normals: ALL training rows
    if model_cfg.get("reliable_series"):
        # preregistration v12 §2: train only on district-crop series reported in
        # >= max(5, 0.6 x n) of the n training years; validation/test untouched
        n0 = df_train.groupby(["landkreis", "crop", "year"]).ngroups
        df_train = reliable_series(df_train)
        log.info("Reliable series: %d -> %d training crop-years",
                 n0, df_train.groupby(["landkreis", "crop", "year"]).ngroups)
    log.info("Split sizes: train=%d, val=%d, test=%d",
             len(df_train), len(df_val), len(split.test))

    # Per-model static overrides are still passed through the global exclusion
    # policy, so a model cannot accidentally re-enable a leaking feature.
    spec = build_feature_spec(
        df,
        data_cfg,
        static_features=model_cfg.get("static_features"),
        drop_features=model_cfg.get("drop_features"),
    )
    if model_cfg.get("crop_specific_sat") and "cs_fallback" not in spec.past_observed:
        spec.past_observed.append("cs_fallback")              # v13 §2.3
    if phenology:
        slots = PHENOLOGY_SLOTS_V3 if phenology_version == "v3" else PHENOLOGY_SLOTS
        extra = (phenology_columns(slots) + (STAGE_COLUMNS if stage_features else [])
                 + (STRESS_COLUMNS if model_cfg.get("stress_features") else [])
                 + (FLOWER_COLUMNS if model_cfg.get("flowering_index") else []))
        spec.past_observed += [c for c in extra if c not in spec.past_observed]
    district_clim = None
    if district_anomaly:
        # preregistration v3 §2.3: district-month climatology from training years only
        district_clim = fit_district_climatology(
            df_train_fit, ANOMALY_BASE + (["ndyi_crop"] if model_cfg.get("flowering_index") else []))
        df_train = add_district_anomalies(df_train, district_clim)
        df_val = add_district_anomalies(df_val, district_clim)
        spec.past_observed += anomaly_columns(district_clim)
    log.info("Features: past=%s, static=%s, future=%s",
             spec.past_observed, spec.static, spec.known_future)

    pipeline = FeaturePipeline(spec, anomaly_framing=not args.no_anomaly,
                               trend_test=model_cfg.get("trend_test", "district_year"))
    pipeline.trend_district_ref = bool(model_cfg.get("trend_district_ref", False))   # v10 §2.3
    if model_cfg.get("recent_baseline"):
        # preregistration v7 §2: reported yields up to the fold's last
        # non-test year (rolling: Y-1, known at harvest of Y)
        hist_end = int(max(df_train["year"].max(), df_val["year"].max() if len(df_val) else -1))
        yl = pd.read_parquet(unified, columns=["landkreis", "crop", "year", spec.target])
        yl = yl.dropna().drop_duplicates(["landkreis", "crop", "year"])
        pipeline.recent_baseline_ = True
        pipeline.baseline_blend_ = bool(model_cfg.get("baseline_blend", False))   # v9 §2
        pipeline.yield_hist_ = build_yield_history(yl[yl["year"] <= hist_end], spec.target)
        df_train = add_yield_lags(df_train, pipeline.yield_hist_)
        df_val = add_yield_lags(df_val, pipeline.yield_hist_)
        spec.past_observed += [c for c in YIELD_LAG_COLUMNS if c not in spec.past_observed]
        log.info("Recent baseline on; yield history up to %d", hist_end)
    if district_anomaly and df_train_fit is not df_train:
        df_train_fit = add_district_anomalies(df_train_fit, district_clim)
    if model_cfg.get("recent_baseline") and df_train_fit is not df_train:
        df_train_fit = add_yield_lags(df_train_fit, pipeline.yield_hist_)
    pipeline = pipeline.fit(df_train_fit)
    if getattr(pipeline, "baseline_blend_", False):
        log.info("Recent-level weights (training years): %s",
                 {k: round(v, 2) for k, v in pipeline.recent_weight_by_crop_.items()})
    log.info("Trend test: %s | slopes: %s", pipeline.trend_test,
             {k: round(v, 3) for k, v in pipeline.target_trend_by_crop_.items()})
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
    pipeline.district_clim_ = district_clim
    pipeline.phenology_version_ = phenology_version if phenology else None
    pipeline.stage_features_ = stage_features
    sequence_length = int(
        df.groupby(["landkreis", "crop", "year"])["doy"].nunique().max()
    )
    pipeline.sequence_length_ = sequence_length
    pipeline.window_ = window
    pipeline.common_sample_ = common_sample
    collate = partial(collate_pad, pad_to=sequence_length)
    log.info("Fixed sequence padding length: %d", sequence_length)
    df_train_t = pipeline.transform_features(df_train)
    df_val_t = pipeline.transform_features(df_val)
    pipeline.phenology_ = phenology

    # National component (preregistration v2 §2.3): a per-crop ridge predicts
    # each year's national anomaly; the transformer learns the deviation from it.
    target_offset = None
    if national:
        from src.models.national import NationalModel, season_features

        def crop_year_z(frame):
            k = frame.groupby(["landkreis", "crop", "year"], as_index=False)[spec.target].first()
            z = [pipeline.transform_target(lk, cr, yv, yr) for lk, cr, yv, yr
                 in zip(k["landkreis"], k["crop"], k[spec.target], k["year"])]
            return k.assign(z=z).groupby(["crop", "year"])["z"].mean()

        nat_train = crop_year_z(df_train)
        nm = NationalModel().fit(season_features(df_train_t, spec.past_observed, bool(phenology)),
                                 nat_train)
        pipeline.national_ = nm.to_dict()
        target_offset = crop_year_z(pd.concat([df_train, df_val])).to_dict()
        nat_val = crop_year_z(df_val)
        pred_val = nm.predict(season_features(df_val_t, spec.past_observed, bool(phenology)))
        for (cr, yr), v in nat_val.items():
            log.info("national %s %d: actual %+.3f  ridge %+.3f  (alpha %.3g, %d train years)",
                     cr, yr, v, pred_val.get((cr, yr), float("nan")),
                     nm.models[cr]["alpha"], nm.models[cr]["n_years"])

    bs = model_cfg.get("batch_size", 64)
    sample_weight = None
    if model_cfg.get("area_weights"):                  # preregistration v15 §4, training loss only
        from src.data.dataset import area_weights
        sample_weight = area_weights(df_train_t)
        log.info("area weights: %d training crop-years, mean %.3f", len(sample_weight),
                 float(np.mean(list(sample_weight.values()))))
    train_loader = DataLoader(
        CropYieldDataset(df_train_t, spec, pipeline, target_offset=target_offset,
                         sample_weight=sample_weight),
        batch_size=bs, shuffle=True, collate_fn=collate, num_workers=0,
    )
    val_loader = DataLoader(
        CropYieldDataset(df_val_t, spec, pipeline, target_offset=target_offset),
        batch_size=bs, shuffle=False, collate_fn=collate, num_workers=0,
    )
    log.info("DataLoaders ready: train=%d samples, val=%d samples",
             len(train_loader.dataset), len(val_loader.dataset))

    model = build_model(args.model, model_cfg)
    # Per-model budget overrides (model_cfg wins, else experiment.yaml default)
    _budget = dict(exp_cfg["budget"])
    if os.environ.get("CROP_YIELD_SMOKE"):
        # preflight's end-to-end check: exercise imports, signatures,
        # checkpoint naming and metadata writing without buying an hour of GPU.
        _budget = {**_budget, "max_epochs": 1, "max_minutes": 1,
                   "early_stopping_patience": 1}
        log.info("SMOKE MODE: 1 epoch / 1 minute budget")
    max_epochs = model_cfg.get("max_epochs", _budget["max_epochs"])
    patience = model_cfg.get("early_stopping_patience",
                             _budget["early_stopping_patience"])
    max_minutes = model_cfg.get("max_minutes", _budget["max_minutes"])
    if os.environ.get("CROP_YIELD_SMOKE"):
        # Smoke must beat the per-model config too: model yamls set their own
        # max_epochs/max_minutes, which previously overrode the smoke budget
        # and turned a 1-epoch check into a full 28-minute H100 training run.
        max_epochs, patience, max_minutes = 1, 1, 1
    trainer = Trainer(
        model=model, device=device,
        max_epochs=max_epochs,
        early_stopping_patience=patience,
        max_minutes=max_minutes,
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
    if held is not None:
        # the random holdout, so per-crop fine-tuning (scripts/47) validates
        # on exactly the same crop-years (preregistration v4 §5)
        held.to_parquet(ckpt_dir / f"{name}.valkeys.parquet", index=False)

    log.info("Done. Checkpoint + pipeline in %s", ckpt_dir)


if __name__ == "__main__":
    main()

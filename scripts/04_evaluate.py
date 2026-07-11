"""Evaluate a trained model at multiple in-season horizons.

Usage:
    python scripts/04_evaluate.py --model lstm --crop winter_wheat
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import FeaturePipeline, apply_phenology_window
from src.evaluation.runner import evaluate_at_horizons, save_results
from src.models import build_model
from src.utils import (get_logger, load_data_config, load_experiment_config,
                       load_model_config, PROJECT_ROOT)

log = get_logger("evaluate")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--crop", default="winter_wheat")
    ap.add_argument("--suffix", default="",
                    help="Checkpoint name suffix (matches --suffix used at train).")
    args = ap.parse_args()

    data_cfg = load_data_config()
    exp_cfg = load_experiment_config()
    model_cfg = load_model_config(args.model)

    unified = PROJECT_ROOT / data_cfg["paths"]["unified_dataset"]
    df = pd.read_parquet(unified)
    df = df[df["crop"] == args.crop]

    df = apply_phenology_window(df, data_cfg["phenology"])
    df_test = df[df["year"].isin(data_cfg["splits"]["temporal"]["test_years"])]
    log.info("Test set: %d rows across %d districts × %d years",
             len(df_test), df_test["landkreis"].nunique(), df_test["year"].nunique())

    ckpt_dir = PROJECT_ROOT / exp_cfg["log"]["checkpoints_dir"]
    name = f"{args.model}{args.suffix}"
    pipeline_path = ckpt_dir / f"{name}.pipeline.json"
    if not pipeline_path.exists():
        log.error("Pipeline JSON missing: %s. Retrain via scripts/03_train.py.",
                  pipeline_path)
        sys.exit(1)
    pipeline = FeaturePipeline.load(pipeline_path)

    # If the trained pipeline expects district_baseline (absolute-yield track),
    # synthesize the column on the test set using train-time baselines.
    if "district_baseline" in pipeline.spec.static and "district_baseline" not in df_test.columns:
        gm = pipeline.target_global_mean_
        df_test = df_test.copy()
        df_test["district_baseline"] = [
            pipeline.target_mean_.get((lk, cr), gm)
            for lk, cr in zip(df_test["landkreis"], df_test["crop"])
        ]

    model = build_model(args.model, model_cfg)
    ckpt = None
    for ext in (".pt", ".ubj", ".json"):
        cand = ckpt_dir / f"{name}{ext}"
        if cand.exists():
            ckpt = cand
            break
    if ckpt is None:
        log.error("No checkpoint found for %s in %s — retrain first.",
                  args.model, ckpt_dir)
        sys.exit(1)
    try:
        model.load(ckpt)
    except NotImplementedError:
        log.warning("Model %s cannot load() — re-running fit may be needed.",
                    args.model)

    harvest_doy = data_cfg["phenology"][args.crop]["harvest_doy"]
    results = evaluate_at_horizons(
        model=model,
        df_test=df_test,
        feature_spec=pipeline.spec,
        pipeline=pipeline,
        harvest_doy=harvest_doy,
        horizons_days=data_cfg["in_season_horizons_days"],
        batch_size=model_cfg.get("batch_size", 128),
    )

    out_path = (PROJECT_ROOT / exp_cfg["log"]["metrics_dir"]
                / f"{name}_{args.crop}_horizons.json")
    save_results(results, out_path)
    log.info("Saved %s", out_path)
    for h, m in results.items():
        log.info("%s: R²=%.3f  RMSE=%.2f  MAE=%.2f  n=%d",
                 h, m["r2"], m["rmse"], m["mae"], m["n"])


if __name__ == "__main__":
    main()

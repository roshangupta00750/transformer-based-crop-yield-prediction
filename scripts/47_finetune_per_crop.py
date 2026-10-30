"""Per-crop fine-tuning of an all-crop model (paper/preregistration_2026-09-29_v4.md §2.2).

For one rolling fold (test year Y, seed s): load the all-crop checkpoint
<base>_roll{Y}_s{s} and its pipeline (input scaling, baseline, district
climatology are reused unchanged), then for every crop try the registered
candidates and keep the one with the lowest loss on that crop's validation
year Y-1:

    0  no fine-tuning (the all-crop model as is)
    1  lr 1e-4, dropout 0.15      2  lr 1e-4, dropout 0.30
    3  lr 3e-5, dropout 0.15      4  lr 3e-5, dropout 0.30

Training uses that crop's years <= Y-2 only, up to 20 epochs, early stopping
(patience 4) with best-epoch weights on Y-1. Year Y is never read.

Output (results/checkpoints/): <out>_roll{Y}_s{s}_<crop>.pt for each crop and
<out>_roll{Y}_s{s}.finetune.json with every candidate's validation loss.

Paper protocols (preregistration v4 §5): --tag cmp<TAG> --train-end E and either
--val-years (AgroForecast-DE, CropFusionNet) or the random 15% holdout the base
run saved as <base>_<tag>_s<seed>.valkeys.parquet (Srivastava). The test years
are never read.

Usage: python scripts/47_finetune_per_crop.py --test-year 2020 --seed 42
           [--base itransformer_v4T] [--out itransformer_v4F]
       python scripts/47_finetune_per_crop.py --tag cmpAFDE --train-end 2019 \
           --val-years 2020,2021,2022 --seed 42
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from functools import partial
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.data.dataset import (CropYieldDataset, FeaturePipeline, collate_pad,  # noqa: E402
                              load_modeling_frame)
from src.models import build_model  # noqa: E402
from src.utils import load_data_config, load_model_config, set_seed  # noqa: E402

CANDIDATES = [None, (1e-4, 0.15), (1e-4, 0.30), (3e-5, 0.15), (3e-5, 0.30)]
MAX_EPOCHS, PATIENCE, MAX_MINUTES = 20, 4, 15


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--test-year", type=int)
    ap.add_argument("--tag", help="run tag, default roll<test-year>")
    ap.add_argument("--train-end", type=int, help="protocol split: last training year")
    ap.add_argument("--val-years", default="", help="protocol split: validation years")
    ap.add_argument("--crops", default="", help="only these crops (default: the base config's crops, else all)")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--base", default="itransformer_v4T")
    ap.add_argument("--out", default="itransformer_v4F")
    args = ap.parse_args()
    smoke = bool(os.environ.get("CROP_YIELD_SMOKE"))
    max_epochs, patience = (1, 1) if smoke else (MAX_EPOCHS, PATIENCE)
    Y, s = args.test_year, args.seed
    if args.train_end is None and Y is None:
        ap.error("give --test-year (rolling fold) or --train-end (protocol)")
    tag = args.tag or f"roll{Y}"
    set_seed(s)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ck = ROOT / "results/checkpoints"
    base_name = f"{args.base}_{tag}_s{s}" + ("_smoke" if smoke else "")
    out_name = f"{args.out}_{tag}_s{s}" + ("_smoke" if smoke else "")
    cfg = load_model_config(args.base)
    pipe = FeaturePipeline.load(ck / f"{base_name}.pipeline.json")
    pipe.require_current_feature_policy()
    frame = load_modeling_frame(ROOT / cfg["dataset"], load_data_config(), window=pipe.window_,
                                common_sample=pipe.common_sample_,
                                cache_dir=ROOT / "data/processed/cache",
                                phenology=cfg.get("phenology"),
                                phenology_version=cfg.get("phenology_version", "v2"),
                                stage_features=bool(cfg.get("stage_features", False)),
                                flowering_index=bool(cfg.get("flowering_index", False)),
                                crop_specific_sat=bool(cfg.get("crop_specific_sat", False)),
                                stress_features=bool(cfg.get("stress_features", False)))
    if args.train_end is None:                       # rolling fold: train <= Y-2, validate Y-1
        tr, va = frame[frame["year"] <= Y - 2], frame[frame["year"] == Y - 1]
        split = {"train_end": Y - 2, "val_years": [Y - 1]}
    else:
        tr = frame[frame["year"] <= args.train_end]
        vk = ck / f"{base_name}.valkeys.parquet"
        if args.val_years:
            vy = [int(v) for v in args.val_years.split(",")]
            va = frame[frame["year"].isin(vy)]
            split = {"train_end": args.train_end, "val_years": vy}
        elif vk.exists():                            # the base run's own random holdout
            key = ["landkreis", "crop", "year"]
            held = pd.MultiIndex.from_frame(pd.read_parquet(vk)[key])
            is_val = pd.MultiIndex.from_frame(tr[key]).isin(held)
            tr, va = tr[~is_val], tr[is_val]
            split = {"train_end": args.train_end, "val_keys": vk.name, "n_val_keys": len(held)}
        else:
            sys.exit(f"no --val-years and no {vk.name}")
    if cfg.get("reliable_series"):                     # preregistration v12 §2, training rows only
        from src.data.dataset import reliable_series
        tr = reliable_series(tr)
    train, val = pipe.transform_features(tr), pipe.transform_features(va)
    del frame, tr, va
    collate = partial(collate_pad, pad_to=pipe.sequence_length_)
    bs = cfg.get("batch_size", 64)

    arch = cfg.get("name", "itransformer")            # v17: tftnet / patchtst17 on the same loop
    base = build_model(arch, cfg)
    base.load(ck / f"{base_name}.pt")
    base.net.to(device); base.device = device
    init = {k: v.detach().clone() for k, v in base.net.state_dict().items()}
    mse = nn.MSELoss()

    log = {"base": base_name, "tag": tag, "test_year": Y, "seed": s, "split": split, "crops": {}}
    only = [c for c in args.crops.split(",") if c] or cfg.get("crops") or []
    crops = sorted(c for c in train["crop"].unique() if not only or c in only)[: 2 if smoke else None]
    for crop in crops:
        t0 = time.time()
        sw = None
        if cfg.get("area_weights"):                    # preregistration v15 §4, training loss only
            from src.data.dataset import area_weights
            sw = area_weights(train[train["crop"] == crop])
        tl = DataLoader(CropYieldDataset(train[train["crop"] == crop], pipe.spec, pipe, sample_weight=sw),
                        batch_size=bs, shuffle=True, collate_fn=collate, num_workers=0)
        vl = DataLoader(CropYieldDataset(val[val["crop"] == crop], pipe.spec, pipe),
                        batch_size=bs, shuffle=False, collate_fn=collate, num_workers=0)
        base.net.load_state_dict(init)
        results = [{"candidate": 0, "lr": None, "dropout": None, "val_loss": base._eval(vl, mse)}]
        best_model, best_loss = base, results[0]["val_loss"]
        best_state = {k: v.clone() for k, v in init.items()}
        for i, cand in enumerate(CANDIDATES[1:], start=1):
            lr, dropout = cand
            set_seed(s)
            m = build_model(arch, {**base.config, "dropout": dropout})
            m.fit(tl, vl, max_epochs=max_epochs, lr=lr, patience=patience, device=device,
                  max_minutes=MAX_MINUTES, init_state=init)
            loss = m._eval(vl, mse)
            results.append({"candidate": i, "lr": lr, "dropout": dropout, "val_loss": loss,
                            "best_epoch": getattr(m, "best_epoch_", None), "stop_reason": getattr(m, "stop_reason_", None),
                            "history": getattr(m, "history_", [])})
            if loss < best_loss:
                best_model, best_loss = m, loss
                best_state = {k: v.detach().clone() for k, v in m.net.state_dict().items()}
        chosen = min(results, key=lambda r: r["val_loss"])
        best_model.net.load_state_dict(best_state)
        best_model.save(ck / f"{out_name}_{crop}.pt")
        log["crops"][crop] = {"chosen": chosen["candidate"], "candidates": results,
                              "n_train": len(tl.dataset), "n_val": len(vl.dataset),
                              "minutes": round((time.time() - t0) / 60, 1)}
        print(f"{crop:16s} chose {chosen['candidate']} "
              + " ".join(f"{r['candidate']}:{r['val_loss']:.4f}" for r in results), flush=True)
    (ck / f"{out_name}.finetune.json").write_text(json.dumps(log, indent=1))


if __name__ == "__main__":
    main()

"""Predict with the replication-protocol checkpoints (scripts/sbatch_compare.sh).

Checkpoints itransformer_preD_cmp{AFDE,CFN,SRV2017,SRV2018,SRV2019}_s{42,43,44}.
Each is scored on the years its protocol reports (preregistration change log,
2026-09-27): AFDE 2023-24; CFN 2019-24 (2019-21 = their validation years,
kept separate via the `role` column); SRV<Y> year Y.

Output: results/metrics/compare_predictions.parquet -- one row per
(protocol, seed, year, crop, landkreis) incl. seed="ens" (3-seed mean), with
y_true, y_pred, y_hist and role (test / val).
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.data.dataset import FeaturePipeline, load_modeling_frame  # noqa: E402
from src.models import build_model  # noqa: E402
from src.utils import load_data_config, load_model_config  # noqa: E402

_s = importlib.util.spec_from_file_location("ev", ROOT / "scripts/34_eval_prereg.py")
ev = importlib.util.module_from_spec(_s); _s.loader.exec_module(ev)

# CMP_ARM selects the arm (default preD, the 2026-09-27 run; v3A = preregistration v3 §5)
ARM = os.environ.get("CMP_ARM", "preD")
OUT_SUFFIX = "" if ARM == "preD" else f"_{ARM}"
# v18: TFT / PatchTST arms use their own checkpoint prefix
PREFIX = {"v17TF": "tftnet", "v17PF": "patchtst17"}.get(ARM, "itransformer")
PAT = re.compile(rf"^{PREFIX}_{ARM}_cmp(?P<tag>AFDE|CFN|SRV\d{{4}})_s(?P<seed>\d+)$")
# v4F (preregistration v4 §5): one fine-tuned checkpoint per crop,
# <out>_cmp<TAG>_s<seed>_<crop>.pt, listed in <out>_cmp<TAG>_s<seed>.finetune.json,
# scored with the base run's pipeline
FINETUNED = ARM in ("v4F", "v7RF", "v9BF", "v10DF", "v12QF", "v13CF", "v14HF", "v15WF", "v17TF", "v17PF")
BASE_CFG = {"v4F": "itransformer_v4T", "v7RF": "itransformer_v7R", "v9BF": "itransformer_v9B", "v10DF": "itransformer_v10D", "v12QF": "itransformer_v12Q", "v13CF": "itransformer_v13C", "v14HF": "itransformer_v14H", "v15WF": "itransformer_v15W", "v17TF": "tftnet_v17T", "v17PF": "patchtst17_v17P"}
YEARS = {"AFDE": {2023: "test", 2024: "test"},
         "CFN": {2019: "val", 2020: "val", 2021: "val", 2022: "test", 2023: "test", 2024: "test"}}


def main() -> None:
    dc = load_data_config()
    ck = ROOT / "results/checkpoints"
    if FINETUNED:
        runs = [(m["tag"], m["seed"], n) for f in sorted(ck.glob(f"{PREFIX}_{ARM}_cmp*.finetune.json"))
                if (m := PAT.match(n := f.name[: -len(".finetune.json")]))]
    else:
        runs = [(m["tag"], m["seed"], f.stem) for f in sorted(ck.glob(f"itransformer_{ARM}_cmp*.pt"))
                if (m := PAT.match(f.stem))]
    if len(runs) != 15:
        sys.exit(f"expected 15 checkpoints, found {len(runs)}")
    cfg_name = BASE_CFG[ARM] if FINETUNED else f"itransformer_{ARM}"
    cfg = load_model_config(cfg_name)
    frame = load_modeling_frame(ROOT / cfg.get("dataset", dc["paths"]["unified_dataset"]), dc,
                                window="crop_year", common_sample=True,
                                cache_dir=ROOT / "data/processed/cache",
                                phenology=cfg.get("phenology"),
                                phenology_version=cfg.get("phenology_version", "v2"),
                                stage_features=bool(cfg.get("stage_features", False)),
                                crop_specific_sat=bool(cfg.get("crop_specific_sat", False)))
    out = []
    for tag, seed, name in runs:
        log = json.loads((ck / f"{name}.finetune.json").read_text()) if FINETUNED else None
        pipe = FeaturePipeline.load(ck / f"{log['base'] if FINETUNED else name}.pipeline.json")
        pipe.require_current_feature_policy()
        assert (pipe.window_, pipe.common_sample_) == ("crop_year", True)
        years = YEARS[tag] if tag in YEARS else {int(tag[3:]): "test"}
        rows = frame[frame["year"].isin(list(years))]
        if FINETUNED:
            parts = []
            for crop in sorted(rows["crop"].unique()):
                assert crop in log["crops"], f"{name}: no fine-tuned model for {crop}"
                model = build_model(cfg.get("name", "itransformer"), cfg)
                model.load(ck / f"{name}_{crop}.pt")
                if PREFIX != "itransformer" and torch.cuda.is_available():   # v18: PatchTST is slow on CPU
                    model.net.to("cuda"); model.device = "cuda"
                parts.append(ev.predict(model, pipe, rows[rows["crop"] == crop]))
            p = pd.concat(parts, ignore_index=True)
        else:
            model = build_model("itransformer", cfg)
            model.load(ck / f"{name}.pt")
            p = ev.predict(model, pipe, rows)
        out.append(p.assign(protocol=tag, seed=seed, role=p["year"].map(years)))
        print(f"  {name}: {len(p)} predictions", flush=True)
    P = pd.concat(out, ignore_index=True)
    k = ["protocol", "role", "year", "crop", "landkreis"]
    ens = (P.groupby(k, as_index=False)
             .agg(y_true=("y_true", "first"), y_hist=("y_hist", "first"), y_pred=("y_pred", "mean"),
                  n_seeds=("seed", "nunique")))
    assert (ens["n_seeds"] == 3).all()
    P = pd.concat([P, ens.drop(columns="n_seeds").assign(seed="ens")], ignore_index=True)
    P.to_parquet(ROOT / f"results/metrics/compare_predictions{OUT_SUFFIX}.parquet", index=False)
    print(f"wrote {len(P):,} rows")


if __name__ == "__main__":
    main()

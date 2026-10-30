"""Evaluate pre-registration v11 (leave-one-out of the v10 changes) on the hindcast 2010-2015.

Arms W, T, M, Y (itransformer_v11{X}F_roll{Y}_s{s}_<crop>.pt, pipelines itransformer_v11{X}_roll{Y}_s{s})
vs D10 (results/metrics/v10_predictions.parquet) and v4 F (results/metrics/v7_predictions.parquet),
on the v10 decision rows: district-crop-years present everywhere with the same reported yield.
Reference: v4 F's y_hist. Output: results/metrics/v11_scores.json
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.data.dataset import FeaturePipeline, load_modeling_frame  # noqa: E402
from src.models import build_model  # noqa: E402
from src.utils import load_data_config, load_model_config  # noqa: E402

_s = importlib.util.spec_from_file_location("e7", ROOT / "scripts/51_eval_v7.py")
e7 = importlib.util.module_from_spec(_s); _s.loader.exec_module(e7)
SEEDS, YEARS, ARMS = ("42", "43", "44"), range(2010, 2016), ("W", "T", "M", "Y")
K = ["year", "crop", "landkreis"]


def predict(arm):
    ck = ROOT / "results/checkpoints"; cfg = load_model_config(f"itransformer_v11{arm}")
    frame = load_modeling_frame(ROOT / cfg["dataset"], load_data_config(), window="crop_year", common_sample=True,
                                cache_dir=ROOT / "data/processed/cache", phenology=cfg["phenology"],
                                phenology_version=cfg["phenology_version"], stage_features=bool(cfg["stage_features"]))
    out = []
    for Y in YEARS:
        fy = frame[frame["year"] == Y]
        for s in SEEDS:
            pipe = FeaturePipeline.load(ck / f"itransformer_v11{arm}_roll{Y}_s{s}.pipeline.json")
            for crop in sorted(fy["crop"].unique()):
                m = build_model("itransformer", cfg); m.load(ck / f"itransformer_v11{arm}F_roll{Y}_s{s}_{crop}.pt")
                out.append(e7.v2.predict(m, pipe, fy[fy["crop"] == crop]).assign(seed=s))
        print(f"  {arm} {Y}", flush=True)
    P = pd.concat(out, ignore_index=True)
    return P.groupby(K, as_index=False).agg(y_true=("y_true", "first"), pred=("y_pred", "mean"), n=("seed", "nunique"))


def boot(D, a, b):
    yy = np.array(sorted(D.year.unique())); rng = np.random.default_rng(0); by = {y: D[D.year == y] for y in yy}
    d = []
    for _ in range(10000):
        S = pd.concat([by[y] for y in rng.choice(yy, len(yy))], ignore_index=True)
        d.append(e7.macro_skill(S, a, "y_hist") - e7.macro_skill(S, b, "y_hist"))
    return np.array(d)


def main():
    v4 = pd.read_parquet(ROOT / "results/metrics/v7_predictions.parquet")[K + ["y_true", "y_hist", "F"]]
    d10 = pd.read_parquet(ROOT / "results/metrics/v10_predictions.parquet")[K + ["y_true", "D10"]]
    D = v4.merge(d10, on=K, suffixes=("", "_10"))
    D = D[D.year.isin(list(YEARS)) & ((D.y_true - D.y_true_10).abs() < 0.05)]
    for arm in ARMS:
        P = predict(arm); assert (P["n"] == 3).all()
        D = D.merge(P.rename(columns={"pred": arm, "y_true": f"y_true_{arm}"})[K + [arm, f"y_true_{arm}"]], on=K)
        D = D[(D.y_true - D[f"y_true_{arm}"]).abs() < 0.05]
    D.to_parquet(ROOT / "results/metrics/v11_predictions.parquet", index=False)
    mape = lambda c: float(100 * ((D[c] - D.y_true).abs() / D.y_true).mean())
    res = {"n_rows": int(len(D)), "accuracy": {c: 100 - mape(c) for c in ("y_hist", "F", "D10") + ARMS},
           "skill_vs_yhist": {c: e7.macro_skill(D, c, "y_hist") for c in ("F", "D10") + ARMS}}
    verdict = {}
    names = {"W": "ERA5 weather everywhere", "T": "district trend reference", "M": "MODIS end dating", "Y": "AGS-matched yields"}
    for arm in ARMS:
        d = boot(D, arm, "D10")                       # undoing the change minus D10
        harmful = (d > 0).mean() >= 0.90 and mape(arm) < mape("D10")
        helpful = (d < 0).mean() >= 0.90
        verdict[names[arm]] = {"delta_skill_undo_minus_D10": float(res["skill_vs_yhist"][arm] - res["skill_vs_yhist"]["D10"]),
                               "ci90": [float(np.quantile(d, .05)), float(np.quantile(d, .95))],
                               "p_undo_better": float((d > 0).mean()),
                               "verdict": "harmful" if harmful else ("helpful" if helpful else "no detectable effect")}
    res["verdict"] = verdict
    per = {c: {cr: float(100 - 100 * ((g[c] - g.y_true).abs() / g.y_true).mean()) for cr, g in D.groupby("crop")}
           for c in ("F", "D10") + ARMS}
    res["accuracy_by_crop"] = per
    (ROOT / "results/metrics/v11_scores.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: res[k] for k in ("n_rows", "accuracy", "skill_vs_yhist", "verdict")}, indent=1))


if __name__ == "__main__":
    main()

"""Evaluate pre-registration v10 (paper/preregistration_2026-10-03_v10.md §3).

D10 (itransformer_v10DF_roll{Y}_s{s}_<crop>.pt, pipelines itransformer_v10D_roll{Y}_s{s})
against v4 F (results/metrics/v7_predictions.parquet: F, y_hist, b5; 2010-2025).

1. Decision rows: district-crop-years present in both panels with the same reported
   yield (|diff| < 0.05 dt/ha); reference y_hist = v4 F's. v6 §6 on 2010-2015.
2. Full corrected panel: D10 against its own y_hist and B5 (5-year mean of the
   corrected yields), all windows.
Outputs: results/metrics/v10_predictions.parquet, v10_scores.json, v10_per_crop.csv
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
SEEDS = ("42", "43", "44")
WINDOWS = {"hindcast_2010_2015": range(2010, 2016), "dev_2016_2025": range(2016, 2026),
           "dev_2019_2025": range(2019, 2026)}


def predict_d10(frame):
    ck = ROOT / "results/checkpoints"; cfg = load_model_config("itransformer_v10D"); out = []
    for Y in range(2010, 2026):
        fy = frame[frame["year"] == Y]
        for s in SEEDS:
            pipe = FeaturePipeline.load(ck / f"itransformer_v10D_roll{Y}_s{s}.pipeline.json")
            pipe.require_current_feature_policy()
            for crop in sorted(fy["crop"].unique()):
                m = build_model("itransformer", cfg); m.load(ck / f"itransformer_v10DF_roll{Y}_s{s}_{crop}.pt")
                out.append(e7.v2.predict(m, pipe, fy[fy["crop"] == crop]).assign(seed=s))
        print(f"  D10 {Y}", flush=True)
    P = pd.concat(out, ignore_index=True)
    return P.groupby(["year", "crop", "landkreis"], as_index=False).agg(
        y_true=("y_true", "first"), y_hist=("y_hist", "first"), D10=("y_pred", "mean"), n=("seed", "nunique"))


def b5(D, ytab):
    h = ytab.set_index(["landkreis", "crop", "year"])["yield_dt_per_ha"]
    def f(r):
        v = [h.get((r.landkreis, r.crop, t)) for t in range(r.year - 5, r.year)]
        v = [x for x in v if x is not None and np.isfinite(x)]
        return float(np.mean(v)) if len(v) >= 3 else r.y_hist
    return [f(r) for r in D.itertuples(index=False)]


def main():
    cfg = load_model_config("itransformer_v10D")
    frame = load_modeling_frame(ROOT / cfg["dataset"], load_data_config(), window="crop_year", common_sample=True,
                                cache_dir=ROOT / "data/processed/cache", phenology=cfg["phenology"],
                                phenology_version=cfg["phenology_version"], stage_features=bool(cfg["stage_features"]))
    P = predict_d10(frame); assert (P["n"] == 3).all()
    P.to_parquet(ROOT / "results/metrics/v10_predictions.parquet", index=False)
    old = pd.read_parquet(ROOT / "results/metrics/v7_predictions.parquet")[["year", "crop", "landkreis", "y_true", "y_hist", "b5", "F"]]
    k = ["year", "crop", "landkreis"]
    S = P.merge(old, on=k, suffixes=("_v5", ""))
    S = S[(S.y_true_v5 - S.y_true).abs() < 0.05].rename(columns={"D10": "R7"})     # R7 slot = D10 in e7.score
    res = {"decision_rows": {w: e7.score(S[S.year.isin(list(y))]) for w, y in WINDOWS.items()}}
    h = res["decision_rows"]["hindcast_2010_2015"]
    h["decision_D10_replaces_F"] = bool(h["p_delta_gt_0"] >= 0.95 and h["mape_R7"] < h["mape_F"]
                                        and not h["crops_worse_every_year"])
    ytab = pd.read_parquet(ROOT / "data/interim/v5/yield.parquet")
    P["b5"] = b5(P, ytab)
    full = {}
    for w, yrs in WINDOWS.items():
        g = P[P.year.isin(list(yrs))]
        full[w] = {"n_rows": int(len(g)), "districts": int(g.landkreis.nunique()),
                   **{f"mape_{c}": float(100 * ((g[c] - g.y_true).abs() / g.y_true).mean()) for c in ("D10", "b5", "y_hist")},
                   "skill_D10_vs_yhist": e7.macro_skill(g.rename(columns={"D10": "x"}), "x", "y_hist"),
                   "skill_D10_vs_b5": e7.macro_skill(g.rename(columns={"D10": "x"}), "x", "b5")}
    res["full_corrected_panel"] = full
    (ROOT / "results/metrics/v10_scores.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    rows = []
    for w, yrs in WINDOWS.items():
        for c, g in P[P.year.isin(list(yrs))].groupby("crop"):
            rows.append({"window": w, "crop": c, **{f"mape_{x}": 100 * ((g[x] - g.y_true).abs() / g.y_true).mean()
                                                    for x in ("D10", "b5", "y_hist")}})
    pd.DataFrame(rows).to_csv(ROOT / "results/metrics/v10_per_crop.csv", index=False)


if __name__ == "__main__":
    main()

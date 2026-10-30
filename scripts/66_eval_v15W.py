"""Evaluate pre-registration v15 arm W15 (area-weighted training), with and without error memory.

W15: itransformer_v15WF_roll{Y}_s{s}_<crop>.pt (pipelines itransformer_v15W_roll{Y}_s{s}).
Q12: results/metrics/v12_predictions.parquet (column D10 = Q12 ensemble prediction).
Same corrected-panel rows, reference = Q12's y_hist. Both get the v15 §2 error memory (lambda causal).
Decision v6 §6 on 2010-2015: WE15 (W15 + memory) vs E15 (Q12 + memory, the main model).
Outputs: results/metrics/v15W_predictions.parquet, v15W_scores.json, v15W_per_crop.csv
"""
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
K, SEEDS = ["year", "crop", "landkreis"], ("42", "43", "44")
WINDOWS = {"hindcast_2010_2015": range(2010, 2016), "dev_2016_2025": range(2016, 2026), "dev_2019_2025": range(2019, 2026)}


HINDCAST_ONLY = "--hindcast-only" in sys.argv      # decision window only (v15 §4), before 2016-2025 finish
TAG = "_hindcast" if HINDCAST_ONLY else ""
if HINDCAST_ONLY:
    WINDOWS = {"hindcast_2010_2015": range(2010, 2016)}


def main():
    cfg = load_model_config("itransformer_v15W"); ck = ROOT / "results/checkpoints"
    frame = load_modeling_frame(ROOT / cfg["dataset"], load_data_config(), window="crop_year", common_sample=True,
                                cache_dir=ROOT / "data/processed/cache", phenology=cfg["phenology"],
                                phenology_version=cfg["phenology_version"], stage_features=bool(cfg["stage_features"]),
                                crop_specific_sat=False)
    out = []
    for Y in (range(2010, 2016) if HINDCAST_ONLY else range(2010, 2026)):
        fy = frame[frame["year"] == Y]
        for s in SEEDS:
            pipe = FeaturePipeline.load(ck / f"itransformer_v15W_roll{Y}_s{s}.pipeline.json")
            for crop in sorted(fy["crop"].unique()):
                m = build_model("itransformer", cfg); m.load(ck / f"itransformer_v15WF_roll{Y}_s{s}_{crop}.pt")
                out.append(e7.v2.predict(m, pipe, fy[fy["crop"] == crop]).assign(seed=s))
        print(f"  W15 {Y}", flush=True)
    P = pd.concat(out, ignore_index=True).groupby(K, as_index=False).agg(
        y_true=("y_true", "first"), W15=("y_pred", "mean"), n=("seed", "nunique"))
    assert (P["n"] == 3).all()
    q = pd.read_parquet(ROOT / "results/metrics/v12_predictions.parquet").rename(columns={"D10": "Q12"})
    D = q[K + ["y_true", "y_hist", "Q12"]].merge(P[K + ["y_true", "W15"]], on=K, suffixes=("", "_c"))
    D = D[(D.y_true - D.y_true_c).abs() < 0.05]
    ytab = pd.read_parquet(ROOT / "data/interim/v5/yield.parquet")
    h = ytab.set_index(["landkreis", "crop", "year"])["yield_dt_per_ha"]
    def b5(r):
        v = [h.get((r.landkreis, r.crop, t)) for t in range(r.year - 5, r.year)]
        v = [x for x in v if x is not None and np.isfinite(x)]
        return float(np.mean(v)) if len(v) >= 3 else r.y_hist
    D["b5"] = [b5(r) for r in D.itertuples(index=False)]
    mem = importlib.util.spec_from_file_location("v15", ROOT / "scripts/64_eval_v15.py")
    v15 = importlib.util.module_from_spec(mem); mem.loader.exec_module(v15)
    for col in ("Q12", "W15"):                        # §2 error memory, lambda chosen causally per column
        r = np.log(D[col] / D.y_true).set_axis(pd.MultiIndex.from_frame(D[["landkreis", "crop", "year"]]))
        m = []
        for lk, cr, y in D[["landkreis", "crop", "year"]].itertuples(index=False):
            past = [r.get((lk, cr, t)) for t in range(y - v15.MEMORY, y)]
            past = [v for v in past if v is not None and np.isfinite(v)]
            m.append(float(np.mean(past)) if past else 0.0)
        D[f"m_{col}"] = m
        lam = {}
        for Y in sorted(D.year.unique()):
            fit = D[(D.year >= 2011) & (D.year < Y)]
            lam[Y] = 0.0 if fit.year.nunique() < v15.MIN_FIT_YEARS else float(min(
                v15.GRID, key=lambda l: v15.mape(fit[col] * np.exp(-l * fit[f"m_{col}"]), fit.y_true)))
        D[f"{col}_mem"] = D[col] * np.exp(-D.year.map(lam) * D[f"m_{col}"])
        print(col, "lambda", lam, flush=True)
    D = D.rename(columns={"Q12_mem": "E15", "W15_mem": "WE15"})
    D.to_parquet(ROOT / f"results/metrics/v15W_predictions{TAG}.parquet", index=False)
    res = {}
    for name, new, old in (("WE15_vs_E15", "WE15", "E15"), ("W15_vs_Q12", "W15", "Q12")):
        S = D.drop(columns=[c for c in ("R7", "F") if c in D]).rename(columns={new: "R7", old: "F"})
        res[name] = {w: e7.score(S[S.year.isin(list(y))]) for w, y in WINDOWS.items()}
    h = res["WE15_vs_E15"]["hindcast_2010_2015"]
    h["decision_WE15_replaces_E15"] = bool(h["p_delta_gt_0"] >= 0.95 and h["mape_R7"] < h["mape_F"]
                                           and not h["crops_worse_every_year"])
    (ROOT / f"results/metrics/v15W_scores{TAG}.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=1))
    rows = [{"window": w, "crop": c, **{f"acc_{x}": 100 - 100 * ((g[x] - g.y_true).abs() / g.y_true).mean()
                                        for x in ("WE15", "E15", "W15", "Q12", "b5")}}
            for w, yrs in WINDOWS.items() for c, g in D[D.year.isin(list(yrs))].groupby("crop")]
    pd.DataFrame(rows).to_csv(ROOT / f"results/metrics/v15W_per_crop{TAG}.csv", index=False)


if __name__ == "__main__":
    main()

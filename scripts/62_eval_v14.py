"""Evaluate pre-registration v14 (DWD HYRAS weather) against v12 Q12, 2010-2025.

H14: itransformer_v14HF_roll{Y}_s{s}_<crop>.pt (pipelines itransformer_v14H_roll{Y}_s{s}).
Q12: results/metrics/v12_predictions.parquet (column D10 = Q12 ensemble prediction).
Same corrected-panel rows, reference = Q12's y_hist. Decision v6 §6 on 2010-2015.
Outputs: results/metrics/v14_predictions.parquet, v14_scores.json, v14_per_crop.csv
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


HINDCAST_ONLY = "--hindcast-only" in sys.argv      # decision window only (v14 §3), before 2016-2025 finish
TAG = "_hindcast" if HINDCAST_ONLY else ""
if HINDCAST_ONLY:
    WINDOWS = {"hindcast_2010_2015": range(2010, 2016)}


def main():
    cfg = load_model_config("itransformer_v14H"); ck = ROOT / "results/checkpoints"
    frame = load_modeling_frame(ROOT / cfg["dataset"], load_data_config(), window="crop_year", common_sample=True,
                                cache_dir=ROOT / "data/processed/cache", phenology=cfg["phenology"],
                                phenology_version=cfg["phenology_version"], stage_features=bool(cfg["stage_features"]),
                                crop_specific_sat=False)
    out = []
    for Y in (range(2010, 2016) if HINDCAST_ONLY else range(2010, 2026)):
        fy = frame[frame["year"] == Y]
        for s in SEEDS:
            pipe = FeaturePipeline.load(ck / f"itransformer_v14H_roll{Y}_s{s}.pipeline.json")
            for crop in sorted(fy["crop"].unique()):
                m = build_model("itransformer", cfg); m.load(ck / f"itransformer_v14HF_roll{Y}_s{s}_{crop}.pt")
                out.append(e7.v2.predict(m, pipe, fy[fy["crop"] == crop]).assign(seed=s))
        print(f"  H14 {Y}", flush=True)
    P = pd.concat(out, ignore_index=True).groupby(K, as_index=False).agg(
        y_true=("y_true", "first"), H14=("y_pred", "mean"), n=("seed", "nunique"))
    assert (P["n"] == 3).all()
    q = pd.read_parquet(ROOT / "results/metrics/v12_predictions.parquet").rename(columns={"D10": "Q12"})
    D = q[K + ["y_true", "y_hist", "Q12"]].merge(P[K + ["y_true", "H14"]], on=K, suffixes=("", "_c"))
    D = D[(D.y_true - D.y_true_c).abs() < 0.05]
    ytab = pd.read_parquet(ROOT / "data/interim/v5/yield.parquet")
    h = ytab.set_index(["landkreis", "crop", "year"])["yield_dt_per_ha"]
    def b5(r):
        v = [h.get((r.landkreis, r.crop, t)) for t in range(r.year - 5, r.year)]
        v = [x for x in v if x is not None and np.isfinite(x)]
        return float(np.mean(v)) if len(v) >= 3 else r.y_hist
    D["b5"] = [b5(r) for r in D.itertuples(index=False)]
    D.to_parquet(ROOT / f"results/metrics/v14_predictions{TAG}.parquet", index=False)
    S = D.rename(columns={"H14": "R7", "Q12": "F"})          # e7.score: R7 = challenger, F = incumbent
    res = {w: e7.score(S[S.year.isin(list(y))]) for w, y in WINDOWS.items()}
    h = res["hindcast_2010_2015"]
    h["decision_H14_replaces_Q12"] = bool(h["p_delta_gt_0"] >= 0.95 and h["mape_R7"] < h["mape_F"]
                                          and not h["crops_worse_every_year"])
    (ROOT / f"results/metrics/v14_scores{TAG}.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=1))
    rows = [{"window": w, "crop": c, **{f"acc_{x}": 100 - 100 * ((g[x] - g.y_true).abs() / g.y_true).mean()
                                        for x in ("H14", "Q12", "b5", "y_hist")}}
            for w, yrs in WINDOWS.items() for c, g in D[D.year.isin(list(yrs))].groupby("crop")]
    pd.DataFrame(rows).to_csv(ROOT / f"results/metrics/v14_per_crop{TAG}.csv", index=False)
    # v14 §3: districts with vs without their own DWD station series (temperature, 2010-2015)
    st = pd.read_parquet(ROOT / "data/interim/v4/weather.parquet", columns=["landkreis", "date", "temperature_air_max_2m"])
    st = st[pd.to_datetime(st["date"], utc=True).dt.year.between(2010, 2015)]
    own = set(st.groupby("landkreis")["temperature_air_max_2m"].apply(lambda s: s.notna().mean() >= 0.5).loc[lambda s: s].index)
    for nm, m in (("own_station", D.landkreis.isin(own)), ("no_station", ~D.landkreis.isin(own))):
        for w, yrs in WINDOWS.items():
            g = D[m & D.year.isin(list(yrs))]
            acc = {x: round(100 - 100 * ((g[x] - g.y_true).abs() / g.y_true).mean(), 2) for x in ("H14", "Q12")}
            print(f"{nm:12s} {w:20s} rows {len(g):6d} districts {g.landkreis.nunique():3d} H14 {acc['H14']} Q12 {acc['Q12']}")


if __name__ == "__main__":
    main()

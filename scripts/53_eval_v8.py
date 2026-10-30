"""Evaluate pre-registration v8 (rapeseed flowering index) with the v6 scoring (paper/preregistration_2026-10-01_v{6,7}.md).

Forecasts per district-crop-year, years 2010-2025:
    F    v4 F   itransformer_v4F_roll{Y}_s{s}_<crop>.pt   (pipeline itransformer_v4T_roll{Y}_s{s})
    R7   v7 R7  itransformer_v7RF_roll{Y}_s{s}_<crop>.pt  (pipeline itransformer_v7R_roll{Y}_s{s})
    y_hist  F's registered baseline (common reference r)
    B5      district mean of Y-5..Y-1 (>= 3 years), else y_hist
3-seed mean. Scores (v6 §4): macro MSE skill vs y_hist (crops weighted equally,
squared error pooled per crop over the window), pooled MAPE, year-block bootstrap
(B=10,000, rng 0). Decision (v6 §6) on the hindcast 2010-2015 only.

Outputs: results/metrics/v8_predictions.parquet, v7_scores.json, v7_per_crop.csv
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

_s = importlib.util.spec_from_file_location("v2", ROOT / "scripts/45_eval_v2.py")
v2 = importlib.util.module_from_spec(_s); _s.loader.exec_module(v2)
SEEDS = ("42", "43", "44")
ARMS = {"F": ("itransformer_v4T", "itransformer_v4F"), "R7": ("itransformer_v8Y", "itransformer_v8YF")}  # R7 slot = arm Y
WINDOWS = {"hindcast_2010_2015": range(2010, 2016), "dev_2016_2025": range(2016, 2026),
           "dev_2019_2025": range(2019, 2026)}


def predict_all(frame: pd.DataFrame, years) -> pd.DataFrame:
    ck = ROOT / "results/checkpoints"
    out = []
    for arm, (base, ft) in ARMS.items():
        cfg = load_model_config(base)
        for Y in years:
            fy = frame[frame["year"] == Y]
            for s in SEEDS:
                pipe = FeaturePipeline.load(ck / f"{base}_roll{Y}_s{s}.pipeline.json")
                pipe.require_current_feature_policy()
                for crop in sorted(fy["crop"].unique()):
                    m = build_model("itransformer", cfg)
                    m.load(ck / f"{ft}_roll{Y}_s{s}_{crop}.pt")
                    out.append(v2.predict(m, pipe, fy[fy["crop"] == crop]).assign(arm=arm, seed=s))
            print(f"  {arm} {Y}", flush=True)
    P = pd.concat(out, ignore_index=True)
    k = ["arm", "year", "crop", "landkreis"]
    return P.groupby(k, as_index=False).agg(y_true=("y_true", "first"), y_hist=("y_hist", "first"),
                                            y_pred=("y_pred", "mean"), n=("seed", "nunique"))


def add_b5(D: pd.DataFrame) -> pd.DataFrame:
    y = pd.read_parquet(ROOT / "data/processed/dataset_v4.parquet",
                        columns=["landkreis", "crop", "year", "yield_dt_per_ha"]).dropna()
    h = y.drop_duplicates(["landkreis", "crop", "year"]).set_index(["landkreis", "crop", "year"])["yield_dt_per_ha"]
    def b5(r):
        v = [h.get((r.landkreis, r.crop, t)) for t in range(r.year - 5, r.year)]
        v = [x for x in v if x is not None and np.isfinite(x)]
        return float(np.mean(v)) if len(v) >= 3 else r.y_hist
    D["b5"] = [b5(r) for r in D.itertuples(index=False)]
    return D


def macro_skill(W: pd.DataFrame, col: str, ref: str) -> float:
    s = [1 - ((g[col] - g.y_true) ** 2).sum() / ((g[ref] - g.y_true) ** 2).sum() for _, g in W.groupby("crop")]
    return float(np.mean(s))


def score(W: pd.DataFrame) -> dict:
    out = {"n_rows": int(len(W))}
    for col in ("R7", "F", "b5", "y_hist"):
        out[f"mape_{col}"] = float(100 * ((W[col] - W.y_true).abs() / W.y_true).mean())
        out[f"skill_{col}_vs_yhist"] = macro_skill(W, col, "y_hist")
        out[f"skill_{col}_vs_b5"] = macro_skill(W, col, "b5")
    years = np.array(sorted(W.year.unique())); rng = np.random.default_rng(0)
    by = {y: W[W.year == y] for y in years}
    d = []
    for _ in range(10000):
        S = pd.concat([by[y] for y in rng.choice(years, len(years))], ignore_index=True)
        d.append(macro_skill(S, "R7", "y_hist") - macro_skill(S, "F", "y_hist"))
    d = np.array(d)
    out["delta_skill_R7_minus_F"] = out["skill_R7_vs_yhist"] - out["skill_F_vs_yhist"]
    out["delta_ci90"] = [float(np.quantile(d, 0.05)), float(np.quantile(d, 0.95))]
    out["p_delta_gt_0"] = float((d > 0).mean())
    sse = W.assign(eR=(W.R7 - W.y_true) ** 2, eF=(W.F - W.y_true) ** 2).groupby(["crop", "year"])[["eR", "eF"]].sum()
    worse_all = (sse.eR > sse.eF).groupby(level="crop").all()
    out["crops_worse_every_year"] = [c for c, v in worse_all.items() if v]
    return out


def main() -> None:
    cfg = load_model_config("itransformer_v8Y")
    frame = load_modeling_frame(ROOT / cfg["dataset"], load_data_config(), window="crop_year",
                                common_sample=True, cache_dir=ROOT / "data/processed/cache",
                                phenology=cfg["phenology"], phenology_version=cfg["phenology_version"],
                                stage_features=bool(cfg["stage_features"]), flowering_index=True)
    P = predict_all(frame, range(2010, 2026))
    assert (P["n"] == 3).all()
    k = ["year", "crop", "landkreis"]
    F = P[P.arm == "F"][k + ["y_true", "y_hist", "y_pred"]].rename(columns={"y_pred": "F"})
    R = P[P.arm == "R7"][k + ["y_pred"]].rename(columns={"y_pred": "R7"})
    D = add_b5(F.merge(R, on=k, how="inner"))
    D.to_parquet(ROOT / "results/metrics/v8_predictions.parquet", index=False)

    res = {w: score(D[D.year.isin(list(yrs))]) for w, yrs in WINDOWS.items()}
    h = res["hindcast_2010_2015"]
    h["decision_Y_replaces_F_all_crops"] = bool(h["p_delta_gt_0"] >= 0.95 and h["mape_R7"] < h["mape_F"]
                                       and not h["crops_worse_every_year"])
    # v8 §3: rapeseed decision on the hindcast; v8 §2.2 rule N_r reported descriptively
    from src.models.national_gate import apply_gate
    rs = D[D.crop == "winter_rapeseed"].copy()
    for w, yrs in WINDOWS.items():
        g = rs[rs.year.isin(list(yrs))]
        sk = lambda col: 1 - ((g[col] - g.y_true) ** 2).sum() / ((g.y_hist - g.y_true) ** 2).sum()
        yy = np.array(sorted(g.year.unique())); rng = np.random.default_rng(0); by = {y: g[g.year == y] for y in yy}
        dd = []
        for _ in range(10000):
            S = pd.concat([by[y] for y in rng.choice(yy, len(yy))])
            h = ((S.y_hist - S.y_true) ** 2).sum()
            dd.append(((S.F - S.y_true) ** 2).sum() / h - ((S.R7 - S.y_true) ** 2).sum() / h)
        nr = apply_gate(g.assign(y_pred=g.F).rename(columns={}), {"winter_rapeseed"})
        res[w]["rapeseed"] = {"skill_Y": float(sk("R7")), "skill_F": float(sk("F")), "skill_b5": float(sk("b5")),
                              "p_Y_better": float((np.array(dd) > 0).mean()),
                              "mape_Y": float(100 * ((g.R7 - g.y_true).abs() / g.y_true).mean()),
                              "mape_F": float(100 * ((g.F - g.y_true).abs() / g.y_true).mean()),
                              "mape_b5": float(100 * ((g.b5 - g.y_true).abs() / g.y_true).mean()),
                              "mape_F_with_Nr": float(100 * ((nr - g.y_true).abs() / g.y_true).mean()),
                              "skill_F_with_Nr": float(1 - ((nr - g.y_true) ** 2).sum() / ((g.y_hist - g.y_true) ** 2).sum())}
    r = res["hindcast_2010_2015"]["rapeseed"]
    res["hindcast_2010_2015"]["decision_Y_for_rapeseed"] = bool(
        r["p_Y_better"] >= 0.95 and r["mape_Y"] < r["mape_F"]
        and res["hindcast_2010_2015"]["delta_skill_R7_minus_F"] >= -0.01)
    (ROOT / "results/metrics/v8_scores.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    rows = []
    for w, yrs in WINDOWS.items():
        W = D[D.year.isin(list(yrs))]
        for c, g in W.groupby("crop"):
            rows.append({"window": w, "crop": c,
                         **{f"mape_{x}": 100 * ((g[x] - g.y_true).abs() / g.y_true).mean() for x in ("R7", "F", "b5", "y_hist")},
                         **{f"skill_{x}": 1 - ((g[x] - g.y_true) ** 2).sum() / ((g.y_hist - g.y_true) ** 2).sum() for x in ("R7", "F", "b5")}})
    C = pd.DataFrame(rows); C.to_csv(ROOT / "results/metrics/v8_per_crop.csv", index=False)
    print(C[C.window == "hindcast_2010_2015"].round(3).to_string(index=False))
    print(C[C.window == "dev_2019_2025"].round(3).to_string(index=False))


if __name__ == "__main__":
    main()

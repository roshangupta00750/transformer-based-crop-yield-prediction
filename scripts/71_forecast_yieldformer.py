"""Issue frozen YieldFormer-DE forecasts (pre-registration v16 §4).

--stage predict  (GPU cluster, 2026): network part. Inputs data/processed/forecast_<Y>_v5.parquet
    (scripts/70); every day after --data-end is set to missing and crops harvested later are
    extended to harvest (scripts/36). Q12 per-crop fine-tuned checkpoints
    itransformer_v12QF_roll<Y>_s{42,43,44}_<crop>.pt with pipelines itransformer_v12Q_roll<Y>_s*.
    Output results/forecasts/work/q12_<Y>.parquet (one row per district-crop, seed mean + sd).
--stage normal   (GPU cluster, 2027): fold-"<Y>" normal y_hist from a Q12 feature pipeline fitted on
    years <= Y-2 (no network; no season data exist). Output results/forecasts/work/normal_<Y>.parquet.
--stage freeze   (Mac): error memory (v15 §2 / v16 §4) from the rolling backtests, 80% interval,
    SHA-256. 2026: E15 memory of Q12's errors; 2027: A16a memory of y_hist's errors if v16 §2
    found A16a better than y_hist under v6 §6, else the plain normal.

Usage:
    python scripts/71_forecast_yieldformer.py --stage predict --year 2026 --data-end 2026-09-19
    python scripts/71_forecast_yieldformer.py --stage normal --year 2027
    python scripts/71_forecast_yieldformer.py --stage freeze --year 2026 [--dry-run]
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
CK = ROOT / "results/checkpoints"
OUT = ROOT / "results/forecasts"
WORK = OUT / "work"
SEEDS = (42, 43, 44)
CFG = "itransformer_v12Q"


def _load(name: str, file: str):
    s = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return m


def frame_for(f: pd.DataFrame, cfg: dict, dc: dict, pipe) -> pd.DataFrame:
    """load_modeling_frame's steps on an in-memory build (crop window, phenology v3, stages)."""
    from src.data.dataset import (PHENOLOGY_OBS_V3, PHENOLOGY_SLOTS_V3, add_phenology_features,
                                  add_stage_features, prepare_modeling_frame)
    fr = prepare_modeling_frame(f, dc, window=pipe.window_, common_sample=pipe.common_sample_)
    fr = add_phenology_features(fr, cfg["phenology"], PHENOLOGY_OBS_V3, PHENOLOGY_SLOTS_V3)
    return add_stage_features(fr, cfg["phenology"]) if cfg.get("stage_features") else fr


ARMS = {"Q": ("itransformer_v12Q", "itransformer_v12QF", "Q12"),      # v18 ensemble members
        "T": ("tftnet_v17T", "tftnet_v17TF", "T17"),
        "P": ("patchtst17_v17P", "patchtst17_v17PF", "P17")}


def predict(y: int, end: pd.Timestamp, arm: str = "Q") -> None:
    cfg_name, out_name, col = ARMS[arm]
    from src.data.dataset import FeaturePipeline
    from src.models import build_model
    from src.utils import load_data_config, load_model_config
    x36 = _load("x36", "36_forecast.py")
    dc, cfg = load_data_config(), load_model_config(cfg_name)
    phen = dc["phenology"]
    f = pd.read_parquet(ROOT / f"data/processed/forecast_{y}_v5.parquet")
    f["date"] = pd.to_datetime(f["date"])
    print(f"data end by source coverage {x36.data_end(f, y).date()}, registered data end {end.date()}")
    f = x36.extend_to_harvest(f, y, end, phen)
    pipes = {s: FeaturePipeline.load(CK / f"{cfg_name}_roll{y}_s{s}.pipeline.json") for s in SEEDS}
    for p in pipes.values():
        p.require_current_feature_policy()
    fr = frame_for(f, cfg, dc, pipes[SEEDS[0]])
    raw = fr[fr["year"] == y]
    print(f"{y}: {raw.groupby(['landkreis', 'crop']).ngroups} district-crop sequences after the crop window")
    preds = []
    # phenology/stage flags are int8; scripts/36's masking writes the training mean into them
    num = [c for c in pipes[SEEDS[0]].spec.past_observed if c in raw.columns and pd.api.types.is_integer_dtype(raw[c])]
    raw = raw.astype({c: "float32" for c in num})
    for s, pipe in pipes.items():
        masked = x36.mask_after_date(raw, pipe, end)
        for crop in sorted(masked["crop"].unique()):
            m = build_model(cfg.get("name", "itransformer"), cfg)
            m.load(CK / f"{out_name}_roll{y}_s{s}_{crop}.pt")
            import torch
            if torch.cuda.is_available():
                m.net.to("cuda"); m.device = "cuda"
            preds.append(x36.ev.predict(m, pipe, masked[masked["crop"] == crop]).assign(seed=s))
        print(f"  seed {s} done", flush=True)
    P = pd.concat(preds)
    fc = (P.groupby(["landkreis", "crop", "year"], as_index=False)
            .agg(**{col: ("y_pred", "mean")}, seed_sd=("y_pred", "std"), y_hist=("y_hist", "first"), n=("seed", "nunique")))
    p0 = pipes[SEEDS[0]]
    fc["grown"] = [(lk, cr) in p0.target_mean_ for lk, cr in zip(fc["landkreis"], fc["crop"])]
    fc["data_end"] = end.date().isoformat()
    fc["lead_days"] = [max(0, (x36.harvest_date(y, c, phen) - end).days) for c in fc["crop"]]
    WORK.mkdir(parents=True, exist_ok=True)
    fc.to_parquet(WORK / f"{col.lower()}_{y}.parquet", index=False)
    print(fc.groupby("crop")[[col, "y_hist", "seed_sd"]].mean().round(2).to_string())


def normal(y: int) -> None:
    """Fold-Y normal: Q12's feature pipeline fitted on years <= Y-2, as scripts/03_train does."""
    from src.data.dataset import (ANOMALY_BASE, PHENOLOGY_SLOTS_V3, STAGE_COLUMNS, FeaturePipeline,
                                  add_district_anomalies, anomaly_columns, build_feature_spec,
                                  fit_district_climatology, load_modeling_frame, phenology_columns)
    from src.utils import load_data_config, load_model_config
    dc, cfg = load_data_config(), load_model_config(CFG)
    fr = load_modeling_frame(ROOT / cfg["dataset"], dc, window=cfg["window"], common_sample=True,
                             cache_dir=ROOT / "data/processed/cache", phenology=cfg["phenology"],
                             phenology_version=cfg["phenology_version"], stage_features=bool(cfg["stage_features"]))
    tr = fr[fr["year"] <= y - 2]
    spec = build_feature_spec(fr, dc, static_features=cfg.get("static_features"), drop_features=cfg.get("drop_features"))
    spec.past_observed += [c for c in phenology_columns(PHENOLOGY_SLOTS_V3) + STAGE_COLUMNS if c not in spec.past_observed]
    clim = fit_district_climatology(tr, ANOMALY_BASE)
    tr = add_district_anomalies(tr, clim)
    spec.past_observed += anomaly_columns(clim)
    pipe = FeaturePipeline(spec, anomaly_framing=True, trend_test=cfg.get("trend_test", "district_year"))
    pipe.trend_district_ref = bool(cfg.get("trend_district_ref", False))
    pipe = pipe.fit(tr)
    keys = sorted(pipe.target_mean_)
    out = pd.DataFrame({"landkreis": [k[0] for k in keys], "crop": [k[1] for k in keys], "year": y,
                        "y_hist": [pipe.baseline_yield(lk, cr, y) for lk, cr in keys]})
    WORK.mkdir(parents=True, exist_ok=True)
    out.to_parquet(WORK / f"normal_{y}.parquet", index=False)
    # check: --year 2025 must reproduce the rolling fold-2025 y_hist in results/metrics/v15_predictions.parquet
    print(out.groupby("crop")["y_hist"].mean().round(2).to_string())


def memory_for(P: pd.DataFrame, col: str, keys: pd.DataFrame, y: int) -> tuple[np.ndarray, float]:
    """v15 §2: m_y from `col`'s errors in y-3..y-1; lambda_y chosen on 2011..y-1."""
    v15 = _load("v15", "64_eval_v15.py")
    r = np.log(P[col] / P.y_true).set_axis(pd.MultiIndex.from_frame(P[["landkreis", "crop", "year"]]))
    r = r[np.isfinite(r)].to_dict()
    m = []
    for lk, cr in zip(keys["landkreis"], keys["crop"]):
        past = [r[(lk, cr, t)] for t in range(y - v15.MEMORY, y) if (lk, cr, t) in r]
        m.append(float(np.mean(past)) if past else 0.0)
    # the column's memory on the backtest rows, to choose lambda causally
    mb = []
    for lk, cr, t in P[["landkreis", "crop", "year"]].itertuples(index=False):
        past = [r[(lk, cr, s)] for s in range(t - v15.MEMORY, t) if (lk, cr, s) in r]
        mb.append(float(np.mean(past)) if past else 0.0)
    fit = P.assign(_m=mb)
    fit = fit[(fit.year >= 2011) & (fit.year < y)]
    lam = float(min(v15.GRID, key=lambda l: v15.mape(fit[col] * np.exp(-l * fit._m), fit.y_true)))
    return np.array(m), lam


def freeze(y: int, dry: bool, ensemble: bool = False) -> None:
    P = pd.read_parquet(ROOT / "results/metrics/v16_predictions.parquet")
    tag = "yieldformer"
    if ensemble:                                     # v18: mean of the three transformers + XE18 memory
        P = pd.read_parquet(ROOT / "results/metrics/v18_predictions.parquet")
        k = ["landkreis", "crop", "year"]
        parts = [pd.read_parquet(WORK / f"{c.lower()}_{y}.parquet") for c in ("Q12", "T17", "P17")]
        for d, c in zip(parts, ("Q12", "T17", "P17")):
            if not (d["n"] == len(SEEDS)).all():
                sys.exit(f"{c}: not every district-crop has all three seeds")
        fc = parts[0].merge(parts[1][k + ["T17", "y_hist"]], on=k, suffixes=("", "_t")).merge(
            parts[2][k + ["P17", "y_hist"]], on=k, suffixes=("", "_p"))
        if len(fc) != len(parts[0]) or (fc[["y_hist_t", "y_hist_p"]].sub(fc.y_hist, axis=0).abs().max().max() > 1e-6):
            sys.exit("ensemble members do not cover the same rows with the same normal")
        fc = fc[fc["grown"]].drop(columns=["grown", "n", "y_hist_t", "y_hist_p"])
        fc["X18"] = fc[["Q12", "T17", "P17"]].mean(axis=1)
        m, lam = memory_for(P, "X18", fc, y)
        fc["memory"], fc["lambda"] = m, lam
        fc["y_pred"] = fc["X18"] * np.exp(-lam * m)
        fc["seed_sd"] = fc[["Q12", "T17", "P17"]].std(axis=1)          # spread across the three networks
        fc["method"] = "YieldFormer-DE (v18): mean of iTransformer, TFT, PatchTST (3 seeds each) + district error memory"
        err = P[P.year.between(2016, 2025)].assign(e=lambda d: d.y_true - d.XE18)
        tag = "yieldformer-ensemble"
    elif y == 2026:
        fc = pd.read_parquet(WORK / f"q12_{y}.parquet")
        if not (fc["n"] == len(SEEDS)).all():
            sys.exit("not every district-crop has all three seeds")
        fc = fc[fc["grown"]].drop(columns=["grown", "n"])
        m, lam = memory_for(P, "Q12", fc, y)
        fc["memory"], fc["lambda"] = m, lam
        fc["y_pred"] = fc["Q12"] * np.exp(-lam * m)
        fc["method"] = "YieldFormer-DE (v15): Q12 iTransformer 3 seeds + district error memory"
        err = P[P.year.between(2016, 2025)].assign(e=lambda d: d.y_true - d.E15)
    else:
        fc = pd.read_parquet(WORK / f"normal_{y}.parquet")
        S = P.rename(columns={"A16a": "R7"}).assign(F=P.y_hist)
        h = _load("v16", "69_eval_v16.py").fast_score(S[S.year.between(2010, 2015)])
        use = bool(h["p_delta_gt_0"] >= 0.95 and h["mape_R7"] < h["mape_F"] and not h["crops_worse_every_year"])
        print(f"A16a vs y_hist on 2010-2015: P(Δ>0) {h['p_delta_gt_0']:.3f}, accuracy {100 - h['mape_R7']:.2f} vs "
              f"{100 - h['mape_F']:.2f}, crops worse every year {h['crops_worse_every_year']} -> memory {'ON' if use else 'OFF'}")
        m, lam = memory_for(P, "y_hist", fc, y) if use else (np.zeros(len(fc)), 0.0)
        fc["memory"], fc["lambda"] = m, lam
        fc["y_pred"] = fc["y_hist"] * np.exp(-lam * m)
        fc["seed_sd"] = 0.0
        fc["method"] = ("YieldFormer-DE long-lead rule: district normal + error memory (no season data)" if use
                        else "YieldFormer-DE long-lead rule: district normal (no season data)")
        fc["data_end"] = "none (season not started)"
        x36 = _load("x36", "36_forecast.py")
        from src.utils import load_data_config
        phen = load_data_config()["phenology"]
        fc["lead_days"] = [(x36.harvest_date(y, c, phen) - pd.Timestamp(date.today())).days for c in fc["crop"]]
        col = "A16a" if use else "y_hist"
        err = P[P.year.between(2016, 2025)].assign(e=lambda d: d.y_true - d[col])
    q = err.groupby("crop")["e"].quantile([0.1, 0.9]).unstack()
    fc = fc.join(q.set_axis(["q10", "q90"], axis=1), on="crop")
    fc["lo80"], fc["hi80"] = fc["y_pred"] + fc.pop("q10"), fc["y_pred"] + fc.pop("q90")
    fc["issued"] = date.today().isoformat()
    last = int(P.year.max())
    fc["model_trained_through"] = (f"train<={y - 2}, val={y - 1}" if y - 1 <= last
                                   else f"district normal from reported yields up to {last}")
    fc["year"] = y
    cols = ["landkreis", "crop", "year", "y_pred", "lo80", "hi80", "y_hist", "seed_sd", "memory", "lambda",
            "method", "data_end", "lead_days", "issued", "model_trained_through"]
    fc = fc[cols].sort_values(["crop", "landkreis"]).reset_index(drop=True)
    s = fc.groupby("crop").agg(districts=("landkreis", "size"), forecast=("y_pred", "mean"), normal=("y_hist", "mean"),
                               lead_days=("lead_days", "first"))
    s["vs_normal_pct"] = 100 * (s.forecast / s.normal - 1)
    print(f"lambda_{y} = {lam}\n{s.round(2).to_string()}")
    if dry:
        return
    path = OUT / f"forecast_{y}_{tag}_issued_{date.today().isoformat()}.csv"
    if path.exists():
        sys.exit(f"{path.name} already exists; frozen forecasts are never overwritten")
    fc.to_csv(path, index=False, float_format="%.3f")
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    with open(OUT / "FROZEN.sha256", "a") as fh:
        fh.write(f"{sha}  {path.name}\n")
    print(f"frozen {path.relative_to(ROOT)}\nsha256 {sha}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=["predict", "normal", "freeze"], required=True)
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--data-end", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--arm", choices=sorted(ARMS), default="Q", help="predict: which transformer")
    ap.add_argument("--ensemble", action="store_true", help="freeze: v18 ensemble of the three transformers")
    a = ap.parse_args()
    if a.stage == "predict":
        predict(a.year, pd.Timestamp(a.data_end), a.arm)
    elif a.stage == "normal":
        normal(a.year)
    else:
        freeze(a.year, a.dry_run, a.ensemble)


if __name__ == "__main__":
    main()

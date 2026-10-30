"""Pre-registration v17 (paper/preregistration_2026-10-10_v17.md): TFT (T17) and PatchTST (P17).

--stage predict  (GPU cluster): rolling predictions of one arm, per-crop fine-tuned checkpoints
    <out>_roll{Y}_s{s}_<crop>.pt with pipelines <cfg>_roll{Y}_s{s}, 3-seed mean
    -> results/metrics/v17_predictions_<arm>.parquet; for T17 also the mean TFT variable-selection
    weight per crop and input -> results/metrics/v17_tft_variable_weights.csv
--stage score    (Mac): §3 decision. Memory (v15 §2) on each arm's own errors (TE17, PE17); each vs
    E15 under v6 §6 on 2010-2015; 2016-2025, per crop, vs Q12 and vs the one non-transformer
    reference GE16 (XGBoost + memory; reporting only, never a candidate).
    -> results/metrics/v17_predictions.parquet, v17_scores.json, v17_comparison.csv, v17_per_crop.csv

Usage:
    python scripts/73_eval_v17.py --stage predict --arm T [--years 2015] [--smoke]
    python scripts/73_eval_v17.py --stage score
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
M = ROOT / "results/metrics"
CK = ROOT / "results/checkpoints"
SEEDS = (42, 43, 44)
ARMS = {"T": ("tftnet_v17T", "tftnet_v17TF", "T17", "TE17"),
        "P": ("patchtst17_v17P", "patchtst17_v17PF", "P17", "PE17")}
# prediction only (scored by scripts/79_eval_v20.py): v12 iTransformer for the in-season cuts, v20 stress arms
PREDICT_ONLY = {"Q": ("itransformer_v12Q", "itransformer_v12QF", "Q12", None),
                "Q20": ("itransformer_v20Q", "itransformer_v20QF", "Q20", None),
                "T20": ("tftnet_v20T", "tftnet_v20TF", "T20", None),
                "P20": ("patchtst17_v20P", "patchtst17_v20PF", "P20", None)}
KEY = ["year", "crop", "landkreis"]
WINDOWS = {"hindcast_2010_2015": range(2010, 2016), "dev_2016_2025": range(2016, 2026),
           "all_2010_2025": range(2010, 2026)}


def _load(name: str, file: str):
    s = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return m


def predict(arm: str, years: list[int], smoke: bool, cut: str | None = None) -> None:
    """cut 'MM-DD': every input after that date of the harvest year is set to missing as in scripts/36
    (preregistration v20 §3, in-season forecasts)."""
    import torch
    from torch.utils.data import DataLoader
    from src.data.dataset import CropYieldDataset, FeaturePipeline, collate_pad, load_modeling_frame
    from src.models import build_model
    from src.utils import load_data_config, load_model_config
    x34 = _load("x34", "34_eval_prereg.py")
    cfg_name, out_name, col, _ = {**ARMS, **PREDICT_ONLY}[arm]
    x36 = _load("x36", "36_forecast.py") if cut else None
    cfg = load_model_config(cfg_name)
    frame = load_modeling_frame(ROOT / cfg["dataset"], load_data_config(), window="crop_year", common_sample=True,
                                cache_dir=ROOT / "data/processed/cache", phenology=cfg["phenology"],
                                phenology_version=cfg["phenology_version"], stage_features=bool(cfg["stage_features"]),
                                stress_features=bool(cfg.get("stress_features", False)))
    tag = "_smoke" if smoke else ""
    preds, wts = [], []
    for Y in years:
        fy = frame[frame["year"] == Y]
        for s in SEEDS[:1] if smoke else SEEDS:
            pipe = FeaturePipeline.load(CK / f"{cfg_name}_roll{Y}_s{s}{tag}.pipeline.json")
            pipe.require_current_feature_policy()
            for crop in sorted(fy["crop"].unique()):
                f = CK / f"{out_name}_roll{Y}_s{s}{tag}_{crop}.pt"
                if smoke and not f.exists():
                    continue
                m = build_model(cfg["name"], cfg).load(f)
                if torch.cuda.is_available():                    # prediction on the GPU (CPU: ~6 min per year)
                    m.net.to("cuda"); m.device = "cuda"
                raw = fy[fy["crop"] == crop]
                if cut:
                    ints = [c for c in pipe.spec.past_observed if c in raw.columns and pd.api.types.is_integer_dtype(raw[c])]
                    raw = x36.mask_after_date(raw.astype({c: "float32" for c in ints}), pipe, pd.Timestamp(f"{Y}-{cut}"))
                preds.append(x34.predict(m, pipe, raw).assign(seed=s))
                if arm == "T" and not cut:                       # TFT variable-selection weights
                    ds = CropYieldDataset(pipe.transform_features(raw), pipe.spec, pipe)
                    dl = DataLoader(ds, batch_size=256, collate_fn=partial(collate_pad, pad_to=pipe.sequence_length_))
                    m.net.eval(); acc, n = 0, 0
                    with torch.no_grad():
                        for b in dl:
                            x = torch.cat([b["x_past"], b["x_future"]], dim=-1).to(m.device)
                            msk = b.get("mask")
                            m.net(x, None if msk is None else msk.to(m.device), b["x_static"].to(m.device))
                            acc = acc + m.net.last_weights["variables"].sum(0).cpu(); n += len(x)
                    names = list(pipe.spec.past_observed) + list(pipe.spec.known_future)
                    wts.append(pd.DataFrame({"year": Y, "seed": s, "crop": crop, "input": names,
                                             "weight": (acc / n).numpy()}))
        print(f"  {col} {Y}", flush=True)
    P = pd.concat(preds, ignore_index=True)
    out = P.groupby(KEY, as_index=False).agg(y_true=("y_true", "first"), y_hist=("y_hist", "first"),
                                              **{col: ("y_pred", "mean")}, n=("seed", "nunique"))
    stem = (f"horizon_{cut}_{col}" if cut else f"v17_predictions_{arm}" if arm in ARMS else f"v20_predictions_{col}")
    out.to_parquet(M / f"{stem}{tag}.parquet", index=False)
    print(f"{col}: {len(out):,} rows, accuracy {100 - 100 * np.mean(np.abs(out[col] - out.y_true) / out.y_true):.2f}% "
          f"(normal {100 - 100 * np.mean(np.abs(out.y_hist - out.y_true) / out.y_true):.2f}%)")
    if wts:
        W = pd.concat(wts).groupby(["crop", "input"], as_index=False)["weight"].mean()
        W.to_csv(M / f"v17_tft_variable_weights{tag}.csv", index=False)
        print(W.sort_values("weight", ascending=False).groupby("crop").head(3).to_string(index=False))


def score() -> None:
    v16 = _load("v16", "69_eval_v16.py")
    P = pd.read_parquet(M / "v16_predictions.parquet")
    res, rows, lam = {}, [], {}
    arms = []
    for a, (_, _, col, mem) in ARMS.items():
        f = M / f"v17_predictions_{a}.parquet"
        if not f.exists():
            print(f"{f.name} missing: arm {col} not scored")
            continue
        A = pd.read_parquet(f)
        if not (A["n"] == len(SEEDS)).all():
            sys.exit(f"{col}: not every row has all 3 seeds")
        P = P.merge(A[KEY + ["y_hist", col]].rename(columns={"y_hist": f"y_hist_{col}"}), on=KEY, how="left")
        miss = int(P[col].isna().sum())
        gap = float((P[f"y_hist_{col}"] - P.y_hist).abs().max())
        print(f"{col}: {miss} of {len(P):,} rows missing, max |y_hist diff| {gap:.2e}")
        if miss:
            sys.exit(f"{col} does not cover every E15 row; v17 compares on the same rows")
        P = P.drop(columns=f"y_hist_{col}")
        P, lam[mem] = v16.memory(P, col, mem)
        arms.append((col, mem))
    for col, mem in arms:
        for chal, inc in ((mem, "E15"), (col, "Q12"), (mem, "GE16")):   # GE16: reference only
            S = P.rename(columns={chal: "R7"}).assign(F=P[inc])
            r = {w: v16.fast_score(S[S.year.isin(list(y))]) for w, y in WINDOWS.items()}
            h = r["hindcast_2010_2015"]
            r["passes_v6"] = bool(h["p_delta_gt_0"] >= 0.95 and h["mape_R7"] < h["mape_F"]
                                  and not h["crops_worse_every_year"])
            res[f"{chal}_vs_{inc}"] = r
            for w in WINDOWS:
                s = r[w]
                rows.append({"window": w, "challenger": chal, "incumbent": inc, "rows": s["n_rows"],
                             "acc_challenger": 100 - s["mape_R7"], "acc_incumbent": 100 - s["mape_F"],
                             "delta_skill": s["delta_skill_R7_minus_F"], "ci90_lo": s["delta_ci90"][0],
                             "ci90_hi": s["delta_ci90"][1], "p_delta_gt_0": s["p_delta_gt_0"],
                             "crops_worse_every_year": ", ".join(s["crops_worse_every_year"]),
                             "passes_v6": r["passes_v6"] if w == "hindcast_2010_2015" else None})
    passing = [m for _, m in arms if res[f"{m}_vs_E15"]["passes_v6"]]
    res["new_main_model"] = (max(passing, key=lambda m: res[f"{m}_vs_E15"]["hindcast_2010_2015"]["skill_R7_vs_yhist"])
                             if passing else None)
    res["lambda"] = lam
    P.to_parquet(M / "v17_predictions.parquet", index=False)
    (M / "v17_scores.json").write_text(json.dumps(res, indent=2, default=float))
    pd.DataFrame(rows).to_csv(M / "v17_comparison.csv", index=False)
    cols = ["E15", "Q12", "GE16"] + [c for a in arms for c in a]
    pd.DataFrame([{"window": w, "crop": c, **{f"acc_{x}": 100 - 100 * np.mean(np.abs(g[x] - g.y_true) / g.y_true)
                                               for x in cols}}
                  for w, yrs in WINDOWS.items() for c, g in P[P.year.isin(list(yrs))].groupby("crop")]
                 ).to_csv(M / "v17_per_crop.csv", index=False)
    pd.set_option("display.width", 220)
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print("new main model:", res["new_main_model"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=["predict", "score"], required=True)
    ap.add_argument("--arm", choices=sorted({**ARMS, **PREDICT_ONLY}))
    ap.add_argument("--cut", default=None, help="MM-DD: in-season cut date (v20 §3)")
    ap.add_argument("--years", default="2010-2025")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.stage == "predict":
        lo, _, hi = a.years.partition("-")
        predict(a.arm, list(range(int(lo), int(hi or lo) + 1)), a.smoke, a.cut)
    else:
        score()


if __name__ == "__main__":
    main()

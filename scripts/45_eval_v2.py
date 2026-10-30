"""Evaluate pre-registration v2 (paper/preregistration_2026-09-28_v2.md).

Checkpoints itransformer_v2{D0,P,N,PN,PNrt}_roll{Y}_s{42,43,44}, each scored on
its own harvest year Y. Arms with a national component: prediction =
baseline + (ridge national z + transformer within-year z) x crop std.

Outputs (results/metrics/):
    v2_predictions.parquet   per district, per arm/seed (+ "ens" = 3-seed mean)
    v2_eval.csv              per arm/seed/year/crop: R², RMSE, MAE, skill, direction
    v2_national.csv          per arm/crop: national swing ratio, sign right, corr
    v2_hypotheses.json       H4-H6 decisions with every count (§4)
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.data.dataset import (CropYieldDataset, FeaturePipeline, collate_pad,  # noqa: E402
                              load_modeling_frame)
from src.models import build_model  # noqa: E402
from src.models.national import NationalModel, season_features  # noqa: E402
from src.utils import load_data_config, load_model_config  # noqa: E402

_s = importlib.util.spec_from_file_location("ev", ROOT / "scripts/34_eval_prereg.py")
ev = importlib.util.module_from_spec(_s); _s.loader.exec_module(ev)

PAT = re.compile(r"^itransformer_v2(?P<arm>D0|PNrt|PN|P|N)_roll(?P<year>\d{4})_s(?P<seed>\d+)$")
HYP = {"H4": ("P", "D0"), "H5": ("N", "D0"), "H6": ("PN", "P")}
THRESHOLD, N_EXPECTED, VETO = 59, 100, 7


def predict(model, pipe: FeaturePipeline, raw: pd.DataFrame) -> pd.DataFrame:
    t = pipe.transform_features(raw)
    ds = CropYieldDataset(t, pipe.spec, pipe)
    loader = DataLoader(ds, batch_size=256, collate_fn=partial(collate_pad, pad_to=pipe.sequence_length_))
    zp = np.asarray(model.predict(loader)).ravel()
    lk = np.array([k[0] for k in ds.keys]); cr = np.array([k[1] for k in ds.keys])
    yr = np.array([k[2] for k in ds.keys])
    z_nat = np.zeros_like(zp)
    if pipe.national_:
        nat = NationalModel.from_dict(pipe.national_).predict(
            season_features(t, pipe.spec.past_observed, bool(pipe.phenology_)))
        z_nat = np.array([nat.get((c, int(y)), 0.0) for c, y in zip(cr, yr)])
    zt = np.asarray(ds._targets_z).ravel()
    return pd.DataFrame({"landkreis": lk, "crop": cr, "year": yr,
                         "y_true": pipe.inverse_transform_observed_array(lk, cr, zt, yr),
                         "y_pred": pipe.inverse_transform_array(lk, cr, zp + z_nat, yr),
                         "y_hist": pipe.inverse_transform_array(lk, cr, np.zeros_like(zp), yr),
                         "z_national": z_nat})


def main() -> None:
    dc = load_data_config()
    ck = ROOT / "results/checkpoints"
    runs = [m.groupdict() | {"name": m.group(0)} for f in sorted(ck.glob("itransformer_v2*_roll*.pt"))
            if (m := PAT.match(f.stem))]
    print(f"{len(runs)} checkpoints")
    frames, preds = {}, []
    for r in runs:
        pipe = FeaturePipeline.load(ck / f"{r['name']}.pipeline.json")
        pipe.require_current_feature_policy()
        cfg = load_model_config(f"itransformer_v2{r['arm']}")
        key = (cfg["dataset"], pipe.window_, pipe.common_sample_, pipe.phenology_)
        if key not in frames:
            frames[key] = load_modeling_frame(ROOT / key[0], dc, window=key[1], common_sample=key[2],
                                              cache_dir=ROOT / "data/processed/cache", phenology=key[3])
        f = frames[key]
        model = build_model("itransformer", cfg)
        model.load(ck / f"{r['name']}.pt")
        preds.append(predict(model, pipe, f[f["year"] == int(r["year"])]).assign(arm=r["arm"], seed=r["seed"]))
        print(f"  {r['name']}", flush=True)
    P = pd.concat(preds, ignore_index=True)
    k = ["arm", "year", "crop", "landkreis"]
    ens = (P.groupby(k, as_index=False).agg(y_true=("y_true", "first"), y_hist=("y_hist", "first"),
                                            y_pred=("y_pred", "mean"), z_national=("z_national", "mean"),
                                            n=("seed", "nunique")))
    assert (ens["n"] == 3).all(), "every crop-year needs all 3 seeds"
    P = pd.concat([P, ens.drop(columns="n").assign(seed="ens")], ignore_index=True)
    P.to_parquet(ROOT / "results/metrics/v2_predictions.parquet", index=False)

    E = pd.DataFrame([{"arm": a, "seed": s, "year": y, "crop": c, **ev.metrics(g.y_true, g.y_pred, g.y_hist)}
                      for (a, s, y, c), g in P.groupby(["arm", "seed", "year", "crop"])])
    E.to_csv(ROOT / "results/metrics/v2_eval.csv", index=False)

    Q = P[P.seed == "ens"].assign(a_t=lambda d: d.y_true - d.y_hist, a_p=lambda d: d.y_pred - d.y_hist)
    N = Q.groupby(["arm", "crop", "year"])[["a_t", "a_p"]].mean().reset_index()
    nat = N.groupby(["arm", "crop"]).apply(lambda g: pd.Series({
        "swing_ratio": g.a_p.std() / g.a_t.std(), "sign_right": int((np.sign(g.a_t) == np.sign(g.a_p)).sum()),
        "corr": np.corrcoef(g.a_t, g.a_p)[0, 1]})).reset_index()
    nat.to_csv(ROOT / "results/metrics/v2_national.csv", index=False)

    S = E.pivot_table(index=["year", "crop"], columns=["arm", "seed"], values="skill")
    out = {}
    for h, (new, ref) in HYP.items():
        if new not in S.columns.get_level_values(0) or ref not in S.columns.get_level_values(0):
            out[h] = {"comparison": f"{new} vs {ref}", "status": "arm missing"}
            continue
        d = (S[(new, "ens")] - S[(ref, "ens")]).dropna()
        per_seed = {s: int(((S[(new, s)] - S[(ref, s)]).dropna() > 0).sum()) for s in ("42", "43", "44")}
        worse = (d < 0).groupby(level="crop").sum().astype(int).to_dict()
        better = int((d > 0).sum())
        out[h] = {"comparison": f"{new} vs {ref}", "crop_years": int(len(d)), "expected": N_EXPECTED,
                  "better": better, "threshold": THRESHOLD, "per_seed_better": per_seed,
                  "worse_years_by_crop": worse,
                  "criterion_count": better >= THRESHOLD,
                  "criterion_all_seeds": all(v > len(d) / 2 for v in per_seed.values()),
                  "criterion_no_crop_veto": max(worse.values()) < VETO,
                  "confirmed": bool(better >= THRESHOLD and all(v > len(d) / 2 for v in per_seed.values())
                                    and max(worse.values()) < VETO and len(d) == N_EXPECTED)}
    (ROOT / "results/metrics/v2_hypotheses.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    beat = E[E.seed == "ens"].groupby("arm").apply(lambda g: pd.Series({
        "beats_hist": int((g.skill > 0).sum()), "median_skill": g.skill.median()}))
    print(beat.round(3).to_string())
    print(nat.groupby("arm")[["swing_ratio", "sign_right", "corr"]].mean().round(2).to_string())


if __name__ == "__main__":
    main()

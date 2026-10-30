"""Evaluate pre-registration v3 (paper/preregistration_2026-09-29_v3.md).

Checkpoints itransformer_v3{H,S,A}_roll{Y}_s{42,43,44} plus the reused
itransformer_v2D0_roll{Y}_s* (arm D0), each scored on its own harvest year Y.

Outputs (results/metrics/):
    v3_predictions.parquet   per district, per arm/seed (+ "ens" = 3-seed mean)
    v3_eval.csv              per arm/seed/year/crop metrics incl. skill
    v3_diagnostics.csv       per arm: within-year district corr, national swing / sign / corr
    v3_hypotheses.json       H7-H9 decisions with every count (§4)
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.data.dataset import FeaturePipeline, load_modeling_frame  # noqa: E402
from src.models import build_model  # noqa: E402
from src.utils import load_data_config, load_model_config  # noqa: E402


def _load(name: str, file: str):
    s = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return m


ev = _load("ev", "34_eval_prereg.py")
v2 = _load("v2", "45_eval_v2.py")

PAT = re.compile(r"^itransformer_(?P<cfg>v3(?P<a3>H|S|A)|v2(?P<a2>D0))_roll(?P<year>\d{4})_s(?P<seed>\d+)$")
HYP = {"H7": ("H", "D0"), "H8": ("S", "H"), "H9": ("A", "S")}
THRESHOLD, N_EXPECTED, VETO = 59, 100, 7


def main() -> None:
    dc = load_data_config()
    ck = ROOT / "results/checkpoints"
    runs = []
    for f in sorted(ck.glob("itransformer_v*_roll*.pt")):
        m = PAT.match(f.stem)
        if m:
            runs.append({"name": m.group(0), "cfg": f"itransformer_{m['cfg']}",
                         "arm": m["a3"] or m["a2"], "year": int(m["year"]), "seed": m["seed"]})
    print(f"{len(runs)} checkpoints")
    assert len(runs) == 120, "expected 4 arms x 10 years x 3 seeds"
    frames, preds = {}, []
    for r in runs:
        pipe = FeaturePipeline.load(ck / f"{r['name']}.pipeline.json")
        pipe.require_current_feature_policy()
        cfg = load_model_config(r["cfg"])
        key = (cfg["dataset"], pipe.window_, pipe.common_sample_, cfg.get("phenology"),
               cfg.get("phenology_version", "v2"), bool(cfg.get("stage_features", False)))
        if key not in frames:
            frames[key] = load_modeling_frame(ROOT / key[0], dc, window=key[1], common_sample=key[2],
                                              cache_dir=ROOT / "data/processed/cache", phenology=key[3],
                                              phenology_version=key[4], stage_features=key[5])
        f = frames[key]
        model = build_model("itransformer", cfg)
        model.load(ck / f"{r['name']}.pt")
        preds.append(v2.predict(model, pipe, f[f["year"] == r["year"]]).assign(arm=r["arm"], seed=r["seed"]))
        print(f"  {r['name']}", flush=True)
    P = pd.concat(preds, ignore_index=True)
    k = ["arm", "year", "crop", "landkreis"]
    ens = P.groupby(k, as_index=False).agg(y_true=("y_true", "first"), y_hist=("y_hist", "first"),
                                           y_pred=("y_pred", "mean"), n=("seed", "nunique"))
    assert (ens["n"] == 3).all(), "every crop-year needs all 3 seeds"
    P = pd.concat([P, ens.drop(columns="n").assign(seed="ens")], ignore_index=True)
    P.to_parquet(ROOT / "results/metrics/v3_predictions.parquet", index=False)

    E = pd.DataFrame([{"arm": a, "seed": s, "year": y, "crop": c, **ev.metrics(g.y_true, g.y_pred, g.y_hist)}
                      for (a, s, y, c), g in P.groupby(["arm", "seed", "year", "crop"])])
    E.to_csv(ROOT / "results/metrics/v3_eval.csv", index=False)

    Q = P[P.seed == "ens"].assign(a_t=lambda d: d.y_true - d.y_hist, a_p=lambda d: d.y_pred - d.y_hist)
    diag = []
    for arm, g in Q.groupby("arm"):
        gy = g.groupby(["crop", "year"])
        nt, npd = gy.a_t.transform("mean"), gy.a_p.transform("mean")
        N = gy[["a_t", "a_p"]].mean().reset_index()
        per_crop = N.groupby("crop").apply(lambda h: pd.Series({
            "swing": h.a_p.std() / h.a_t.std(), "sign": (np.sign(h.a_t) == np.sign(h.a_p)).sum(),
            "corr": np.corrcoef(h.a_t, h.a_p)[0, 1]}))
        diag.append({"arm": arm, "within_year_corr": np.corrcoef(g.a_t - nt, g.a_p - npd)[0, 1],
                     "national_swing_ratio": per_crop.swing.mean(),
                     "national_sign_right_of_10": per_crop.sign.mean(),
                     "national_corr": per_crop["corr"].mean()})
    D = pd.DataFrame(diag)
    D.to_csv(ROOT / "results/metrics/v3_diagnostics.csv", index=False)

    S = E.pivot_table(index=["year", "crop"], columns=["arm", "seed"], values="skill")
    out = {}
    for h, (new, ref) in HYP.items():
        d = (S[(new, "ens")] - S[(ref, "ens")]).dropna()
        per_seed = {s: int(((S[(new, s)] - S[(ref, s)]).dropna() > 0).sum()) for s in ("42", "43", "44")}
        worse = (d < 0).groupby(level="crop").sum().astype(int).to_dict()
        better = int((d > 0).sum())
        crit = {"criterion_count": better >= THRESHOLD,
                "criterion_all_seeds": all(v > len(d) / 2 for v in per_seed.values()),
                "criterion_no_crop_veto": max(worse.values()) < VETO}
        out[h] = {"comparison": f"{new} vs {ref}", "crop_years": int(len(d)), "expected": N_EXPECTED,
                  "better": better, "threshold": THRESHOLD, "per_seed_better": per_seed,
                  "worse_years_by_crop": worse, **crit,
                  "confirmed": bool(all(crit.values()) and len(d) == N_EXPECTED)}
    (ROOT / "results/metrics/v3_hypotheses.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({h: {k: v[k] for k in ("comparison", "better", "per_seed_better", "confirmed")}
                      for h, v in out.items()}, indent=1))
    beat = E[E.seed == "ens"].groupby("arm").apply(lambda g: pd.Series({
        "beats_hist": int((g.skill > 0).sum()), "median_skill": g.skill.median()}))
    print(beat.round(3).to_string())
    print(D.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

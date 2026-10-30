"""Evaluate pre-registration v4 (paper/preregistration_2026-09-29_v4.md).

Arms: A = itransformer_v3A_roll{Y}_s{s} (reused), T = itransformer_v4T_roll{Y}_s{s},
F = itransformer_v4F_roll{Y}_s{s}_<crop> (per crop, T's pipeline). Each scored on Y.

Outputs (results/metrics/): v4_predictions.parquet, v4_eval.csv, v4_hypotheses.json,
v4_rmse_T_vs_A.csv, v4_labels.csv, v4_finetune_choices.csv, v4_trends.csv
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


def _load(name, file):
    s = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return m


ev = _load("ev", "34_eval_prereg.py")
v2 = _load("v2", "45_eval_v2.py")
YEARS, SEEDS = range(2016, 2026), ("42", "43", "44")
HYP = {"H10": ("T", "A"), "H11": ("F", "T")}
THRESHOLD, N_EXPECTED, VETO = 59, 100, 7
AUTUMN = {"winter_wheat", "winter_barley", "rye", "triticale", "winter_rapeseed"}


def main() -> None:
    ck = ROOT / "results/checkpoints"
    dc = load_data_config()
    cfg = load_model_config("itransformer_v4T")
    frame = load_modeling_frame(ROOT / cfg["dataset"], dc, window="crop_year", common_sample=True,
                                cache_dir=ROOT / "data/processed/cache", phenology=cfg["phenology"],
                                phenology_version=cfg["phenology_version"],
                                stage_features=bool(cfg["stage_features"]))
    preds, trends, choices = [], [], []
    for Y in YEARS:
        fy = frame[frame["year"] == Y]
        for s in SEEDS:
            for arm, base in (("A", "itransformer_v3A"), ("T", "itransformer_v4T")):
                name = f"{base}_roll{Y}_s{s}"
                pipe = FeaturePipeline.load(ck / f"{name}.pipeline.json")
                pipe.require_current_feature_policy()
                m = build_model("itransformer", load_model_config(base)); m.load(ck / f"{name}.pt")
                preds.append(v2.predict(m, pipe, fy).assign(arm=arm, seed=s))
                trends.append({"arm": arm, "year": Y, "seed": s, **pipe.target_trend_by_crop_})
            pipe = FeaturePipeline.load(ck / f"itransformer_v4T_roll{Y}_s{s}.pipeline.json")
            log = json.loads((ck / f"itransformer_v4F_roll{Y}_s{s}.finetune.json").read_text())
            for crop in sorted(fy["crop"].unique()):
                m = build_model("itransformer", cfg); m.load(ck / f"itransformer_v4F_roll{Y}_s{s}_{crop}.pt")
                preds.append(v2.predict(m, pipe, fy[fy["crop"] == crop]).assign(arm="F", seed=s))
                choices.append({"year": Y, "seed": s, "crop": crop, "chosen": log["crops"][crop]["chosen"]})
            print(f"  {Y} s{s}", flush=True)
    P = pd.concat(preds, ignore_index=True)
    k = ["arm", "year", "crop", "landkreis"]
    ens = P.groupby(k, as_index=False).agg(y_true=("y_true", "first"), y_hist=("y_hist", "first"),
                                           y_pred=("y_pred", "mean"), n=("seed", "nunique"))
    assert (ens["n"] == 3).all()
    P = pd.concat([P, ens.drop(columns="n").assign(seed="ens")], ignore_index=True)
    P.to_parquet(ROOT / "results/metrics/v4_predictions.parquet", index=False)
    pd.DataFrame(trends).to_csv(ROOT / "results/metrics/v4_trends.csv", index=False)
    C = pd.DataFrame(choices); C.to_csv(ROOT / "results/metrics/v4_finetune_choices.csv", index=False)

    E = pd.DataFrame([{"arm": a, "seed": s, "year": y, "crop": c, **ev.metrics(g.y_true, g.y_pred, g.y_hist)}
                      for (a, s, y, c), g in P.groupby(["arm", "seed", "year", "crop"])])
    E.to_csv(ROOT / "results/metrics/v4_eval.csv", index=False)
    S = E.pivot_table(index=["year", "crop"], columns=["arm", "seed"], values="skill")
    out = {}
    for h, (new, ref) in HYP.items():
        d = (S[(new, "ens")] - S[(ref, "ens")]).dropna()
        per_seed = {s: int(((S[(new, s)] - S[(ref, s)]).dropna() > 0).sum()) for s in SEEDS}
        worse = (d < 0).groupby(level="crop").sum().astype(int).to_dict()
        better = int((d > 0).sum())
        crit = {"criterion_count": better >= THRESHOLD,
                "criterion_all_seeds": all(v > len(d) / 2 for v in per_seed.values()),
                "criterion_no_crop_veto": max(worse.values()) < VETO}
        out[h] = {"comparison": f"{new} vs {ref}", "crop_years": int(len(d)), "better": better,
                  "threshold": THRESHOLD, "per_seed_better": per_seed, "worse_years_by_crop": worse,
                  **crit, "confirmed": bool(all(crit.values()) and len(d) == N_EXPECTED)}
    (ROOT / "results/metrics/v4_hypotheses.json").write_text(json.dumps(out, indent=2))

    R = E[E.seed == "ens"].pivot_table(index=["year", "crop"], columns="arm", values="rmse")
    R.to_csv(ROOT / "results/metrics/v4_rmse_T_vs_A.csv")
    Q = P[P.seed == "ens"].assign(a_t=lambda d: (d.y_true - d.y_hist) / d.y_hist,
                                  a_p=lambda d: (d.y_pred - d.y_hist) / d.y_hist)
    N = Q.groupby(["arm", "crop", "year"])[["a_t", "a_p"]].mean().reset_index()
    lab = lambda x: np.where(x.rank(method="first") <= 3, "bad", np.where(x.rank(method="first") >= 8, "good", "average"))
    N["true"] = N.groupby(["arm", "crop"]).a_t.transform(lab)
    N["pred"] = N.groupby(["arm", "crop"]).a_p.transform(lab)
    N.to_csv(ROOT / "results/metrics/v4_labels.csv", index=False)

    print(json.dumps({h: {k: v[k] for k in ("comparison", "better", "per_seed_better", "worse_years_by_crop", "confirmed")}
                      for h, v in out.items()}, indent=1))
    e = E[E.seed == "ens"].assign(group=lambda d: np.where(d.crop.isin(AUTUMN), "autumn-sown", "spring-sown"))
    print(e.groupby("arm").apply(lambda g: pd.Series({"beats_hist": int((g.skill > 0).sum()),
                                                        "median_skill": g.skill.median()})).round(3).to_string())
    print(e.groupby(["group", "arm"]).skill.median().unstack().round(3).to_string())
    print("T vs A: lower RMSE in", int((R["T"] < R["A"]).sum()), "of", len(R), "crop-years")
    print("labels right:", N.assign(ok=N.true == N.pred).groupby("arm").ok.sum().to_dict())
    print("fine-tune choice counts:", C.chosen.value_counts().sort_index().to_dict())
    print(E[E.seed == "ens"].pivot_table(index="crop", columns="arm", values="skill", aggfunc="median").round(3).to_string())


if __name__ == "__main__":
    main()

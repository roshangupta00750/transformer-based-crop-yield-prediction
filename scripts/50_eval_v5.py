"""Evaluate pre-registration v5 (paper/preregistration_2026-09-30_v5.md).

Arms: F = v4 F (predictions reused from v4_predictions.parquet), G = F + national
gate, P = autumn-/spring-sown models + per-crop fine-tuning
(itransformer_v5P{A,S}_roll{Y}_s{s}_<crop>.pt with the v5{A,S} pipelines),
PG = P + gate. Gate decisions: data/processed/static/gate_decisions_v5.csv
(training years only, scripts/49).

Outputs (results/metrics/): v5_predictions.parquet, v5_eval.csv, v5_hypotheses.json,
v5_validation_2019_2025.csv
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
from src.models.national_gate import apply_gate  # noqa: E402
from src.utils import load_data_config, load_model_config  # noqa: E402


def _load(name, file):
    s = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return m


ev = _load("ev", "34_eval_prereg.py")
v2 = _load("v2", "45_eval_v2.py")
YEARS, SEEDS = range(2016, 2026), ("42", "43", "44")
AUTUMN = {"winter_wheat", "winter_barley", "rye", "triticale", "winter_rapeseed"}
HYP = {"H12": ("G", "F"), "H13": ("P", "F")}
THRESHOLD, N_EXPECTED, VETO = 59, 100, 7


def gated(P: pd.DataFrame, gates: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (y, s), g in P.groupby(["year", "seed"]):
        on = set(gates[(gates.test_year == y) & gates.gate].crop)
        out.append(g.assign(y_pred=apply_gate(g, on).to_numpy()))
    return pd.concat(out, ignore_index=True)


def main() -> None:
    ck = ROOT / "results/checkpoints"
    gates = pd.read_csv(ROOT / "data/processed/static/gate_decisions_v5.csv")
    F = pd.read_parquet(ROOT / "results/metrics/v4_predictions.parquet")
    F = F[(F.arm == "F") & F.seed.isin(SEEDS)].drop(columns="arm")

    cfg = load_model_config("itransformer_v5A")
    frame = load_modeling_frame(ROOT / cfg["dataset"], load_data_config(), window="crop_year",
                                common_sample=True, cache_dir=ROOT / "data/processed/cache",
                                phenology=cfg["phenology"], phenology_version=cfg["phenology_version"],
                                stage_features=bool(cfg["stage_features"]))
    preds = []
    for Y in YEARS:
        fy = frame[frame["year"] == Y]
        for s in SEEDS:
            for grp in ("A", "S"):
                pipe = FeaturePipeline.load(ck / f"itransformer_v5{grp}_roll{Y}_s{s}.pipeline.json")
                pipe.require_current_feature_policy()
                gcfg = load_model_config(f"itransformer_v5{grp}")
                for crop in gcfg["crops"]:
                    m = build_model("itransformer", gcfg)
                    m.load(ck / f"itransformer_v5P{grp}_roll{Y}_s{s}_{crop}.pt")
                    preds.append(v2.predict(m, pipe, fy[fy["crop"] == crop]).assign(seed=s))
            print(f"  {Y} s{s}", flush=True)
    P = pd.concat(preds, ignore_index=True)
    cols = ["year", "crop", "landkreis", "seed", "y_true", "y_pred", "y_hist"]
    arms = {"F": F[cols], "G": gated(F[cols], gates), "P": P[cols], "PG": gated(P[cols], gates)}
    A = pd.concat([d.assign(arm=a) for a, d in arms.items()], ignore_index=True)
    k = ["arm", "year", "crop", "landkreis"]
    ens = A.groupby(k, as_index=False).agg(y_true=("y_true", "first"), y_hist=("y_hist", "first"),
                                           y_pred=("y_pred", "mean"), n=("seed", "nunique"))
    assert (ens["n"] == 3).all()
    A = pd.concat([A, ens.drop(columns="n").assign(seed="ens")], ignore_index=True)
    A.to_parquet(ROOT / "results/metrics/v5_predictions.parquet", index=False)

    E = pd.DataFrame([{"arm": a, "seed": s, "year": y, "crop": c, **ev.metrics(g.y_true, g.y_pred, g.y_hist)}
                      for (a, s, y, c), g in A.groupby(["arm", "seed", "year", "crop"])])
    E.to_csv(ROOT / "results/metrics/v5_eval.csv", index=False)
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
                  "per_seed_better": per_seed, "worse_years_by_crop": worse, **crit,
                  "confirmed": bool(all(crit.values()) and len(d) == N_EXPECTED)}
    (ROOT / "results/metrics/v5_hypotheses.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({h: {k2: v[k2] for k2 in ("comparison", "better", "per_seed_better",
                                               "worse_years_by_crop", "confirmed")} for h, v in out.items()}, indent=1))

    e = E[E.seed == "ens"]
    print(e.groupby("arm").apply(lambda g: pd.Series({"beats_hist": int((g.skill > 0).sum()),
                                                        "median_skill": g.skill.median()})).round(3).to_string())
    # validation vs actual 2019-2025
    V = A[(A.seed == "ens") & A.year.between(2019, 2025)].copy()
    V["ape"] = 100 * (V.y_pred - V.y_true).abs() / V.y_true
    V["ape_hist"] = 100 * (V.y_hist - V.y_true).abs() / V.y_true
    val = V.groupby(["arm", "crop"]).agg(mape=("ape", "mean"), mape_hist=("ape_hist", "mean")).reset_index()
    nat = V.groupby(["arm", "crop", "year"]).agg(t=("y_true", "mean"), p=("y_pred", "mean"), h=("y_hist", "mean")).reset_index()
    nat["dir"] = np.sign(nat.p - nat.h) == np.sign(nat.t - nat.h)
    val = val.merge(nat.groupby(["arm", "crop"]).dir.sum().rename("direction_right_of_7").reset_index())
    val.to_csv(ROOT / "results/metrics/v5_validation_2019_2025.csv", index=False)
    print(val.pivot_table(index="crop", columns="arm", values="mape").round(2).assign(
        hist=val.groupby("crop").mape_hist.first().round(2)).to_string())
    print("all crops MAPE 2019-2025:", V.groupby("arm").ape.mean().round(2).to_dict(),
          "| hist", round(V[V.arm == "F"].ape_hist.mean(), 2))
    grp = V.assign(g=np.where(V.crop.isin(AUTUMN), "autumn", "spring")).groupby(["g", "arm"]).ape.mean().unstack()
    print(grp.round(2).to_string())
    print("gate on (crop-folds of 100):", int(gates.gate.sum()), gates.groupby("crop").gate.sum().astype(int).to_dict())


if __name__ == "__main__":
    main()

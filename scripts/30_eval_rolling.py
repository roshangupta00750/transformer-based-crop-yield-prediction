"""Score the yearly rolling test: every checkpoint on its own harvest year.

Checkpoints are named ``{model}_roll{Y}_{arm}_s{seed}`` (scripts/sbatch_rolling.sh):
trained on years <= Y-2, early-stopped on Y-1, scored here on Y only.
Every row is one (model, arm, seed, year, crop) -- yields are annual, so
nothing is pooled across years. seed="ens" is the mean prediction over the
available seeds of that (model, arm, year).

Scores per crop-year, against the district historical average (the
pipeline's train-only baseline incl. trend):
    rmse, hist_rmse    dt/ha
    skill              1 - SSE_model / SSE_historical  (0 = no better than history)
    actual_vs_hist_pct / pred_vs_hist_pct   the year's mean deviation, real vs predicted
    direction_ok       predicted the sign of that deviation correctly

Writes results/metrics/rolling_eval.csv and, per sample,
results/metrics/rolling_predictions.parquet (input for the adaptive weighting).
"""
from __future__ import annotations

import re
import sys
from collections import defaultdict
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.data.dataset import (CropYieldDataset, FeaturePipeline, collate_pad,  # noqa: E402
                              prepare_modeling_frame, require_current_dataset_policy)
from src.models import build_model  # noqa: E402
from src.utils import load_data_config, load_experiment_config, load_model_config  # noqa: E402

PAT = re.compile(r"^(?P<model>[a-z]+)_roll(?P<year>\d{4})_(?P<arm>drv|nodrv)_s(?P<seed>\d+)$")


def score(yt, yp, yh) -> dict:
    sse, ss_hist = float(((yt - yp) ** 2).sum()), float(((yt - yh) ** 2).sum())
    a = (yt.mean() / yh.mean() - 1) * 100
    p = (yp.mean() / yh.mean() - 1) * 100
    return {"n": len(yt), "rmse": float(np.sqrt(np.mean((yt - yp) ** 2))),
            "hist_rmse": float(np.sqrt(np.mean((yt - yh) ** 2))),
            "skill": 1 - sse / ss_hist if ss_hist > 0 else np.nan,
            "actual_vs_hist_pct": a, "pred_vs_hist_pct": p,
            "direction_ok": bool(np.sign(a) == np.sign(p))}


def main() -> None:
    dc, ec = load_data_config(), load_experiment_config()
    ck = ROOT / ec["log"]["checkpoints_dir"]
    found = []
    for f in sorted(ck.glob("*_roll*.pipeline.json")):
        m = PAT.match(f.name.replace(".pipeline.json", ""))
        if m:
            found.append(m.groupdict() | {"name": m.group(0)})
    if not found:
        print("no rolling checkpoints found"); sys.exit(1)
    print(f"{len(found)} rolling checkpoints")

    unified = ROOT / dc["paths"]["unified_dataset"]
    require_current_dataset_policy(unified)
    df = prepare_modeling_frame(pd.read_parquet(unified), dc)

    samples = []
    for c in found:
        year = int(c["year"])
        pipe = FeaturePipeline.load(ck / f"{c['name']}.pipeline.json")
        cfg_name = f"{c['model']}_roll_{c['arm']}"
        model = build_model(c["model"], load_model_config(cfg_name))
        wpath = next((ck / f"{c['name']}{e}" for e in (".pt", ".ubj", ".json")
                      if (ck / f"{c['name']}{e}").exists()), None)
        if wpath is None:
            print(f"{c['name']}: weights missing, skipped"); continue
        model.load(wpath)
        for crop in dc["crops"]:
            d = df[(df["crop"] == crop) & (df["year"] == year)]
            if d.empty:
                continue
            ds = CropYieldDataset(pipe.transform_features(d), pipe.spec, pipe)
            if len(ds) == 0:
                continue
            loader = DataLoader(ds, batch_size=256,
                                collate_fn=partial(collate_pad, pad_to=pipe.sequence_length_))
            zp = np.asarray(model.predict(loader)).ravel()
            zt = np.asarray(ds._targets_z).ravel()
            lk = np.array([k[0] for k in ds.keys]); cr = np.array([k[1] for k in ds.keys])
            yr = np.array([k[2] for k in ds.keys])
            samples.append(pd.DataFrame({
                "model": c["model"], "arm": c["arm"], "seed": c["seed"], "year": year,
                "crop": crop, "landkreis": lk,
                "y_true": pipe.inverse_transform_observed_array(lk, cr, zt, yr),
                "y_pred": pipe.inverse_transform_array(lk, cr, zp, yr),
                "y_hist": pipe.inverse_transform_array(lk, cr, np.zeros_like(zp), yr)}))
        print(f"  scored {c['name']}", flush=True)

    sm = pd.concat(samples, ignore_index=True)
    # seed ensemble: mean prediction per district over the seeds of (model, arm, year)
    ens = (sm.groupby(["model", "arm", "year", "crop", "landkreis"], as_index=False)
             .agg(y_true=("y_true", "first"), y_hist=("y_hist", "first"),
                  y_pred=("y_pred", "mean"), n_seeds=("seed", "nunique")))
    ens = ens[ens["n_seeds"] > 1].drop(columns="n_seeds").assign(seed="ens")
    sm = pd.concat([sm, ens], ignore_index=True)
    sm.to_parquet(ROOT / "results/metrics/rolling_predictions.parquet", index=False)

    rows = [{"model": m, "arm": a, "seed": s, "year": y, "crop": c,
             **score(g.y_true.to_numpy(), g.y_pred.to_numpy(), g.y_hist.to_numpy())}
            for (m, a, s, y, c), g in sm.groupby(["model", "arm", "seed", "year", "crop"])]
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results/metrics/rolling_eval.csv", index=False)

    view = out[out["seed"].isin(["ens"]) | (out["model"] == "xgboost")]
    print("\nper model/arm over all crop-years (counts, not averages):")
    for (m, a), g in view.groupby(["model", "arm"]):
        print(f"  {m:13s} {a:6s} crop-years={len(g):3d}  beat history={int((g.skill > 0).sum()):3d}  "
              f"direction right={int(g.direction_ok.sum()):3d}")
    print("\nWrote results/metrics/rolling_eval.csv and rolling_predictions.parquet")


if __name__ == "__main__":
    main()

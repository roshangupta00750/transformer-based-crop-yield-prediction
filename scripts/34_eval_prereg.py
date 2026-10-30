"""Evaluate the pre-registered experiment (paper/preregistration_2026-09-26.md).

Checkpoints: itransformer_pre{A,B,C,D}_roll{Y}_s{42,43,44} (scripts/sbatch_prereg.sh).
Each is scored on its own harvest year Y only. Outputs, one row per
(arm, seed, horizon, year, crop) -- never pooled across years only:

    results/metrics/prereg_eval.csv          R², RMSE, MAE, r, skill, direction (§5)
    results/metrics/prereg_uncertainty.csv   80% interval coverage/width, seed spread (§6.2)
    results/metrics/prereg_hypotheses.json   H1-H3 decisions with every count (§6)
    results/metrics/prereg_predictions.parquet  per-district predictions

seed="ens" = mean prediction of the three seeds. horizon=0 = end of season;
30/60/90/120 = §6.1 (arms A and B): the full timeline is kept and every day
after (harvest - h) is set to missing (feature -> training mean, *_missing -> 1,
*_observed -> 0, *_stale -> 1) -- the sequence is never shortened.
"""
from __future__ import annotations

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
                              load_modeling_frame, require_current_dataset_policy)
from src.models import build_model  # noqa: E402
from src.utils import load_data_config, load_experiment_config, load_model_config  # noqa: E402

PAT = re.compile(r"^itransformer_pre(?P<arm>[ABCD])_roll(?P<year>\d{4})_s(?P<seed>\d+)$")
HORIZONS = [30, 60, 90, 120]
HORIZON_ARMS = {"A", "B"}
WINTER = ["winter_wheat", "winter_barley", "rye", "triticale", "winter_rapeseed"]
HYP = {  # id: (new arm, reference arm, crops (None = all), threshold, n)
    "H1": ("B", "A", WINTER, 32, 50),
    "H2": ("C", "B", None, 59, 100),
    "H3": ("D", "C", None, 59, 100),
}


def metrics(yt, yp, yh) -> dict:
    yt, yp, yh = map(np.asarray, (yt, yp, yh))
    sse, sst, ss_hist = ((yt - yp) ** 2).sum(), ((yt - yt.mean()) ** 2).sum(), ((yt - yh) ** 2).sum()
    a = (yt.mean() / yh.mean() - 1) * 100
    p = (yp.mean() / yh.mean() - 1) * 100
    return {"n": len(yt), "r2": 1 - sse / sst if sst > 0 else np.nan,
            "rmse": float(np.sqrt(np.mean((yt - yp) ** 2))),
            "mae": float(np.mean(np.abs(yt - yp))),
            "r": float(np.corrcoef(yt, yp)[0, 1]) if len(yt) > 2 else np.nan,
            "skill": 1 - sse / ss_hist if ss_hist > 0 else np.nan,
            "actual_vs_hist_pct": a, "pred_vs_hist_pct": p,
            "direction_ok": bool(np.sign(a) == np.sign(p))}


def mask_after(raw: pd.DataFrame, pipe: FeaturePipeline, h: int, phen: dict) -> pd.DataFrame:
    """§6.1: keep the timeline, set every day after (harvest - h) to missing."""
    out = raw.copy()
    harvest = (pd.to_datetime(out["year"].astype(str) + "-01-01")
               + pd.to_timedelta(out["crop"].map(lambda c: phen[c]["harvest_doy"]) - 1, unit="D"))
    late = pd.to_datetime(out["date"]) > harvest - pd.Timedelta(days=h)
    # only columns already in the frame: derived inputs (district anomalies, recent-yield
    # lags) are rebuilt by transform_features from the masked raw columns; masking them
    # here would create all-NaN columns that transform_features then keeps (v10 audit)
    for c in [c for c in pipe.spec.past_observed if c in out.columns]:
        if c.endswith("_missing") or c.endswith("_stale"):
            out.loc[late, c] = 1
        elif c.endswith("_observed"):
            out.loc[late, c] = 0
        else:
            out.loc[late, c] = pipe.feature_mean_[c]
    return out


def predict(model, pipe, raw) -> pd.DataFrame:
    ds = CropYieldDataset(pipe.transform_features(raw), pipe.spec, pipe)
    loader = DataLoader(ds, batch_size=256, collate_fn=partial(collate_pad, pad_to=pipe.sequence_length_))
    zp = np.asarray(model.predict(loader)).ravel()
    zt = np.asarray(ds._targets_z).ravel()
    lk = np.array([k[0] for k in ds.keys]); cr = np.array([k[1] for k in ds.keys])
    yr = np.array([k[2] for k in ds.keys])
    return pd.DataFrame({"landkreis": lk, "crop": cr, "year": yr,
                         "y_true": pipe.inverse_transform_observed_array(lk, cr, zt, yr),
                         "y_pred": pipe.inverse_transform_array(lk, cr, zp, yr),
                         "y_hist": pipe.inverse_transform_array(lk, cr, np.zeros_like(zp), yr)})


def main() -> None:
    dc, ec = load_data_config(), load_experiment_config()
    ck = ROOT / ec["log"]["checkpoints_dir"]
    unified = ROOT / dc["paths"]["unified_dataset"]
    require_current_dataset_policy(unified)
    cache = ROOT / "data/processed/cache"
    runs = [m.groupdict() | {"name": m.group(0)} for f in sorted(ck.glob("itransformer_pre*_roll*.pt"))
            if (m := PAT.match(f.stem))]
    if not runs:
        sys.exit("no pre-registered checkpoints found")
    print(f"{len(runs)} checkpoints")
    frames: dict[str, pd.DataFrame] = {}
    preds = []
    for r in runs:
        pipe = FeaturePipeline.load(ck / f"{r['name']}.pipeline.json")
        pipe.require_current_feature_policy()
        key = (pipe.window_, pipe.common_sample_)
        if key not in frames:
            frames[key] = load_modeling_frame(unified, dc, window=key[0], common_sample=key[1], cache_dir=cache)
        f = frames[key]
        raw = f[f["year"] == int(r["year"])]
        model = build_model("itransformer", load_model_config(f"itransformer_pre{r['arm']}"))
        model.load(ck / f"{r['name']}.pt")
        hs = [0] + (HORIZONS if r["arm"] in HORIZON_ARMS else [])
        for h in hs:
            p = predict(model, pipe, raw if h == 0 else mask_after(raw, pipe, h, dc["phenology"]))
            preds.append(p.assign(arm=r["arm"], seed=r["seed"], horizon=h))
        print(f"  {r['name']}", flush=True)

    P = pd.concat(preds, ignore_index=True)
    k = ["arm", "horizon", "year", "crop", "landkreis"]
    ens = (P.groupby(k, as_index=False)
             .agg(y_true=("y_true", "first"), y_hist=("y_hist", "first"),
                  y_pred=("y_pred", "mean"), seed_sd=("y_pred", "std"), n_seeds=("seed", "nunique")))
    P = pd.concat([P, ens.drop(columns=["seed_sd", "n_seeds"]).assign(seed="ens")], ignore_index=True)
    P.to_parquet(ROOT / "results/metrics/prereg_predictions.parquet", index=False)

    rows = [{"arm": a, "seed": s, "horizon": h, "year": y, "crop": c, **metrics(g.y_true, g.y_pred, g.y_hist)}
            for (a, s, h, y, c), g in P.groupby(["arm", "seed", "horizon", "year", "crop"])]
    E = pd.DataFrame(rows)
    E.to_csv(ROOT / "results/metrics/prereg_eval.csv", index=False)

    # ---- §6.2 uncertainty (end of season, seed ensemble) -------------------
    U = []
    e0 = P[(P.seed == "ens") & (P.horizon == 0)].assign(err=lambda d: d.y_true - d.y_pred)
    sd = ens[ens.horizon == 0].groupby(["arm", "year", "crop"])["seed_sd"].mean()
    for (a, c), g in e0.groupby(["arm", "crop"]):
        for y in sorted(g.year.unique()):
            past = g[g.year < y]["err"]
            if g[g.year < y].year.nunique() < 2:
                continue
            q10, q90 = np.percentile(past, [10, 90])
            cur = g[g.year == y]
            inside = ((cur.y_true >= cur.y_pred + q10) & (cur.y_true <= cur.y_pred + q90)).mean()
            U.append({"arm": a, "crop": c, "year": y, "coverage_80": inside, "width": q90 - q10,
                      "seed_spread": sd.get((a, y, c), np.nan)})
    pd.DataFrame(U).to_csv(ROOT / "results/metrics/prereg_uncertainty.csv", index=False)

    # ---- §6 hypotheses -------------------------------------------------------
    S = E[E.horizon == 0].pivot_table(index=["year", "crop"], columns=["arm", "seed"], values="skill")
    out = {}
    for hid, (new, ref, crops, thr, n_expected) in HYP.items():
        if (new, "ens") not in S or (ref, "ens") not in S:
            out[hid] = {"status": "arm missing"}; continue
        sub = S if crops is None else S[S.index.get_level_values("crop").isin(crops)]
        better = sub[(new, "ens")] > sub[(ref, "ens")]
        per_seed = {s: int((sub[(new, s)] > sub[(ref, s)]).sum())
                    for s in ("42", "43", "44") if (new, s) in sub and (ref, s) in sub}
        worse_by_crop = (~better).groupby(level="crop").sum()
        crit_count = int(better.sum()) >= thr
        crit_seeds = len(per_seed) == 3 and all(v > len(sub) / 2 for v in per_seed.values())
        crit_veto = bool((worse_by_crop < 7).all())
        out[hid] = {"comparison": f"{new} vs {ref}", "crop_years": int(len(sub)),
                    "expected_crop_years": n_expected, "better": int(better.sum()),
                    "threshold": thr, "per_seed_better": per_seed,
                    "worse_years_by_crop": worse_by_crop.astype(int).to_dict(),
                    "criterion_count": crit_count, "criterion_all_seeds": crit_seeds,
                    "criterion_no_crop_veto": crit_veto,
                    "confirmed": bool(crit_count and crit_seeds and crit_veto)}
    (ROOT / "results/metrics/prereg_hypotheses.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

"""Score the crop-aware experiment: crop-blind vs crop-aware iTransformer.

Both arms are scored on the identical 2024-25 test population, per crop, in
the terms that matter for yield prediction:

    rmse           model error in dt/ha
    hist_rmse      error of the district's historical average (the yardstick)
    pct_vs_hist    % difference in error vs the historical average (negative = better)
    skill          1 - SSE_model / SSE_hist in anomaly space (0 = no better than
                   the historical average; this is the honest skill score)
    r2             absolute-space R2, kept only for comparison with older reports

Writes results/metrics/cropaware_comparison.csv and prints a verdict.

Usage:
    python scripts/24_eval_cropaware.py [--arms cropbase,cropaware]
"""
from __future__ import annotations

import argparse
import sys
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import make_temporal_split
from src.data.dataset import (CropYieldDataset, FeaturePipeline, collate_pad,
                              prepare_modeling_frame,
                              require_current_dataset_policy)
from src.models import build_model
from src.utils import (PROJECT_ROOT, load_data_config, load_experiment_config,
                       load_model_config)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", default="cropbase,cropaware")
    ap.add_argument("--seeds", default="",
                    help="Extra seeds, e.g. 43,44: also scores itransformer_<arm>_s<seed>"
                         " and the seed-average ensemble <arm>_ens.")
    args = ap.parse_args()
    base_arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    seeds = [x.strip() for x in args.seeds.split(",") if x.strip()]
    arms = [a for b in base_arms for a in [b] + [f"{b}_s{x}" for x in seeds]]
    ens: dict = {}   # (base_arm, crop) -> list of predictions in dt/ha
    samples: list = []   # per-sample rows for the per-year table

    dc, ec = load_data_config(), load_experiment_config()
    ck = PROJECT_ROOT / ec["log"]["checkpoints_dir"]
    unified = PROJECT_ROOT / dc["paths"]["unified_dataset"]
    require_current_dataset_policy(unified)
    df = prepare_modeling_frame(pd.read_parquet(unified), dc)
    sp = make_temporal_split(
        df, train_end_year=dc["splits"]["temporal"]["train_end_year"],
        val_years=dc["splits"]["temporal"]["val_years"],
        test_years=dc["splits"]["temporal"]["test_years"])
    test = df.loc[sp.test]

    rows = []
    for arm in arms:
        name = f"itransformer_{arm}"
        ck_path, pipe_path = ck / f"{name}.pt", ck / f"{name}.pipeline.json"
        if not (ck_path.exists() and pipe_path.exists()):
            print(f"{name}: checkpoint missing, skipped")
            continue
        pipe = FeaturePipeline.load(pipe_path)
        pipe.require_current_feature_policy()
        # The checkpoint carries its own config and token layout; the yaml
        # is only needed to instantiate, so fall back to the base config.
        cfg_path = PROJECT_ROOT / "configs" / "models" / f"{name}.yaml"
        model = build_model("itransformer", load_model_config(
            name if cfg_path.exists() else "itransformer"))
        model.load(ck_path)
        for crop in dc["crops"]:
            d = test[test["crop"] == crop]
            if d.empty:
                continue
            ds = CropYieldDataset(pipe.transform_features(d), pipe.spec, pipe)
            if len(ds) == 0:
                continue
            loader = DataLoader(ds, batch_size=256,
                                collate_fn=partial(collate_pad,
                                                   pad_to=pipe.sequence_length_))
            zp = np.asarray(model.predict(loader)).ravel()
            zt = np.asarray(ds._targets_z).ravel()
            lk = np.array([k[0] for k in ds.keys])
            cr = np.array([k[1] for k in ds.keys])
            yr = np.array([k[2] for k in ds.keys])
            yp = pipe.inverse_transform_array(lk, cr, zp, yr)
            yt = pipe.inverse_transform_observed_array(lk, cr, zt, yr)
            yh = pipe.inverse_transform_array(lk, cr, np.zeros_like(zp), yr)
            samples.append(pd.DataFrame({"arm": arm, "crop": crop, "year": yr,
                                         "yt": yt, "yp": yp, "yh": yh}))
            base = arm.split("_s")[0]
            ens.setdefault((base, crop), {"yt": yt, "yh": yh, "preds": []})["preds"].append(yp)
            rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
            hist = float(np.sqrt(np.mean((yt - yh) ** 2)))
            sst = float(((yt - yt.mean()) ** 2).sum())
            rows.append({
                "arm": arm, "crop": crop, "n": len(yt),
                "rmse": rmse, "hist_rmse": hist,
                "pct_vs_hist": (rmse / hist - 1) * 100,
                "skill": 1 - float(((yt - yp) ** 2).sum()) / float(((yt - yh) ** 2).sum()),
                "r2": 1 - float(((yt - yp) ** 2).sum()) / sst,
            })
            r = rows[-1]
            print(f"{arm:10s} {crop:16s} rmse={rmse:7.2f} hist={hist:7.2f} "
                  f"{r['pct_vs_hist']:+6.1f}%  skill={r['skill']:+.3f}", flush=True)

    # Seed-average ensemble: mean prediction over every seed of an arm.
    for (base, crop), e in ens.items():
        if len(e["preds"]) < 2:
            continue
        yt, yh, yp = e["yt"], e["yh"], np.mean(e["preds"], axis=0)
        samples.append(pd.DataFrame({"arm": f"{base}_ens", "crop": crop,
            "year": samples[[i for i, f in enumerate(samples) if f["arm"].iat[0] == base and f["crop"].iat[0] == crop][0]]["year"].to_numpy(),
            "yt": yt, "yp": yp, "yh": yh}))
        rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
        hist = float(np.sqrt(np.mean((yt - yh) ** 2)))
        rows.append({"arm": f"{base}_ens", "crop": crop, "n": len(yt),
                     "rmse": rmse, "hist_rmse": hist,
                     "pct_vs_hist": (rmse / hist - 1) * 100,
                     "skill": 1 - float(((yt - yp) ** 2).sum()) / float(((yt - yh) ** 2).sum()),
                     "r2": 1 - float(((yt - yp) ** 2).sum()) / float(((yt - yt.mean()) ** 2).sum())})
    if not rows:
        print("nothing scored"); sys.exit(1)
    out = pd.DataFrame(rows)
    path = PROJECT_ROOT / ec["log"]["metrics_dir"] / "cropaware_comparison.csv"
    out.to_csv(path, index=False)
    print(f"\nWrote {path}\n")
    summ = out.groupby("arm").agg(
        mean_pct_vs_hist=("pct_vs_hist", "mean"), mean_skill=("skill", "mean"),
        crops_beating_hist=("pct_vs_hist", lambda s: int((s < 0).sum())),
        mean_r2=("r2", "mean"))
    print(summ.round(3).to_string())

    # Per crop, per harvest year: yield is annual, so this is the primary view.
    sm = pd.concat(samples, ignore_index=True)
    def _score(g):
        sse, ss_hist = ((g.yt - g.yp) ** 2).sum(), ((g.yt - g.yh) ** 2).sum()
        return pd.Series({"n": len(g), "rmse": np.sqrt(((g.yt - g.yp) ** 2).mean()),
                          "hist_rmse": np.sqrt(((g.yt - g.yh) ** 2).mean()),
                          "skill": 1 - sse / ss_hist,
                          "actual_vs_hist_pct": (g.yt.mean() / g.yh.mean() - 1) * 100,
                          "pred_vs_hist_pct": (g.yp.mean() / g.yh.mean() - 1) * 100})
    py = sm.groupby(["arm", "crop", "year"]).apply(_score).reset_index()
    py["direction_ok"] = np.sign(py.actual_vs_hist_pct) == np.sign(py.pred_vs_hist_pct)
    ypath = path.with_name("cropaware_comparison_per_year.csv")
    py.to_csv(ypath, index=False)
    print(f"\nPer-year table: {ypath}")
    print(py.groupby("arm").agg(mean_skill_per_year=("skill", "mean"),
                                crop_years_beating_hist=("skill", lambda s: int((s > 0).sum())),
                                crop_years=("skill", "size"),
                                year_direction_right=("direction_ok", "sum")).round(3).to_string())
    if {"cropbase", "cropaware"} <= set(summ.index):
        d = summ.loc["cropaware"] - summ.loc["cropbase"]
        print(f"\nVERDICT crop-aware vs crop-blind: error vs historical "
              f"{d['mean_pct_vs_hist']:+.2f} pts, skill {d['mean_skill']:+.3f}, "
              f"crops beating historical {int(d['crops_beating_hist']):+d}")


if __name__ == "__main__":
    main()

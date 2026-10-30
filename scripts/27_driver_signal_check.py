"""Quick check: do the new drivers carry year-to-year yield signal?

Before rebuilding the dataset, test whether soil moisture, solar radiation,
VPD, water balance and fPAR add predictive skill on years a model has not
seen. Two otherwise identical XGBoost models:

    A  current inputs     (tmax, tmin, precip, humidity, ndvi, lai)
    B  A + new drivers    (swvl1-3, radiation, VPD, PET, AET, water balance, fPAR)

Every variable is summarised per month for the 6 months before the end of the
crop's season window. Evaluation is leave-one-year-out over the TRAINING years
only (2004-2022; test years 2024-25 stay untouched). The target is the yield
anomaly against the district's mean over the *other* training years, so the
baseline never sees the held-out year. Scores are per crop, per year:
skill = 1 - SSE_model / SSE_historical_average (0 = no better than history).

Writes results/metrics/driver_signal_check.csv (one row per crop-year-arm).
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import xgboost as xgb

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.utils import load_data_config  # noqa: E402

DC = load_data_config()
TRAIN_END = DC["splits"]["temporal"]["train_end_year"]      # 2022
YEARS = list(range(2004, TRAIN_END + 1))
N_MONTHS = 6


def es(t_c):
    return 0.6108 * np.exp(17.27 * t_c / (t_c + 237.3))


def load_daily() -> pd.DataFrame:
    cols = ["landkreis", "crop", "year", "date", "yield_dt_per_ha",
            "temperature_air_max_2m", "temperature_air_min_2m",
            "precipitation_height", "humidity", "ndvi", "lai"]
    # The stored table holds full calendar years; the models cut each crop-year
    # to its sowing->harvest window (winter crops reach into the previous
    # autumn). Apply the SAME windowing here, or "months before harvest" means
    # July-December for every crop. (This was the bug in the first runs.)
    from src.data.dataset import prepare_modeling_frame
    d = prepare_modeling_frame(pd.read_parquet(ROOT / DC["paths"]["unified_dataset"]), DC)[cols]
    d["date"] = pd.to_datetime(d["date"]).astype("datetime64[ns]")
    d["landkreis"] = d["landkreis"].astype(str)
    d = d[d["year"] <= TRAIN_END]

    era = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(ROOT / "data/raw/drivers_v4/era5_*.csv")))])
    era["date"] = pd.to_datetime(era["date"]).astype("datetime64[ns]")
    t, td = era["t2m"] - 273.15, era["d2m"] - 273.15
    era["vpd"] = (es(t) - es(td)).clip(lower=0)
    era["ssrd"] = era["ssrd"] / 1e6                       # MJ/m2/day
    era["pet"] = -era["pev"] * 1000                       # mm/day
    era["aet"] = -era["evap"] * 1000
    era = era[["landkreis", "date", "swvl1", "swvl2", "swvl3", "ssrd", "vpd", "pet", "aet"]]

    fp = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(ROOT / "data/raw/drivers_v4/fpar_*.csv")))])
    fp["date"] = pd.to_datetime(fp["date"]).astype("datetime64[ns]")
    # 8-day composites -> carry forward to daily within the window
    fp = fp.sort_values("date")

    d = d.merge(era, on=["landkreis", "date"], how="left")
    d = pd.merge_asof(d.sort_values("date"), fp[["landkreis", "date", "fpar"]],
                      on="date", by="landkreis", direction="backward",
                      tolerance=pd.Timedelta(days=16))
    d["cwb"] = d["precipitation_height"] - d["pet"]
    d["hot"] = (d["temperature_air_max_2m"] > 30).astype(float)
    d["dry_air"] = (d["vpd"] > 1.5).astype(float)
    return d


def monthly_features(d: pd.DataFrame) -> pd.DataFrame:
    key = ["landkreis", "crop", "year"]
    end = d.groupby(key)["date"].transform("max")
    d = d.assign(m=((end - d["date"]).dt.days // 30))
    d = d[d["m"] < N_MONTHS]
    agg_a = {"temperature_air_max_2m": "mean", "temperature_air_min_2m": "mean",
             "precipitation_height": "sum", "humidity": "mean", "ndvi": "mean",
             "lai": "mean", "hot": "sum"}
    agg_b = {"swvl1": "mean", "swvl2": "mean", "swvl3": "mean", "ssrd": "sum",
             "vpd": "mean", "dry_air": "sum", "pet": "sum", "aet": "sum",
             "cwb": "sum", "fpar": "mean"}
    g = d.groupby(key + ["m"]).agg({**agg_a, **agg_b})
    wide = g.unstack("m")
    wide.columns = [f"{v}_m{m}" for v, m in wide.columns]
    wide = wide.reset_index()
    y = d.groupby(key)["yield_dt_per_ha"].first().reset_index()
    out = y.merge(wide, on=key)
    a_cols = [c for c in wide.columns if c.rsplit("_m", 1)[0] in agg_a]
    b_cols = [c for c in wide.columns if c.rsplit("_m", 1)[0] in agg_b]
    return out, a_cols, b_cols


def main() -> None:
    print("loading daily data + drivers ...", flush=True)
    d = load_daily()
    print(f"  daily rows {len(d):,}; new-driver coverage "
          f"{d['swvl3'].notna().mean():.1%} (era5), {d['fpar'].notna().mean():.1%} (fpar)", flush=True)
    X, a_cols, b_cols = monthly_features(d)
    X = X.dropna(subset=["yield_dt_per_ha"])
    crops = sorted(X["crop"].unique())
    for c in crops:
        X[f"crop_{c}"] = (X["crop"] == c).astype(float)
    crop_cols = [f"crop_{c}" for c in crops]
    print(f"  samples {len(X):,}; features A={len(a_cols)} B={len(a_cols) + len(b_cols)}", flush=True)

    rows = []
    for yr in YEARS:
        tr, te = X[X["year"] != yr].copy(), X[X["year"] == yr].copy()
        # historical average from the OTHER training years only
        hist = tr.groupby(["landkreis", "crop"])["yield_dt_per_ha"].mean().rename("hist")
        crop_mean = tr.groupby("crop")["yield_dt_per_ha"].mean().rename("crop_hist")
        for part in (tr, te):
            part["hist"] = part.join(hist, on=["landkreis", "crop"])["hist"]
            part["hist"] = part["hist"].fillna(part.join(crop_mean, on="crop")["crop_hist"])
        # leave-one-out style anomaly for training rows too (hist includes own year there;
        # acceptable for fitting, the scored year never enters its own baseline)
        tr["anom"] = tr["yield_dt_per_ha"] - tr["hist"]
        for arm, cols in (("A_current", a_cols), ("B_plus_drivers", a_cols + b_cols)):
            feats = cols + crop_cols
            model = xgb.XGBRegressor(n_estimators=400, max_depth=5, learning_rate=0.05,
                                     subsample=0.8, colsample_bytree=0.7,
                                     min_child_weight=5, n_jobs=8, random_state=42)
            model.fit(tr[feats], tr["anom"])
            pred = te["hist"] + model.predict(te[feats])
            for crop, g in te.assign(pred=pred).groupby("crop"):
                sse = ((g.yield_dt_per_ha - g.pred) ** 2).sum()
                ss_hist = ((g.yield_dt_per_ha - g["hist"]) ** 2).sum()
                rows.append({"year": yr, "crop": crop, "arm": arm, "n": len(g),
                             "skill": 1 - sse / ss_hist,
                             "rmse": float(np.sqrt(((g.yield_dt_per_ha - g.pred) ** 2).mean())),
                             "hist_rmse": float(np.sqrt(((g.yield_dt_per_ha - g["hist"]) ** 2).mean()))})
        done = pd.DataFrame(rows)
        s = done[done.year == yr].pivot(index="crop", columns="arm", values="skill")
        print(f"  {yr}: A beats hist in {int((s['A_current'] > 0).sum())}/{len(s)} crops, "
              f"B in {int((s['B_plus_drivers'] > 0).sum())}/{len(s)}; "
              f"B better than A in {int((s['B_plus_drivers'] > s['A_current']).sum())}/{len(s)}", flush=True)

    out = pd.DataFrame(rows)
    path = ROOT / "results/metrics/driver_signal_check.csv"
    out.to_csv(path, index=False)
    w = out.pivot_table(index=["year", "crop"], columns="arm", values="skill")
    print(f"\nWrote {path}")
    print(f"crop-years scored: {len(w)}")
    print(f"beat historical average: A {int((w.A_current > 0).sum())}, "
          f"B {int((w.B_plus_drivers > 0).sum())}")
    print(f"B better than A: {int((w.B_plus_drivers > w.A_current).sum())} of {len(w)} crop-years")
    print(f"median skill: A {w.A_current.median():+.3f}, B {w.B_plus_drivers.median():+.3f}")
    print("\nper crop: crop-years where B beats A / beats history")
    for crop, g in w.groupby(level="crop"):
        print(f"  {crop:16s} B>A {int((g.B_plus_drivers > g.A_current).sum()):2d}/{len(g)}  "
              f"hist-beaten A {int((g.A_current > 0).sum()):2d}  B {int((g.B_plus_drivers > 0).sum()):2d}")


if __name__ == "__main__":
    main()

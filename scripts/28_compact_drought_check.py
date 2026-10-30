"""Retest the new drivers as a few agronomic drought/heat indicators.

scripts/27 added 60 monthly driver columns and found no gain: with only ~19
training years, that many inputs let the model chase noise. Here the same data
is condensed into six indicators tied to how crops fail:

    sw_root_min_crit   lowest 7-day mean root-zone soil moisture (28-100 cm)
                       in the two critical months before harvest
    sw_root_season     mean root-zone soil moisture (7-100 cm) over the season
    cwb_season         water balance (precip - potential evaporation), season
    cwb_crit           water balance in the two critical months
    hotdry_crit        days with tmax > 28 C and VPD > 1.5 kPa, critical months
    rad_crit           solar radiation in the two critical months

"Critical months" = 30-90 days before the end of the crop's season window
(flowering / grain filling for cereals, main bulking for maize, beet, potato).

Arms, same leave-one-year-out protocol and per crop-year scoring as scripts/27:
    A  current inputs (unchanged)
    B  A + the six indicators
    C  the six indicators only (lean)
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("chk", ROOT / "scripts/27_driver_signal_check.py")
chk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chk)

KEY = ["landkreis", "crop", "year"]
COMPACT = ["sw_root_min_crit", "sw_root_season", "cwb_season", "cwb_crit",
           "hotdry_crit", "rad_crit"]


def compact_indicators(d: pd.DataFrame) -> pd.DataFrame:
    end = d.groupby(KEY)["date"].transform("max")
    d = d.assign(days_before=(end - d["date"]).dt.days)
    season = d[d["days_before"] < 180]
    crit = d[(d["days_before"] >= 30) & (d["days_before"] < 90)].sort_values(KEY + ["date"])
    crit = crit.assign(
        sw7=crit.groupby(KEY)["swvl3"].transform(lambda s: s.rolling(7, min_periods=4).mean()),
        hotdry=((crit["temperature_air_max_2m"] > 28) & (crit["vpd"] > 1.5)).astype(float))
    out = season.groupby(KEY).agg(
        sw_root_season=("swvl2", "mean"), sw3_season=("swvl3", "mean"),
        cwb_season=("cwb", "sum"))
    out["sw_root_season"] = (out["sw_root_season"] + out.pop("sw3_season")) / 2
    c = crit.groupby(KEY).agg(sw_root_min_crit=("sw7", "min"), cwb_crit=("cwb", "sum"),
                              hotdry_crit=("hotdry", "sum"), rad_crit=("ssrd", "sum"))
    return out.join(c).reset_index()


def main() -> None:
    print("loading daily data + drivers ...", flush=True)
    d = chk.load_daily()
    X, a_cols, _ = chk.monthly_features(d)
    X = X.merge(compact_indicators(d), on=KEY, how="left")
    X = X.dropna(subset=["yield_dt_per_ha"])
    crops = sorted(X["crop"].unique())
    for c in crops:
        X[f"crop_{c}"] = (X["crop"] == c).astype(float)
    crop_cols = [f"crop_{c}" for c in crops]
    print(f"  samples {len(X):,}; indicator coverage "
          f"{X[COMPACT].notna().all(axis=1).mean():.1%}", flush=True)
    print(X[COMPACT].describe().T[["mean", "min", "max"]].round(3).to_string(), flush=True)

    arms = {"A_current": a_cols, "B_plus_indicators": a_cols + COMPACT,
            "C_indicators_only": COMPACT}
    rows = []
    for yr in chk.YEARS:
        tr, te = X[X["year"] != yr].copy(), X[X["year"] == yr].copy()
        hist = tr.groupby(["landkreis", "crop"])["yield_dt_per_ha"].mean().rename("hist")
        crop_mean = tr.groupby("crop")["yield_dt_per_ha"].mean().rename("crop_hist")
        for part in (tr, te):
            part["hist"] = part.join(hist, on=["landkreis", "crop"])["hist"]
            part["hist"] = part["hist"].fillna(part.join(crop_mean, on="crop")["crop_hist"])
        tr["anom"] = tr["yield_dt_per_ha"] - tr["hist"]
        for arm, cols in arms.items():
            feats = cols + crop_cols
            m = xgb.XGBRegressor(n_estimators=400, max_depth=5, learning_rate=0.05,
                                 subsample=0.8, colsample_bytree=0.7, min_child_weight=5,
                                 n_jobs=8, random_state=42)
            m.fit(tr[feats], tr["anom"])
            te["pred"] = te["hist"] + m.predict(te[feats])
            for crop, g in te.groupby("crop"):
                sse = ((g["yield_dt_per_ha"] - g["pred"]) ** 2).sum()
                ss_hist = ((g["yield_dt_per_ha"] - g["hist"]) ** 2).sum()
                rows.append({"year": yr, "crop": crop, "arm": arm, "skill": 1 - sse / ss_hist})
        s = pd.DataFrame(rows)
        s = s[s.year == yr].pivot(index="crop", columns="arm", values="skill")
        print(f"  {yr}: beats hist A {int((s.A_current > 0).sum())}/10  "
              f"B {int((s.B_plus_indicators > 0).sum())}/10  C {int((s.C_indicators_only > 0).sum())}/10 | "
              f"B>A {int((s.B_plus_indicators > s.A_current).sum())}/10", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results/metrics/compact_drought_check.csv", index=False)
    w = out.pivot_table(index=["year", "crop"], columns="arm", values="skill")
    print(f"\ncrop-years: {len(w)}")
    for a in arms:
        print(f"  {a:18s} beats historical average in {int((w[a] > 0).sum()):3d}  | median skill {w[a].median():+.3f}")
    print(f"  B better than A in {int((w.B_plus_indicators > w.A_current).sum())} of {len(w)}")
    print(f"  C better than A in {int((w.C_indicators_only > w.A_current).sum())} of {len(w)}")
    print("\nper crop: years beating historical average (A / B / C) and B>A")
    for crop, g in w.groupby(level="crop"):
        print(f"  {crop:16s} {int((g.A_current > 0).sum()):2d} / {int((g.B_plus_indicators > 0).sum()):2d} / "
              f"{int((g.C_indicators_only > 0).sum()):2d}   B>A {int((g.B_plus_indicators > g.A_current).sum()):2d}/19")


if __name__ == "__main__":
    main()

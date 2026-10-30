"""Merge the pre-registered arm C and arm D inputs into the unified dataset.

Definitions are exactly those of paper/preregistration_2026-09-26.md §4.2-4.3
(plus the change-log clarification for transp_ratio at near-zero demand).

Arm C (daily indicators):
    heavyrain     1 if precipitation_height >= 20 mm (ETCCDI R20mm)
    wetsoil       1 if swvl1 >= the district's 90th percentile of swvl1 over
                  2003-01-01..2015-12-31 (precedes every test year)
    soiltemp      ERA5-Land soil_temperature_level_1 in deg C
    snowdepth     ERA5-Land snow_depth in m
    barefrost     1 if temperature_air_min_2m <= -15 C and snowdepth < 0.05 m
    transp_ratio  ACTUAL / potential evaporation (AET/PET), clipped to [0, 1.5];
                  1.0 when potential evaporation < 0.1 mm/day (no demand)
Arm D (cropland-only satellite, CORINE 2018 class 211):
    ndvi_crop, evi_crop   MOD13Q1, carried forward <= 32 days
    lai_crop, fpar_crop   MOD15A2H, carried forward <= 16 days
    districts with < 1% arable land keep the whole-district values

Left join on (landkreis, date); the row count must not change. A backup of
the current dataset is written first. Sources: data/raw/prereg_v1 (scripts/32),
data/raw/drivers_v4/era5_*.csv (swvl1 reference period).

Usage:
    python scripts/33_merge_prereg_features.py --dry-run
    python scripts/33_merge_prereg_features.py
"""
from __future__ import annotations

import argparse
import glob
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
DATASET = REPO / "data/processed/dataset.parquet"
POLICY = DATASET.with_suffix(".policy.json")
RAW = REPO / "data/raw/prereg_v1"
ARM_C = ["heavyrain", "wetsoil", "soiltemp", "snowdepth", "barefrost", "transp_ratio"]
ARM_D = ["ndvi_crop", "evi_crop", "lai_crop", "fpar_crop"]
MAX_AGE = {"ndvi_crop": 32, "evi_crop": 32, "lai_crop": 16, "fpar_crop": 16}


def read_all(pattern: str) -> pd.DataFrame:
    files = sorted(glob.glob(str(RAW / pattern)))
    if not files:
        sys.exit(f"no files for {pattern} in {RAW}")
    d = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    d["date"] = pd.to_datetime(d["date"]).astype("datetime64[ns]")
    return d.drop_duplicates(["landkreis", "date"])


def wetsoil_threshold() -> pd.Series:
    era = pd.concat(pd.read_csv(f, usecols=["landkreis", "date", "swvl1"])
                    for f in sorted(glob.glob(str(REPO / "data/raw/drivers_v4/era5_*.csv"))))
    era["date"] = pd.to_datetime(era["date"])
    ref = era[(era["date"] >= "2003-01-01") & (era["date"] <= "2015-12-31")]
    return ref.groupby("landkreis")["swvl1"].quantile(0.90).rename("swvl1_p90")


def carry_forward(obs: pd.DataFrame, col: str, days: int, frame: pd.DataFrame) -> pd.Series:
    o = obs[["landkreis", "date", col]].dropna().sort_values("date")
    f = frame[["landkreis", "date"]].reset_index().sort_values("date")
    m = pd.merge_asof(f, o, on="date", by="landkreis", direction="backward",
                      tolerance=pd.Timedelta(days=days))
    return m.set_index("index")[col].reindex(frame.index)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--modis-end-stamp", action="store_true",
                    help="preregistration v10 §2.4: date MOD13Q1 values at period start + 16 d, "
                         "MOD15A2H at + 8 d (when the composite is complete)")
    ap.add_argument("--dataset", default=None,
                    help="merge into this parquet instead (e.g. a forecast-year build); no backup")
    args = ap.parse_args()
    global DATASET, POLICY
    if args.dataset:
        DATASET = Path(args.dataset).resolve()
        POLICY = DATASET.with_suffix(".policy.json")

    df = pd.read_parquet(DATASET)
    n0, date_dtype = len(df), df["date"].dtype
    clash = [c for c in ARM_C + ARM_D if c in df.columns]
    if clash:
        sys.exit(f"columns already present, refusing to merge twice: {clash}")
    df["date"] = pd.to_datetime(df["date"]).astype("datetime64[ns]")
    df["landkreis"] = df["landkreis"].astype(str)

    # ---- arm C ------------------------------------------------------------
    ex = read_all("era5x_*.csv")
    ex["soiltemp"] = ex["stl1"] - 273.15
    ex["snowdepth"] = ex["sd"].clip(lower=0)
    # transp_ratio = ACTUAL / potential evaporation (preregistration change log,
    # 2026-09-26): the ERA5-Land transpiration band in ECMWF/ERA5_LAND/DAILY_AGGR
    # is 0 on every day, so the registered transpiration/PET ratio is replaced
    # by the standard AET/PET water-stress ratio from total_evaporation_sum
    # (already verified, data/raw/drivers_v4).
    ev = pd.concat(pd.read_csv(f, usecols=["landkreis", "date", "evap"])
                   for f in sorted(glob.glob(str(REPO / "data/raw/drivers_v4/era5_*.csv"))))
    ev["date"] = pd.to_datetime(ev["date"]).astype("datetime64[ns]")
    ex = ex.merge(ev.drop_duplicates(["landkreis", "date"]), on=["landkreis", "date"], how="left")
    pet_mm = -ex["pev"] * 1000
    ratio = (ex["evap"] / ex["pev"]).clip(0, 1.5)
    ex["transp_ratio"] = np.where(pet_mm < 0.1, 1.0, ratio)
    ex.loc[ex["evap"].isna() | ex["pev"].isna(), "transp_ratio"] = np.nan
    out = df.merge(ex[["landkreis", "date", "soiltemp", "snowdepth", "transp_ratio"]],
                   on=["landkreis", "date"], how="left")
    if len(out) != n0:
        sys.exit(f"row count changed {n0} -> {len(out)} (era5x); aborting")
    out = out.merge(wetsoil_threshold(), on="landkreis", how="left")
    out["wetsoil"] = (out["swvl1"] >= out["swvl1_p90"]).astype(float)
    out.loc[out["swvl1"].isna() | out["swvl1_p90"].isna(), "wetsoil"] = np.nan
    out = out.drop(columns=["swvl1_p90"])
    out["heavyrain"] = (out["precipitation_height"] >= 20).astype(float)
    out.loc[out["precipitation_height"].isna(), "heavyrain"] = np.nan
    out["barefrost"] = ((out["temperature_air_min_2m"] <= -15)
                        & (out["snowdepth"] < 0.05)).astype(float)
    out.loc[out["temperature_air_min_2m"].isna() | out["snowdepth"].isna(), "barefrost"] = np.nan

    # ---- arm D ------------------------------------------------------------
    vi, la = read_all("vicrop_*.csv"), read_all("laicrop_*.csv")
    if args.modis_end_stamp:
        vi["date"] = vi["date"] + pd.Timedelta(days=16)
        la["date"] = la["date"] + pd.Timedelta(days=8)
    for col, src in (("ndvi_crop", vi), ("evi_crop", vi), ("lai_crop", la), ("fpar_crop", la)):
        out[col] = carry_forward(src, col, MAX_AGE[col], out)
    ar = pd.read_csv(RAW / "arable.csv")
    low = set(ar.loc[ar["arable_share"] < 0.01, "landkreis"])
    fb = out["landkreis"].isin(low)
    for col, whole in (("ndvi_crop", "ndvi"), ("evi_crop", "evi"), ("lai_crop", "lai"), ("fpar_crop", "fpar")):
        out.loc[fb, col] = out.loc[fb, whole]
    print(f"arable-share fallback districts (<1%): {len(low)} "
          f"({sorted(low & set(df['landkreis']))[:8]}...)")

    new = ARM_C + ARM_D
    print("coverage:\n" + out[new].notna().mean().round(4).to_string())
    print(out[new].describe().T[["mean", "min", "max"]].round(3).to_string())
    for a, b in (("ndvi", "ndvi_crop"), ("fpar", "fpar_crop"), ("lai", "lai_crop")):
        print(f"corr {a} vs {b}: {out[a].corr(out[b]):.3f}")
    if args.dry_run:
        print("dry run: nothing written"); return

    out["date"] = out["date"].astype(date_dtype)
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    backup = DATASET.with_name(f"dataset.pre_prereg.{stamp}.parquet")
    if not args.dataset:
        shutil.copy2(DATASET, backup); shutil.copy2(POLICY, backup.with_suffix(".policy.json"))
    out.to_parquet(DATASET, index=False)
    pol = json.loads(POLICY.read_text())
    pol["columns"], pol["rows"] = list(out.columns), len(out)
    pol.setdefault("notes", []).append(
        f"{stamp}: merged pre-registered arm C {ARM_C} and arm D {ARM_D} "
        "(paper/preregistration_2026-09-26.md); causal, policy version unchanged")
    POLICY.write_text(json.dumps(pol, indent=2))
    print(f"backup {backup.name}; wrote {len(out):,} rows, {len(out.columns)} columns")


if __name__ == "__main__":
    main()

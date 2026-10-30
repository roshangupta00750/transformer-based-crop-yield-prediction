"""Merge the ERA5-Land + MODIS fPAR daily drivers into the unified dataset.

Adds water- and energy-stress drivers the models lacked, as daily columns
(left join on landkreis + date; the dataset row count must not change):

    swvl1, swvl2, swvl3   soil moisture 0-7 / 7-28 / 28-100 cm (m3/m3)
    swroot7               7-day trailing mean of root-zone moisture (swvl2+swvl3)/2
    ssrd                  solar radiation (MJ/m2/day)
    vpd                   vapour pressure deficit from daily mean T and dew point (kPa)
    pet                   potential evaporation (mm/day)
    cwb                   climatic water balance: precipitation - pet (mm/day)
    hotdry                1 if tmax > 28 C and vpd > 1.5 kPa, else 0
    fpar                  MODIS fPAR, 8-day, carried forward at most 16 days

Every column is observed on or before its own date (swroot7 is trailing), so
the leakage policy is unchanged and the dataset stays at policy v3; existing
checkpoints keep their own saved feature lists.

Source CSVs: data/raw/drivers_v4/ (scripts/25 + 26).

Usage:
    python scripts/29_merge_driver_features.py --dry-run
    python scripts/29_merge_driver_features.py
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
RAW = REPO / "data/raw/drivers_v4"
NEW_COLS = ["swvl1", "swvl2", "swvl3", "swroot7", "ssrd", "vpd", "pet", "cwb",
            "hotdry", "fpar"]


def es(t_c):
    return 0.6108 * np.exp(17.27 * t_c / (t_c + 237.3))


def load_drivers() -> pd.DataFrame:
    era = pd.concat(pd.read_csv(f) for f in sorted(glob.glob(str(RAW / "era5_*.csv"))))
    era["date"] = pd.to_datetime(era["date"]).astype("datetime64[ns]")
    era["vpd"] = (es(era["t2m"] - 273.15) - es(era["d2m"] - 273.15)).clip(lower=0)
    era["ssrd"] = era["ssrd"] / 1e6
    era["pet"] = (-era["pev"] * 1000).clip(lower=0)
    era = era.sort_values(["landkreis", "date"])
    root = (era["swvl2"] + era["swvl3"]) / 2
    era["swroot7"] = root.groupby(era["landkreis"]).transform(
        lambda s: s.rolling(7, min_periods=4).mean())
    era = era[["landkreis", "date", "swvl1", "swvl2", "swvl3", "swroot7", "ssrd", "vpd", "pet"]]

    fp = pd.concat(pd.read_csv(f) for f in sorted(glob.glob(str(RAW / "fpar_*.csv"))))
    fp["date"] = pd.to_datetime(fp["date"]).astype("datetime64[ns]")
    fp = fp.sort_values("date")
    era = pd.merge_asof(era.sort_values("date"), fp[["landkreis", "date", "fpar"]],
                        on="date", by="landkreis", direction="backward",
                        tolerance=pd.Timedelta(days=16))
    return era


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--dataset", default=None,
                    help="merge into this parquet instead (e.g. a forecast-year build); no backup")
    args = ap.parse_args()
    global DATASET, POLICY
    if args.dataset:
        DATASET = Path(args.dataset).resolve()
        POLICY = DATASET.with_suffix(".policy.json")

    df = pd.read_parquet(DATASET)
    n0 = len(df)
    clash = [c for c in NEW_COLS if c in df.columns]
    if clash:
        sys.exit(f"columns already present, refusing to merge twice: {clash}")
    date_dtype = df["date"].dtype
    df["date"] = pd.to_datetime(df["date"]).astype("datetime64[ns]")
    df["landkreis"] = df["landkreis"].astype(str)

    drv = load_drivers()
    print(f"drivers: {len(drv):,} rows, {drv.landkreis.nunique()} districts, "
          f"{drv.date.min().date()} -> {drv.date.max().date()}")
    out = df.merge(drv, on=["landkreis", "date"], how="left")
    if len(out) != n0:
        sys.exit(f"row count changed {n0} -> {len(out)}; aborting")
    out["cwb"] = out["precipitation_height"] - out["pet"]
    out["hotdry"] = ((out["temperature_air_max_2m"] > 28) & (out["vpd"] > 1.5)).astype(float)
    out.loc[out["vpd"].isna() | out["temperature_air_max_2m"].isna(), "hotdry"] = np.nan

    cov = out[NEW_COLS].notna().mean().round(4)
    print("coverage per new column:\n" + cov.to_string())
    print(out[NEW_COLS].describe().T[["mean", "min", "max"]].round(3).to_string())
    if args.dry_run:
        print("dry run: nothing written")
        return

    out["date"] = out["date"].astype(date_dtype)
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    backup = DATASET.with_name(f"dataset.pre_drivers.{stamp}.parquet")
    if not args.dataset:
        shutil.copy2(DATASET, backup)
        shutil.copy2(POLICY, backup.with_suffix(".policy.json"))
        print(f"backup: {backup.name}")
    out.to_parquet(DATASET, index=False)

    pol = json.loads(POLICY.read_text())
    pol["columns"] = list(out.columns)
    pol["rows"] = len(out)
    pol.setdefault("notes", []).append(
        f"{stamp}: merged daily drivers {NEW_COLS} (ERA5-Land + MODIS fPAR via "
        "Earth Engine); causal, policy version unchanged")
    POLICY.write_text(json.dumps(pol, indent=2))
    print(f"wrote {DATASET.name}: {len(out):,} rows, {len(out.columns)} columns")


if __name__ == "__main__":
    main()

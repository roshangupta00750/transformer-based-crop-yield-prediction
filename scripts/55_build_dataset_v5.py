"""Build dataset_v5: the corrected data foundation (paper/preregistration_2026-10-03_v10.md §2).

Differences from dataset_v4 (scripts/42):
    yields   matched by AGS code (src/data/acquisition/destatis_yield.clean_by_code)
    weather  ERA5-Land daily over CORINE cropland for every district (data/raw/prereg_v1/era5w_*.csv)
             replaces DWD stations-inside-polygon; same column names + wind_speed
    MODIS    stamped at the end of the composite period (scripts/33 --modis-end-stamp)
Everything else as dataset_v4 (phenology, drivers, soil, terrain; harvest years 2000-2025).

Output: data/processed/dataset_v5.parquet (+ .policy.json). Older datasets are untouched.
Leave-one-out variants (preregistration v11):
    --weather hybrid       DWD station series where present, ERA5-Land only where a value is missing
    --weather hyras        DWD HYRAS-DE 1 km cropland means (scripts/61), ERA5-Land wind (v14)
    --yields name          v4's name-matched yields (data/interim/v4/yield.parquet)
    --modis-start-stamp    v4's MODIS dating (period start)
    --out PATH             output parquet (default dataset_v5.parquet)

Usage:  python scripts/55_build_dataset_v5.py [--weather era5|hybrid|hyras] [--yields code|name]
                                              [--modis-start-stamp] [--out data/processed/X.parquet]
"""
from __future__ import annotations

import argparse

import glob
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.data.acquisition import gee_satellite  # noqa: E402
from src.data.acquisition.destatis_yield import clean_by_code  # noqa: E402
from src.data.dataset import build_unified_dataset  # noqa: E402
from src.utils import load_data_config  # noqa: E402

V4 = ROOT / "data/processed/dataset_v4.parquet"
V5 = ROOT / "data/processed/dataset_v5.parquet"
WORK = ROOT / "data/interim/v5"
DWD_V4 = ROOT / "data/interim/v4/weather.parquet"      # cleaned DWD stations (scripts/42)
EXT = ROOT / "data/raw/extend_2000"


def era5_weather() -> pd.DataFrame:
    files = sorted(glob.glob(str(ROOT / "data/raw/prereg_v1/era5w_*.csv")))
    w = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    w["date"] = pd.to_datetime(w["date"])
    w = w[w["date"] >= "2000-01-01"].drop_duplicates(["landkreis", "date"])
    t, td = w["tmean_k"] - 273.15, w["dew_k"] - 273.15
    rh = np.exp(17.625 * td / (243.04 + td)) / np.exp(17.625 * t / (243.04 + t))
    out = pd.DataFrame({
        "landkreis": w["landkreis"], "date": w["date"].dt.tz_localize("UTC"),
        "humidity": rh.clip(0, 1.0),
        "precipitation_height": (w["prec_m"] * 1000).clip(lower=0),
        "radiation_global": np.nan,                       # excluded by the feature policy
        "temperature_air_max_2m": w["tmax_k"] - 273.15,
        "temperature_air_min_2m": w["tmin_k"] - 273.15,
        "wind_speed": np.hypot(w["u10"], w["v10"]),
    })
    print(f"ERA5-Land weather: {len(files)} files, {len(out):,} district-days, {out.landkreis.nunique()} districts, "
          f"{out.date.min().date()}..{out.date.max().date()}")
    return out


def hybrid_weather(era: pd.DataFrame) -> pd.DataFrame:
    """DWD station values where present; ERA5-Land fills only missing values (v11 L-W)."""
    dwd = pd.read_parquet(DWD_V4)
    dwd["date"] = pd.to_datetime(dwd["date"], utc=True)
    m = era.merge(dwd, on=["landkreis", "date"], how="left", suffixes=("_era", ""))
    for c in ("humidity", "precipitation_height", "temperature_air_max_2m", "temperature_air_min_2m"):
        filled = m[c].isna()
        m[c] = m[c].fillna(m[f"{c}_era"])
        print(f"  hybrid {c}: {filled.mean():.1%} of district-days from ERA5-Land")
    m["radiation_global"] = np.nan
    return m[["landkreis", "date", "humidity", "precipitation_height", "radiation_global",
              "temperature_air_max_2m", "temperature_air_min_2m", "wind_speed"]]


def hyras_weather(era: pd.DataFrame) -> pd.DataFrame:
    """DWD HYRAS-DE 1 km cropland means for tmax/tmin/precip/humidity; ERA5-Land wind (v14 §2)."""
    h = pd.read_parquet(ROOT / "data/interim/hyras/hyras_weather.parquet")
    h["date"] = pd.to_datetime(h["date"]).dt.tz_localize("UTC")
    h = h.rename(columns={"tmax": "temperature_air_max_2m", "tmin": "temperature_air_min_2m",
                          "pr": "precipitation_height"}).assign(humidity=lambda d: (d.pop("hurs") / 100).clip(0, 1))
    m = era.merge(h, on=["landkreis", "date"], how="left", suffixes=("_era", ""))
    for c in ("humidity", "precipitation_height", "temperature_air_max_2m", "temperature_air_min_2m"):
        filled = m[c].isna()
        m[c] = m[c].fillna(m[f"{c}_era"])
        print(f"  hyras {c}: {filled.mean():.2%} of district-days from ERA5-Land")
    m["radiation_global"] = np.nan
    return m[["landkreis", "date", "humidity", "precipitation_height", "radiation_global",
              "temperature_air_max_2m", "temperature_air_min_2m", "wind_speed"]]


def main() -> None:
    global V5, WORK
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weather", choices=["era5", "hybrid", "hyras"], default="era5")
    ap.add_argument("--yields", choices=["code", "name"], default="code")
    ap.add_argument("--modis-start-stamp", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.out:
        V5 = (ROOT / args.out).resolve()
        WORK = ROOT / "data/interim" / V5.stem
    cfg = load_data_config()
    WORK.mkdir(parents=True, exist_ok=True)
    w = era5_weather()
    if args.weather == "hybrid":
        w = hybrid_weather(w)
    elif args.weather == "hyras":
        w = hyras_weather(w)
    w.to_parquet(WORK / "weather.parquet", index=False)

    clean_by_code([ROOT / "data/raw/destatis_1999/destatis_41241-01-03-4.csv",
                   ROOT / "data/raw/destatis/destatis_41241-01-03-4.csv"],
                  WORK / "yield_full.parquet", ROOT / "data/raw/geo",
                  mapping_out=ROOT / "data/processed/static/ags_to_nuts3.csv")
    y = pd.read_parquet(WORK / "yield_full.parquet")
    if args.yields == "name":
        y = pd.read_parquet(ROOT / "data/interim/v4/yield.parquet").assign(source="name")
    y[y["year"].between(2000, 2025)].drop(columns="source").to_parquet(WORK / "yield.parquet", index=False)

    sats = [pd.read_parquet(EXT / f"sat_{yy}.parquet") for yy in (2000, 2001, 2002)]
    sats += [pd.read_parquet(ROOT / f"data/raw/gee/_by_year/sat_{yy}.parquet") for yy in range(2003, 2026)]
    (pd.concat(sats, ignore_index=True).sort_values(["landkreis", "date"])
       .to_parquet(WORK / "gee_satellite.parquet", index=False))
    gee_satellite.clean(WORK / "gee_satellite.parquet", WORK / "satellite.parquet")
    shutil.copy2(ROOT / "data/interim/soil.parquet", WORK / "soil.parquet")

    build_unified_dataset(interim_dir=WORK, out_path=V5, crops=cfg["crops"], year_range=(2000, 2025))
    subprocess.run([sys.executable, str(ROOT / "scripts/29_merge_driver_features.py"), "--dataset", str(V5)], check=True)
    subprocess.run([sys.executable, str(ROOT / "scripts/33_merge_prereg_features.py"), "--dataset", str(V5)]
                   + ([] if args.modis_start_stamp else ["--modis-end-stamp"]), check=True)

    out = pd.read_parquet(V5)
    ids = (pd.read_parquet(V4, columns=["landkreis", "landkreis_id"]).drop_duplicates()
             .set_index("landkreis")["landkreis_id"])
    extra = sorted(set(out["landkreis"]) - set(ids.index))
    idmap = {**ids.to_dict(), **{k: int(ids.max()) + 1 + i for i, k in enumerate(extra)}}
    out["landkreis_id"] = out["landkreis"].map(idmap).astype("int32")
    out.to_parquet(V5, index=False)
    yy = out.dropna(subset=["yield_dt_per_ha"]).drop_duplicates(["landkreis", "crop", "year"])
    print(f"wrote {V5.name}: {len(out):,} rows; district-crop-years with yield {len(yy):,}, districts {yy.landkreis.nunique()}")
    print(yy.groupby("year").landkreis.nunique().to_string())

    # sanity: ERA5-Land vs the DWD station series where both exist (v4)
    v4 = pd.read_parquet(V4, columns=["landkreis", "date", "crop", "temperature_air_max_2m", "precipitation_height"],
                         filters=[("crop", "==", "oats")]).drop_duplicates(["landkreis", "date"])
    v5 = out[out.crop == "oats"][["landkreis", "date", "temperature_air_max_2m", "precipitation_height"]].drop_duplicates(["landkreis", "date"])
    for d in (v4, v5):
        d["date"] = pd.to_datetime(d["date"]).dt.tz_localize(None)
    m = v4.merge(v5, on=["landkreis", "date"], suffixes=("_dwd", "_era5")).dropna()
    print(f"ERA5 vs DWD where both exist ({len(m):,} district-days): tmax r {m.temperature_air_max_2m_dwd.corr(m.temperature_air_max_2m_era5):.3f}, "
          f"bias {(m.temperature_air_max_2m_era5 - m.temperature_air_max_2m_dwd).mean():+.2f} C; precip r {m.precipitation_height_dwd.corr(m.precipitation_height_era5):.3f}, "
          f"mean {m.precipitation_height_dwd.mean():.2f} vs {m.precipitation_height_era5.mean():.2f} mm/d")


if __name__ == "__main__":
    main()

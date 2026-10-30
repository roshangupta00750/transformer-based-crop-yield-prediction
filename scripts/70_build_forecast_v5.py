"""Build the input rows of an unlabelled forecast year in the dataset_v5_hybridW format
(pre-registration v16 §4.1), without touching the training dataset.

Same code as scripts/55 --weather hybrid, on the short window [Y-2 Jul .. data end]:
    weather   ERA5-Land over cropland (data/raw/prereg_v1/era5w_<Y-2..Y>.csv); DWD station values
              where present: the cleaned v4 series (data/interim/v4/weather.parquet) up to Y-1,
              data/raw/forecast/<Y>/dwd_daily.parquet cleaned the same way for year Y
    MODIS     data/raw/gee/_by_year/sat_<Y-2>, sat_<Y-1> + data/raw/forecast/<Y>/sat_<Y>.parquet,
              end-stamped (scripts/33 --modis-end-stamp)
    yields    data/interim/v5/yield.parquet (code-matched); Y-1 built unlabelled, yields joined back
    drivers   scripts/29 and scripts/33 on the build
Writes data/processed/forecast_<Y>_v5.parquet with dataset_v5_hybridW's columns and landkreis_id
encoding. Refused (exit 1) if the Y-1 rows differ from dataset_v5_hybridW in a model input
(reproduction check of scripts/35, against the v5 training dataset).

Usage:  python scripts/70_build_forecast_v5.py --year 2026
"""
from __future__ import annotations

import argparse
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.data.acquisition import dwd_weather, gee_satellite  # noqa: E402
from src.data.dataset import build_unified_dataset  # noqa: E402
from src.data.landkreise import load_station_mapping  # noqa: E402
from src.utils import load_data_config  # noqa: E402


def _load(name: str, file: str):
    s = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return m


x55 = _load("x55", "55_build_dataset_v5.py")
x35 = _load("x35", "35_build_forecast_year.py")
TRAIN = ROOT / "data/processed/dataset_v5_hybridW.parquet"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--year", type=int, required=True)
    args = ap.parse_args()
    y = args.year
    cfg = load_data_config()
    raw = ROOT / f"data/raw/forecast/{y}"
    work = ROOT / f"data/interim/forecast_{y}_v5"
    out_path = ROOT / f"data/processed/forecast_{y}_v5.parquet"
    work.mkdir(parents=True, exist_ok=True)
    start = pd.Timestamp(f"{y - 2}-07-01", tz="UTC")

    # weather: ERA5-Land over cropland, DWD stations where present (scripts/55 hybrid_weather)
    files = [ROOT / f"data/raw/prereg_v1/era5w_{t}.csv" for t in (y - 2, y - 1, y)]
    missing = [f.name for f in files if not f.exists()]
    if missing:
        sys.exit(f"missing ERA5-Land exports: {missing}")
    orig = x55.glob.glob
    x55.glob.glob = lambda _p: [str(f) for f in files]           # era5_weather() reads exactly these
    try:
        era = x55.era5_weather()
    finally:
        x55.glob.glob = orig
    era = era[era["date"] >= start]
    dwd_y = work / "dwd_weather_y.parquet"
    dwd_weather.clean(raw / "dwd_daily.parquet", dwd_y,
                      station_to_landkreis=load_station_mapping(ROOT / "data/interim/station_to_landkreis.pkl"))
    old = pd.read_parquet(x55.DWD_V4)
    new = pd.read_parquet(dwd_y)
    for d in (old, new):
        d["date"] = pd.to_datetime(d["date"], utc=True)
    cut = pd.Timestamp(f"{y}-01-01", tz="UTC")
    dwd = pd.concat([old[(old["date"] >= start) & (old["date"] < cut)], new[new["date"] >= cut]], ignore_index=True)
    dwd_path = work / "dwd_v4_plus_y.parquet"
    dwd.to_parquet(dwd_path, index=False)
    x55.DWD_V4 = dwd_path
    x55.hybrid_weather(era).to_parquet(work / "weather.parquet", index=False)

    # satellite
    parts = [pd.read_parquet(ROOT / f"data/raw/gee/_by_year/sat_{t}.parquet") for t in (y - 2, y - 1)]
    parts.append(pd.read_parquet(raw / f"sat_{y}.parquet"))
    (pd.concat(parts, ignore_index=True).sort_values(["landkreis", "date"])
       .to_parquet(work / "gee_satellite.parquet", index=False))
    gee_satellite.clean(work / "gee_satellite.parquet", work / "satellite.parquet")
    shutil.copy2(ROOT / "data/interim/v5/soil.parquet", work / "soil.parquet")

    # yields: Y-2 labelled; Y-1 and Y built unlabelled, Y-1 joined back (scripts/35)
    yld = pd.read_parquet(ROOT / "data/interim/v5/yield.parquet")
    yld[yld["year"] == y - 2].to_parquet(work / "yield.parquet", index=False)
    build_unified_dataset(interim_dir=work, out_path=out_path, crops=cfg["crops"], year_range=(y - 2, y))
    b = pd.read_parquet(out_path)
    prev = (yld[yld["year"] == y - 1][["landkreis", "year", "crop", "yield_dt_per_ha"]]
              .rename(columns={"yield_dt_per_ha": "_y"}))
    b = b.merge(prev, on=["landkreis", "year", "crop"], how="left", validate="many_to_one")
    b["yield_dt_per_ha"] = b["yield_dt_per_ha"].fillna(b.pop("_y"))
    b.to_parquet(out_path, index=False)
    subprocess.run([sys.executable, str(ROOT / "scripts/29_merge_driver_features.py"), "--dataset", str(out_path)], check=True)
    subprocess.run([sys.executable, str(ROOT / "scripts/33_merge_prereg_features.py"), "--dataset", str(out_path),
                    "--modis-end-stamp"], check=True)

    out = pd.read_parquet(out_path)
    ids = (pd.read_parquet(TRAIN, columns=["landkreis", "landkreis_id"]).drop_duplicates()
             .set_index("landkreis")["landkreis_id"])
    unknown = sorted(set(out["landkreis"]) - set(ids.index))
    if unknown:
        print(f"dropping {len(unknown)} districts absent from training: {unknown[:6]}...")
        out = out[out["landkreis"].isin(ids.index)]
    out["landkreis_id"] = out["landkreis"].map(ids).astype("int32")
    cols = pd.read_parquet(TRAIN, filters=[("year", "==", 2003)]).columns.tolist()
    lacking = [c for c in cols if c not in out.columns]
    if lacking:
        sys.exit(f"forecast build lacks training columns: {lacking}")
    out = out[cols]
    out.to_parquet(out_path, index=False)

    x35.MAIN = TRAIN
    x35.verify(out, y - 1)
    f = out[out["year"] == y]
    cov = f.groupby(f["date"].dt.to_period("M"))[
        ["temperature_air_max_2m", "swvl1", "ndvi_crop", "lai_crop", "soiltemp"]].apply(lambda g: g.notna().mean()).round(2)
    print(f"\n{y}: {f['landkreis'].nunique()} districts x {f['crop'].nunique()} crops, "
          f"{f['date'].min().date()} -> {f['date'].max().date()}\ncoverage by month:\n{cov}")
    print(f"wrote {out_path} ({len(out):,} rows)")


if __name__ == "__main__":
    main()

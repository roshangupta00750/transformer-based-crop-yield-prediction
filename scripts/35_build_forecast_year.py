"""Build the input rows for an unlabelled forecast year, without touching the
pre-registered training dataset.

Runs the SAME code as the historical build (scripts/01, 02, 29, 33) on a short
window [Y-2 Jul .. today] and writes data/processed/forecast_<Y>.parquet:

    Y-2 (Jul-Dec) + Y-1   labelled rows; Y-1 autumn feeds Y's winter crop-year,
                          and Y-1 is re-built only to check reproduction
    Y                     every district x crop with yield = NaN

Reproduction check: the Y-1 rows of the new build are compared column by
column with data/processed/dataset.parquet. The build is refused (exit 1) if
a model input differs beyond tolerance, so the forecast sees inputs built
exactly as the training data was. DWD may revise recent station values, so
small weather differences are reported, not hidden.

landkreis_id is re-mapped to the training dataset's encoding (the builder
derives it from whichever districts are present, which differ in a short build).

Usage:
    python scripts/35_build_forecast_year.py --year 2026
    python scripts/35_build_forecast_year.py --year 2026 --skip-download
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.data.acquisition import dwd_weather, gee_satellite  # noqa: E402
from src.data.dataset import build_unified_dataset  # noqa: E402
from src.data.landkreise import load_station_mapping  # noqa: E402
from src.utils import load_data_config  # noqa: E402

MAIN = ROOT / "data/processed/dataset.parquet"
TOL = 1e-6          # derived inputs must match to float precision
WEATHER_TOL = 0.02  # share of weather cells DWD may have revised


def build_satellite(raw: Path, year: int) -> Path:
    """Cached MODIS years Y-2, Y-1 plus a fresh reduction of year Y."""
    new = raw / f"sat_{year}.parquet"
    if not new.exists():
        gee_satellite._init(None)
        fc = gee_satellite._build_landkreis_fc(ROOT / "data/raw/geo")
        gee_satellite._reduce_year(fc, year).to_parquet(new, index=False)
    parts = [pd.read_parquet(ROOT / f"data/raw/gee/_by_year/sat_{y}.parquet")
             for y in (year - 2, year - 1)] + [pd.read_parquet(new)]
    out = raw / "gee_satellite.parquet"
    (pd.concat(parts, ignore_index=True)
       .sort_values(["landkreis", "date"]).reset_index(drop=True)
       .to_parquet(out, index=False))
    return out


def verify(out: pd.DataFrame, year: int) -> None:
    cols = [c for c in out.columns if c not in ("landkreis", "date", "crop")]
    main = pd.read_parquet(MAIN, filters=[("year", "==", year)])
    main["date"] = pd.to_datetime(main["date"])
    new = out[out["year"] == year].copy()
    new["date"] = pd.to_datetime(new["date"])
    k = ["landkreis", "date", "crop"]
    m = main.merge(new, on=k, suffixes=("_m", "_n"), how="inner")
    print(f"\nreproduction check on {year}: main {len(main):,} rows, new {len(new):,} "
          f"(incl. district x crop pairs without a {year} report), matched {len(m):,}")
    if len(m) != len(main):
        sys.exit("reproduction check FAILED: training rows missing from the new build")
    weather = ("temperature_", "precip", "radiation", "humid", "cwb", "hotdry",
               "heavyrain", "barefrost")
    # The training build's MODIS download ended at the last composite of Y-1
    # (e.g. 27 Dec 2025), so its final days are "missing" there; this build
    # continues into Y and carries that composite forward correctly. Those
    # days (not part of any training sample) are exempt and reported.
    last_sat = pd.to_datetime(pd.read_parquet(
        ROOT / f"data/raw/gee/_by_year/sat_{year}.parquet", columns=["date"])["date"]).max()
    sat_cols = ("ndvi", "evi", "lai")
    edge = m["date"] > last_sat
    print(f"  satellite columns exempt after {last_sat.date()} "
          f"({edge.mean():.4%} of rows, end of the training build's MODIS download)")
    bad = []
    for c in cols:
        rows = ~edge if c.startswith(sat_cols) else slice(None)
        a, b = m.loc[rows, f"{c}_m"], m.loc[rows, f"{c}_n"]
        if not pd.api.types.is_numeric_dtype(a):
            differ = (a.astype(str) != b.astype(str)).mean()
        else:
            a, b = a.astype(float), b.astype(float)
            both_nan = a.isna() & b.isna()
            differ = (~both_nan & ~(np.abs(a - b) <= TOL)).mean()
        limit = WEATHER_TOL if c.startswith(weather) else 0.0
        flag = "" if differ <= limit else "  <-- DIFFERS"
        if differ > 0:
            print(f"  {c:32s} {differ:8.4%} of cells differ{flag}")
        if flag:
            bad.append(c)
    if bad:
        sys.exit(f"reproduction check FAILED for {bad}; forecast inputs not trusted")
    print("reproduction check passed")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--skip-download", action="store_true")
    args = ap.parse_args()
    y = args.year
    cfg = load_data_config()
    raw = ROOT / f"data/raw/forecast/{y}"
    interim = ROOT / f"data/interim/forecast_{y}"
    out_path = ROOT / f"data/processed/forecast_{y}.parquet"
    raw.mkdir(parents=True, exist_ok=True)
    interim.mkdir(parents=True, exist_ok=True)

    start, end = f"{y - 2}-07-01", date.today().isoformat()
    if not args.skip_download or not (raw / "dwd_daily.parquet").exists():
        dwd_weather.download(raw, start_date=start, end_date=end)
    dwd_weather.clean(raw / "dwd_daily.parquet", interim / "weather.parquet",
                      station_to_landkreis=load_station_mapping(
                          ROOT / "data/interim/station_to_landkreis.pkl"))
    w = pd.read_parquet(interim / "weather.parquet")
    d = pd.to_datetime(w["date"])
    d = d.dt.tz_localize(None) if d.dt.tz is not None else d
    w[(d >= start).to_numpy()].to_parquet(interim / "weather.parquet", index=False)

    gee_satellite.clean(build_satellite(raw, y), interim / "satellite.parquet")
    # Y-1 is built as an unlabelled year (every district x crop) and its
    # reported yields are joined back afterwards. A labelled build keeps Y-1
    # weather only where a yield was reported, so districts without a Y-1
    # report would lose their autumn sowing period for Y's winter crops.
    yld = pd.read_parquet(ROOT / "data/interim/yield.parquet")
    yld[yld["year"] == y - 2].to_parquet(interim / "yield.parquet", index=False)
    shutil.copy2(ROOT / "data/interim/soil.parquet", interim / "soil.parquet")

    build_unified_dataset(interim_dir=interim, out_path=out_path,
                          crops=cfg["crops"], year_range=(y - 2, y))
    b = pd.read_parquet(out_path)
    prev = (yld[yld["year"] == y - 1][["landkreis", "year", "crop", "yield_dt_per_ha"]]
              .rename(columns={"yield_dt_per_ha": "_y"}))
    b = b.merge(prev, on=["landkreis", "year", "crop"], how="left", validate="many_to_one")
    b["yield_dt_per_ha"] = b["yield_dt_per_ha"].fillna(b.pop("_y"))
    b.to_parquet(out_path, index=False)
    for script in ("29_merge_driver_features.py", "33_merge_prereg_features.py"):
        subprocess.run([sys.executable, str(ROOT / "scripts" / script),
                        "--dataset", str(out_path)], check=True)

    out = pd.read_parquet(out_path)
    ids = (pd.read_parquet(MAIN, columns=["landkreis", "landkreis_id"])
             .drop_duplicates().set_index("landkreis")["landkreis_id"])
    unknown = sorted(set(out["landkreis"]) - set(ids.index))
    if unknown:
        print(f"dropping {len(unknown)} districts absent from training: {unknown[:6]}...")
        out = out[out["landkreis"].isin(ids.index)]
    out["landkreis_id"] = out["landkreis"].map(ids).astype("int32")
    main_cols = pd.read_parquet(MAIN, filters=[("year", "==", 2003)]).columns.tolist()
    missing = [c for c in main_cols if c not in out.columns]
    if missing:
        sys.exit(f"forecast build lacks training columns: {missing}")
    out = out[main_cols]
    out.to_parquet(out_path, index=False)

    verify(out, y - 1)
    f = out[out["year"] == y]
    last = f.groupby(f["date"].dt.to_period("M"))[
        ["temperature_air_max_2m", "swvl1", "ndvi_crop", "lai_crop", "soiltemp"]
    ].apply(lambda g: g.notna().mean()).round(2)
    print(f"\n{y}: {f['landkreis'].nunique()} districts x {f['crop'].nunique()} crops, "
          f"{f['date'].min().date()} -> {f['date'].max().date()}\ncoverage by month:\n{last}")
    print(f"wrote {out_path} ({len(out):,} rows)")


if __name__ == "__main__":
    main()

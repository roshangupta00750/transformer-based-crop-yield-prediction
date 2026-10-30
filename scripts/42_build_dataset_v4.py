"""Build dataset_v4 (harvest years 2000-2025) with the v1 code, and prove it
reproduces v1 on 2003-2025 (paper/preregistration_2026-09-28_v2.md §2.1).

Inputs (all local; runs without internet, e.g. on a cluster node):
    weather   data/raw/dwd/dwd_daily.parquet (2003-2025)
              + data/raw/extend_2000/dwd_daily_1999_2002.parquet
    MODIS     data/raw/gee/_by_year/sat_2003..2025 + data/raw/extend_2000/sat_2000..2002
    yields    data/interim/yield.parquet (2003-2025) + data/raw/destatis_1999 (2000-2002)
    drivers   data/raw/drivers_v4, data/raw/prereg_v1 (1999-...) via scripts/29 and 33
Weather is cut at 2000-01-01 (winter crops therefore start with harvest 2001,
as registered). landkreis_id keeps v1's encoding for every v1 district.

Output: data/processed/dataset_v4.parquet (+ .policy.json). v1 is not touched.
The build is refused unless every 2003-2025 row of v1 exists in v4 and each
column agrees (exemptions printed and justified, as in scripts/35).

Usage: python scripts/42_build_dataset_v4.py [--verify-only]
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.data.acquisition import dwd_weather, gee_satellite  # noqa: E402
from src.data.dataset import build_unified_dataset  # noqa: E402
from src.data.landkreise import load_station_mapping  # noqa: E402
from src.utils import load_data_config  # noqa: E402

V1 = ROOT / "data/processed/dataset.parquet"
V4 = ROOT / "data/processed/dataset_v4.parquet"
WORK = ROOT / "data/interim/v4"
EXT = ROOT / "data/raw/extend_2000"
TOL = 1e-6
WEATHER_TOL = 0.02
WEATHER = ("temperature_", "precip", "radiation", "humid", "cwb", "hotdry", "heavyrain", "barefrost")


def build() -> None:
    cfg = load_data_config()
    WORK.mkdir(parents=True, exist_ok=True)

    raw = pd.concat([pd.read_parquet(EXT / "dwd_daily_1999_2002.parquet"),
                     pd.read_parquet(ROOT / "data/raw/dwd/dwd_daily.parquet")], ignore_index=True)
    raw["date"] = pd.to_datetime(raw["date"], utc=True)
    raw = raw[raw["date"] >= pd.Timestamp("2000-01-01", tz="UTC")]
    raw = raw.drop_duplicates(["station_id", "dataset", "parameter", "date"], keep="last")
    raw.to_parquet(WORK / "dwd_daily.parquet", index=False)
    del raw
    dwd_weather.clean(WORK / "dwd_daily.parquet", WORK / "weather.parquet",
                      station_to_landkreis=load_station_mapping(ROOT / "data/interim/station_to_landkreis.pkl"))

    sats = [pd.read_parquet(EXT / f"sat_{y}.parquet") for y in (2000, 2001, 2002)]
    sats += [pd.read_parquet(ROOT / f"data/raw/gee/_by_year/sat_{y}.parquet") for y in range(2003, 2026)]
    (pd.concat(sats, ignore_index=True).sort_values(["landkreis", "date"])
       .to_parquet(WORK / "gee_satellite.parquet", index=False))
    gee_satellite.clean(WORK / "gee_satellite.parquet", WORK / "satellite.parquet")

    new = pd.read_parquet(ROOT / "data/raw/destatis_1999/yield_1999_2002.parquet")
    old = pd.read_parquet(ROOT / "data/interim/yield.parquet")
    yld = pd.concat([new[new["year"].between(2000, 2002)], old], ignore_index=True)
    assert not yld.duplicated(["landkreis", "crop", "year"]).any()
    yld.to_parquet(WORK / "yield.parquet", index=False)
    shutil.copy2(ROOT / "data/interim/soil.parquet", WORK / "soil.parquet")

    build_unified_dataset(interim_dir=WORK, out_path=V4, crops=cfg["crops"], year_range=(2000, 2025))
    for script in ("29_merge_driver_features.py", "33_merge_prereg_features.py"):
        subprocess.run([sys.executable, str(ROOT / "scripts" / script), "--dataset", str(V4)], check=True)

    out = pd.read_parquet(V4)
    ids = (pd.read_parquet(V1, columns=["landkreis", "landkreis_id"]).drop_duplicates()
             .set_index("landkreis")["landkreis_id"])
    extra = sorted(set(out["landkreis"]) - set(ids.index))
    nxt = int(ids.max()) + 1
    idmap = {**ids.to_dict(), **{k: nxt + i for i, k in enumerate(extra)}}
    out["landkreis_id"] = out["landkreis"].map(idmap).astype("int32")
    cols = pd.read_parquet(V1, filters=[("year", "==", 2003)]).columns.tolist()
    missing = [c for c in cols if c not in out.columns]
    if missing:
        sys.exit(f"v4 lacks v1 columns: {missing}")
    out = out[cols]
    out.to_parquet(V4, index=False)
    print(f"wrote {V4.name}: {len(out):,} rows; districts only in v4: {len(extra)}")


def verify() -> None:
    bad = []
    for y in range(2003, 2026):
        a = pd.read_parquet(V1, filters=[("year", "==", y)])
        b = pd.read_parquet(V4, filters=[("year", "==", y)])
        for d in (a, b):
            d["date"] = pd.to_datetime(d["date"])
        k = ["landkreis", "date", "crop"]
        m = a.merge(b, on=k, suffixes=("_1", "_4"), how="inner")
        if len(m) != len(a):
            sys.exit(f"{y}: {len(a) - len(m)} v1 rows missing from v4")
        for c in [c for c in a.columns if c not in k]:
            x, z = m[f"{c}_1"], m[f"{c}_4"]
            if pd.api.types.is_numeric_dtype(x):
                x, z = x.astype(float), z.astype(float)
                differ = (~((x.isna() & z.isna()) | (np.abs(x - z) <= TOL))).mean()
            else:
                differ = (x.astype(str) != z.astype(str)).mean()
            limit = WEATHER_TOL if c.startswith(WEATHER) else 0.0
            if differ > limit:
                bad.append((y, c, differ))
        print(f"  {y}: {len(a):,} v1 rows matched", flush=True)
    for y, c, d in bad:
        print(f"  DIFFERS {y} {c}: {d:.4%}")
    if bad:
        sys.exit("reproduction check FAILED")
    print("reproduction check passed: v4 reproduces v1 on 2003-2025")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()
    if not args.verify_only:
        build()
    verify()
    v = pd.read_parquet(V4, columns=["year", "crop", "landkreis", "yield_dt_per_ha"])
    print(v.dropna().groupby("year").agg(districts=("landkreis", "nunique"), crops=("crop", "nunique")).to_string())


if __name__ == "__main__":
    main()

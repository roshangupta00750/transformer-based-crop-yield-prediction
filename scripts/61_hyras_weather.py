"""DWD HYRAS-DE v6-1 (1 km, daily) -> district means over cropland (preregistration v14 §2).

Weight of a 1 km cell for a district = (share of the cell inside the district, from a
100 m rasterisation of the NUTS-3 2021 polygons) x (share of the cell that is CORINE
2018 class 211, data/raw/hyras/cropland_frac_hyras1km.tif from Earth Engine).
Districts with no cropland weight fall back to the plain area mean. Each yearly
NetCDF is downloaded, aggregated and deleted (unless --keep).

Output: data/interim/hyras/hyras_<var>_<year>.parquet (landkreis, date, value) and
        data/interim/hyras/hyras_weather.parquet (landkreis, date, tmax, tmin, pr, hurs)

Usage:  python scripts/61_hyras_weather.py [--years 2000-2025] [--vars tasmax,tasmin,pr,hurs] [--keep]
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import Affine
from scipy import sparse

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data/raw/hyras"
OUT = ROOT / "data/interim/hyras"
URL = "https://opendata.dwd.de/climate_environment/CDC/grids_germany/daily/hyras_de"
DIRS = {"tasmax": "air_temperature_max", "tasmin": "air_temperature_min",
        "pr": "precipitation", "hurs": "humidity"}
NX, NY, X0, Y1, RES = 665, 890, 4021000.0, 3564000.0, 1000.0   # HYRAS 1 km grid, EPSG:3035
SUB = 10                                                          # 100 m sub-grid for district shares


def weights() -> tuple[list[str], sparse.csr_matrix]:
    """(district codes, W) with W[d, cell] = district share of cell x cropland share; rows sum to 1."""
    g = gpd.read_file(ROOT / "data/raw/geo/NUTS_RG_01M_2021_4326_LEVL_3.shp")
    g = g[g["CNTR_CODE"] == "DE"].to_crs("EPSG:3035").sort_values("NUTS_ID").reset_index(drop=True)
    fine = rasterize(((geom, i + 1) for i, geom in enumerate(g.geometry)), out_shape=(NY * SUB, NX * SUB),
                     transform=Affine(RES / SUB, 0, X0, 0, -RES / SUB, Y1), fill=0, dtype="uint16")
    # count 100 m sub-cells per (district, 1 km cell)
    rr, cc = np.nonzero(fine)
    cell = (rr // SUB) * NX + (cc // SUB)
    A = sparse.coo_matrix((np.ones(len(rr), np.float32), (fine[rr, cc].astype(np.int64) - 1, cell)),
                          shape=(len(g), NX * NY)).tocsr() / SUB ** 2
    with rasterio.open(RAW / "cropland_frac_hyras1km.tif") as s:
        assert s.width == NX and s.height == NY and abs(s.transform.c - X0) < 1 and abs(s.transform.f - Y1) < 1
        crop = np.nan_to_num(s.read(1).astype(np.float32).ravel(), nan=0.0)
    W = A.multiply(crop[None, :]).tocsr()
    empty = np.asarray(W.sum(1)).ravel() <= 0
    if empty.any():
        print(f"  {empty.sum()} districts without cropland weight -> area mean: {list(g.NUTS_ID[empty])}")
        W = sparse.vstack([A[i] if e else W[i] for i, e in enumerate(empty)]).tocsr()
    print(f"weights: {len(g)} districts, {W.nnz:,} district-cell pairs")
    return list(g["NUTS_ID"]), W


def aggregate(var: str, year: int, codes: list[str], W: sparse.csr_matrix, keep: bool) -> pd.DataFrame:
    import xarray as xr
    out = OUT / f"hyras_{var}_{year}.parquet"
    if out.exists():
        return pd.read_parquet(out)
    fn = RAW / f"{var}_hyras_1_{year}_v6-1_de.nc"
    if not fn.exists():
        subprocess.run(["curl", "-sf", "-o", str(fn), f"{URL}/{DIRS[var]}/{fn.name}"], check=True)
    with xr.open_dataset(fn) as d:
        assert d.sizes["x"] == NX and d.sizes["y"] == NY and float(d.x_bnds[0, 0]) == X0
        a = d[var].values.astype(np.float32)                      # (time, y, x)
        if d.y.values[0] < d.y.values[-1]:                        # file rows run south->north
            a = a[:, ::-1, :]
        dates = pd.to_datetime(d.time.values).normalize()
    a = a.reshape(len(dates), -1).T                               # (cells, time), row-major from the north-west
    ok = np.isfinite(a)
    num = W @ np.where(ok, a, 0.0)
    den = W @ ok.astype(np.float32)
    v = np.where(den > 0, num / np.maximum(den, 1e-9), np.nan)
    df = pd.DataFrame({"landkreis": np.repeat(codes, len(dates)), "date": np.tile(dates, len(codes)),
                       var: v.ravel().astype(np.float32)})
    df.to_parquet(out, index=False)
    if not keep:
        fn.unlink()
    print(f"  {var} {year}: {len(dates)} days, missing {np.isnan(v).mean():.2%}", flush=True)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--years", default="2000-2025")
    ap.add_argument("--vars", default="tasmax,tasmin,pr,hurs")
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args()
    y0, y1 = map(int, a.years.split("-"))
    RAW.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
    codes, W = weights()
    frames = {v: pd.concat([aggregate(v, y, codes, W, a.keep) for y in range(y0, y1 + 1)], ignore_index=True)
              for v in a.vars.split(",")}
    m = None
    for v, f in frames.items():
        m = f if m is None else m.merge(f, on=["landkreis", "date"], how="outer")
    m = m.rename(columns={"tasmax": "tmax", "tasmin": "tmin"})
    m.to_parquet(OUT / "hyras_weather.parquet", index=False)
    print(f"hyras_weather.parquet: {len(m):,} district-days, {m.landkreis.nunique()} districts, "
          f"{m.date.min().date()}..{m.date.max().date()}")
    print(m.drop(columns=["landkreis", "date"]).describe().round(2).to_string())


if __name__ == "__main__":
    main()

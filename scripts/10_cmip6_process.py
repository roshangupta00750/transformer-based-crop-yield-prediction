"""Process the CMIP6 NetCDF zips into a per-NUTS-3-district monthly table.

Reads all 15 zips from ``data/raw/cmip6/`` (3 experiments × 5 variables,
EC-Earth3-CC monthly output). For each district polygon in
``data/raw/geo/NUTS_RG_01M_2021_4326_LEVL_3.shp`` (~401 German districts),
uses ``xarray.Dataset.interp()`` to sample the coarse CMIP6 grid at the
district centroid — bilinear interpolation over the four surrounding
grid cells. Also converts units:

* tas / tasmax / tasmin: Kelvin -> Celsius
* pr:  kg / m² / s  -> mm / month
* rsds: W / m² (kept as-is)

Output ``data/processed/cmip6_nuts3.parquet`` schema::

    landkreis (NUTS-3 code) : str
    year                     : int
    month                    : int
    experiment               : str  (historical / ssp2_4_5 / ssp5_8_5)
    variable                 : str  (tas / tasmax / tasmin / pr / rsds)
    value                    : float

Idempotent: rerunning is safe (writes a single output parquet).

Usage:
    python scripts/10_cmip6_process.py
    python scripts/10_cmip6_process.py --experiment ssp5_8_5
"""
from __future__ import annotations

import argparse
import os
import sys
import zipfile
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw" / "cmip6"
OUT_PATH = ROOT / "data" / "processed" / "cmip6_nuts3.parquet"
SHAPE_PATH = ROOT / "data" / "raw" / "geo" / "NUTS_RG_01M_2021_4326_LEVL_3.shp"

VARIABLES = {
    "near_surface_air_temperature": "tas",
    "daily_maximum_near_surface_air_temperature": "tasmax",
    "daily_minimum_near_surface_air_temperature": "tasmin",
    "precipitation": "pr",
    "surface_downwelling_shortwave_radiation": "rsds",
}
EXPERIMENTS = ["historical", "ssp2_4_5", "ssp5_8_5"]

TEMP_VARS = {"tas", "tasmax", "tasmin"}


def load_district_centroids() -> pd.DataFrame:
    """Return one row per German NUTS-3 district with lat/lon centroids."""
    shp = gpd.read_file(SHAPE_PATH)
    de = shp[shp["CNTR_CODE"] == "DE"].copy()
    # Project to a metric CRS for accurate centroid, then back to lat/lon
    de_metric = de.to_crs(epsg=3035)
    de["centroid"] = de_metric.centroid.to_crs(epsg=4326)
    out = pd.DataFrame({
        "landkreis": de["NUTS_ID"].values,
        "lat": [c.y for c in de["centroid"]],
        "lon": [c.x for c in de["centroid"]],
    })
    return out


def unzip_and_open(zip_path: Path) -> xr.Dataset:
    """Extract the .nc file from a CDS zip and open it. Cached extractions
    live in ``data/raw/cmip6/_unpacked/`` — reused on subsequent runs."""
    unpack_dir = RAW_DIR / "_unpacked" / zip_path.stem
    unpack_dir.mkdir(parents=True, exist_ok=True)
    if not any(unpack_dir.glob("*.nc")):
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(unpack_dir)
    nc_files = list(unpack_dir.glob("*.nc"))
    if not nc_files:
        raise RuntimeError(f"no .nc in {zip_path}")
    return xr.open_dataset(nc_files[0])


def sample_at_districts(ds: xr.Dataset, var: str,
                        centroids: pd.DataFrame) -> pd.DataFrame:
    """Bilinear-interpolate CMIP6 grid at each district centroid.

    Returns a long DataFrame: (landkreis, year, month, value).
    """
    # CMIP6 longitudes are typically 0..360 — remap our -180..180 centroids
    lons = centroids["lon"].to_numpy() % 360
    lats = centroids["lat"].to_numpy()

    # xarray.interp does bilinear over the specified coords
    da = ds[var]
    sampled = da.interp(
        lon=xr.DataArray(lons, dims="district"),
        lat=xr.DataArray(lats, dims="district"),
        method="linear",
    )
    values = sampled.values          # shape (time, districts)
    times = pd.to_datetime(ds["time"].values)

    n_time = values.shape[0]
    n_dist = len(centroids)
    long_rows = pd.DataFrame({
        "landkreis": np.tile(centroids["landkreis"].values, n_time),
        "year":  np.repeat(times.year,  n_dist),
        "month": np.repeat(times.month, n_dist),
        "value": values.ravel(),
    })
    return long_rows


def convert_units(df: pd.DataFrame, variable: str) -> pd.DataFrame:
    """Apply variable-specific unit conversions in place."""
    if variable in TEMP_VARS:
        # Kelvin -> Celsius
        df["value"] = df["value"] - 273.15
    elif variable == "pr":
        # kg/m²/s == mm/s -> mm per month (approximate: days_in_month × 86400)
        days = df.apply(lambda r: pd.Timestamp(year=int(r["year"]),
                                               month=int(r["month"]), day=1)
                        .days_in_month, axis=1)
        df["value"] = df["value"] * days * 86400
    # rsds: leave in W/m²
    return df


DEFAULT_MODEL = "ec_earth3_cc"

# Filename convention: default-model files use the bare `<exp>_<var>.zip`
# form (legacy), while all other models use `<model>_<exp>_<var>.zip`.
def _zip_path_for(model: str, exp: str, var_cds: str) -> Path:
    if model == DEFAULT_MODEL:
        return RAW_DIR / f"{exp}_{var_cds}.zip"
    return RAW_DIR / f"{model}_{exp}_{var_cds}.zip"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiment", choices=EXPERIMENTS)
    ap.add_argument("--variable", choices=list(VARIABLES))
    ap.add_argument("--models", default="ec_earth3_cc,mpi_esm1_2_lr",
                    help="Comma-separated CMIP6 models to include. Each "
                         "model's files must already be pulled via "
                         "`scripts/09_cmip6_pull.py --model <name>`. Output "
                         "gets a `model` column so downstream code can "
                         "average across models.")
    args = ap.parse_args()

    if not SHAPE_PATH.exists():
        sys.exit(f"shapefile missing: {SHAPE_PATH}")

    print(f"Loading district centroids from {SHAPE_PATH.name}...")
    centroids = load_district_centroids()
    print(f"  {len(centroids)} German NUTS-3 districts")
    print(f"  lat range: {centroids['lat'].min():.2f} to {centroids['lat'].max():.2f}")
    print(f"  lon range: {centroids['lon'].min():.2f} to {centroids['lon'].max():.2f}")

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    exps = [args.experiment] if args.experiment else EXPERIMENTS
    vars_ = [args.variable] if args.variable else list(VARIABLES)
    print(f"\nModels to include: {models}")

    all_rows: list[pd.DataFrame] = []
    for model in models:
        print(f"\n=== {model} ===")
        for exp in exps:
            for var_cds, var_short in VARIABLES.items():
                if var_cds not in vars_:
                    continue
                zip_path = _zip_path_for(model, exp, var_cds)
                if not zip_path.exists():
                    print(f"  SKIP {model} / {exp} × {var_short}: {zip_path.name} missing")
                    continue
                print(f"  {model} / {exp} × {var_short}: opening {zip_path.name}")
                ds = unzip_and_open(zip_path)
                long_df = sample_at_districts(ds, var_short, centroids)
                long_df["experiment"] = exp
                long_df["variable"] = var_short
                long_df["model"] = model
                convert_units(long_df, var_short)
                all_rows.append(long_df)
            print(f"    -> {len(long_df)} rows")

    if not all_rows:
        print("no rows produced")
        return

    out = pd.concat(all_rows, ignore_index=True)
    out = out[["landkreis", "year", "month", "experiment", "variable",
               "model", "value"]]
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, index=False)
    print(f"\nWrote {OUT_PATH} ({len(out):,} rows, "
          f"{out['landkreis'].nunique()} districts × {out['year'].nunique()} years "
          f"× {out['month'].nunique()} months × {out['experiment'].nunique()} scenarios "
          f"× {out['variable'].nunique()} variables × {out['model'].nunique()} models)")

    print("\nSample of the output:")
    print(out.head(6).to_string(index=False))
    print("\nPer-model × per-experiment coverage:")
    print(out.groupby(["model", "experiment"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()

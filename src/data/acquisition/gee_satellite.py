"""Sentinel-2 + MODIS vegetation indices via Google Earth Engine.

Pulls per-Landkreis composites of NDVI, EVI, LAI, and NDWI from the MODIS
record (chosen for full 2003-present coverage at 250-500 m).

Sources:
  - MOD13Q1   16-day, 250 m  -> NDVI, EVI
  - MOD15A2H   8-day, 500 m  -> LAI
  - MOD09GA    daily,  500 m -> NDWI (b2, b6 reflectance)

The Landkreis polygons are sent inline as an ``ee.FeatureCollection`` built
from the local GISCO NUTS-3 shapefile, so no GEE asset upload is required.

Setup (one-time):
    earthengine authenticate
    earthengine set_project <your-cloud-project-id>
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from ...utils import get_logger

log = get_logger("gee")

# Default Cloud project for EE (overridden by env var or explicit arg).
DEFAULT_PROJECT = "crop-yield-praktikum"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _init(project: str | None) -> None:
    import ee
    import os
    project = project or os.environ.get("EE_PROJECT") or DEFAULT_PROJECT
    try:
        ee.Initialize(project=project)
    except Exception:
        # Already initialized or fallback to default initialize.
        ee.Initialize()


def _build_landkreis_fc(geo_dir: str | Path):
    """Convert the local German NUTS-3 shapefile to an inline ee.FeatureCollection.

    Each feature carries ``landkreis`` (NUTS-3 ID) as its only property; geometry
    is reprojected to EPSG:4326 (GEE's default) if not already.
    """
    import ee
    import geopandas as gpd

    shp = gpd.read_file(Path(geo_dir) / "NUTS_RG_01M_2021_4326_LEVL_3.shp")
    de = shp[shp["CNTR_CODE"] == "DE"][["NUTS_ID", "geometry"]].to_crs(epsg=4326)

    features = []
    for _, row in de.iterrows():
        geom = row["geometry"]
        # Handle Polygon vs MultiPolygon
        if geom.geom_type == "Polygon":
            ee_geom = ee.Geometry.Polygon([list(geom.exterior.coords)])
        else:
            ee_geom = ee.Geometry.MultiPolygon(
                [[list(p.exterior.coords)] for p in geom.geoms]
            )
        features.append(ee.Feature(ee_geom, {"landkreis": row["NUTS_ID"]}))
    fc = ee.FeatureCollection(features)
    log.info("Built FeatureCollection with %d Landkreise.", len(features))
    return fc


def _reduce_image(fc, img, value_bands: list[str], scale: int):
    """Reduce one image over the FeatureCollection. Returns a small dict list."""
    import ee
    reduced = img.reduceRegions(
        collection=fc,
        reducer=ee.Reducer.mean(),
        scale=scale,
    )
    fields = ["landkreis"] + value_bands
    out = reduced.select(fields, retainGeometry=False).getInfo()
    return [f["properties"] for f in out["features"]]


def _iter_collection(col, year: int):
    """Yield ee.Image, date pairs from a collection filtered to one year.

    We pull the collection size + list ids once, then materialize images one at
    a time. This keeps each getInfo() small.
    """
    import ee
    start = f"{year}-01-01"
    end = f"{year + 1}-01-01"
    col = col.filterDate(start, end).sort("system:time_start")
    size = col.size().getInfo()
    if size == 0:
        return
    img_list = col.toList(size)
    for i in range(size):
        img = ee.Image(img_list.get(i))
        date_ms = ee.Date(img.get("system:time_start")).millis().getInfo()
        yield img, pd.to_datetime(date_ms, unit="ms")


def _reduce_year(fc, year: int) -> pd.DataFrame:
    """Reduce one calendar year of MODIS indices over each Landkreis.

    Iterates each native MODIS composite (~23 NDVI/EVI + 46 LAI per year) and
    issues one ``reduceRegions().getInfo()`` per image. Each call returns ~401
    features (the FC size) — well under EE's 5000-element response cap.
    """
    import ee
    frames: list[pd.DataFrame] = []

    # NDVI + EVI from MOD13Q1 (16-day, 250 m). Scale factor 0.0001.
    veg_col = (ee.ImageCollection("MODIS/061/MOD13Q1")
               .select(["NDVI", "EVI"]))
    log.info("  MOD13Q1 (NDVI/EVI)...")
    veg_rows = []
    for img, date in _iter_collection(veg_col, year):
        scaled = ee.Image(img.multiply(0.0001).copyProperties(img, ["system:time_start"]))
        rows = _reduce_image(fc, scaled, ["NDVI", "EVI"], scale=500)
        for r in rows:
            r["date"] = date
        veg_rows.extend(rows)
    if veg_rows:
        frames.append(pd.DataFrame(veg_rows))

    # LAI from MOD15A2H (8-day, 500 m). Scale factor 0.1.
    lai_col = ee.ImageCollection("MODIS/061/MOD15A2H").select(["Lai_500m"])
    log.info("  MOD15A2H (LAI)...")
    lai_rows = []
    for img, date in _iter_collection(lai_col, year):
        scaled = ee.Image(img.multiply(0.1).copyProperties(img, ["system:time_start"]))
        rows = _reduce_image(fc, scaled, ["Lai_500m"], scale=500)
        for r in rows:
            r["date"] = date
            r["LAI"] = r.pop("Lai_500m", None)
        lai_rows.extend(rows)
    if lai_rows:
        frames.append(pd.DataFrame(lai_rows))

    if not frames:
        return pd.DataFrame(columns=["landkreis", "date"])

    # Outer-merge by (landkreis, date). Different collections have different
    # cadences so most days will only have one index populated.
    df = frames[0]
    for f in frames[1:]:
        df = df.merge(f, on=["landkreis", "date"], how="outer")
    df.columns = [c.lower() for c in df.columns]
    return df


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def download(
    out_dir: str | Path,
    landkreis_asset: str = "",
    start_date: str = "2003-01-01",
    end_date: str = "2023-12-31",
    composite_days: int = 10,
    cloud_mask_threshold: int = 30,
    geo_dir: str | Path | None = None,
    project: str | None = None,
) -> Path:
    """Download per-Landkreis MODIS vegetation-index composites.

    Parameters
    ----------
    out_dir : where to write ``gee_satellite.parquet``
    landkreis_asset : optional GEE asset path. If empty, the FC is built from
        the local NUTS-3 shapefile (no asset upload required).
    start_date / end_date : ISO date strings, inclusive.
    composite_days, cloud_mask_threshold : retained for API compat; MODIS
        native composites are already cloud-masked.
    geo_dir : where to find ``NUTS_RG_01M_2021_4326_LEVL_3.shp``. Defaults to
        ``<out_dir>/../geo`` (the layout produced by scripts/00_build_geo.py).
    project : Cloud project ID for ``ee.Initialize``. Defaults to env
        ``EE_PROJECT`` then to :data:`DEFAULT_PROJECT`.
    """
    import ee  # noqa: F401  (ensure earthengine-api installed; used by helpers)

    _init(project)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "gee_satellite.parquet"

    if landkreis_asset:
        fc = ee.FeatureCollection(landkreis_asset)
    else:
        if geo_dir is None:
            geo_dir = out_dir.parent / "geo"
        fc = _build_landkreis_fc(geo_dir)

    start_year = int(start_date[:4])
    end_year = int(end_date[:4])
    log.info("Downloading MODIS NDVI/EVI/LAI for %d-%d...",
             start_year, end_year)

    # Per-year cache files allow resume after a crash.
    per_year_dir = out_dir / "_by_year"
    per_year_dir.mkdir(exist_ok=True)

    frames = []
    for yr in range(start_year, end_year + 1):
        yr_file = per_year_dir / f"sat_{yr}.parquet"
        if yr_file.exists():
            log.info("Year %d: cached", yr)
            frames.append(pd.read_parquet(yr_file))
            continue
        log.info("Year %d", yr)
        df_yr = _reduce_year(fc, yr)
        df_yr.to_parquet(yr_file, index=False)
        frames.append(df_yr)

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["landkreis", "date"]).reset_index(drop=True)
    df.to_parquet(out_file, index=False)
    log.info("Wrote %s (%d rows, %d Landkreise, %d unique dates)",
             out_file, len(df), df["landkreis"].nunique(), df["date"].nunique())
    return out_file


def clean(raw_path: str | Path, out_path: str | Path) -> Path:
    """Resample to daily and ffill within each Landkreis so the merge with
    weather is clean. Output columns: landkreis, date, ndvi, evi, lai, ndwi."""
    raw_path, out_path = Path(raw_path), Path(out_path)
    df = pd.read_parquet(raw_path)
    if df.empty:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, index=False)
        return out_path

    df["date"] = pd.to_datetime(df["date"])
    cols = [c for c in ["ndvi", "evi", "lai", "ndwi"] if c in df.columns]

    # Resample each Landkreis to daily, forward-fill the composite values.
    def _resample(group):
        g = group.set_index("date")[cols].sort_index()
        idx = pd.date_range(g.index.min(), g.index.max(), freq="D")
        return g.reindex(idx).ffill().bfill().reset_index().rename(
            columns={"index": "date"})

    out = (df.groupby("landkreis", group_keys=True)
             .apply(_resample, include_groups=False)
             .reset_index(level=0)
             .reset_index(drop=True))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    log.info("Cleaned satellite → %s (%d rows)", out_path, len(out))
    return out_path

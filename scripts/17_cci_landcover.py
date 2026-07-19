"""Copernicus Climate Data Store — ESA CCI Land Cover pull + per-district
fractions. Replaces CORINE (which would require a separate CLMS account).

Steps:
  1. cdsapi.retrieve satellite-land-cover for one representative year (2020)
     — the CCI class scheme is stable, per-district land-use mix doesn't need
     annual resolution for our static-feature use.
  2. Reproject NUTS-3 vector to the CCI raster CRS (EPSG:4326).
  3. rasterstats.zonal_stats with categorical=True → per-district counts of
     each land-cover class present.
  4. Bucket the 22-class CCI legend into 7 fraction columns matching CORINE's
     conceptual grouping: arable, permanent_crops, pasture, forest, urban,
     water, other. Fractions sum to 1.0 per district.

Emits data/processed/static/landcover_features.parquet:
  landkreis, frac_arable, frac_permanent_crops, frac_pasture,
  frac_forest, frac_urban, frac_water, frac_other

CCI land cover classes (2015 legend, most-relevant subset):
  10-30    Cropland (rainfed / irrigated / mosaic) → arable
  40       Cropland-natural vegetation mosaic     → arable
  50-90    Tree cover (broadleaved/needleleaved…) → forest
  100      Mosaic tree/shrub                      → forest
  110-153  Shrubland/grassland/sparse veg          → pasture
  160-180  Flooded tree/shrub/herbaceous          → water
  190      Urban                                   → urban
  200-202  Bare areas                              → other
  210      Water bodies                            → water
  220      Permanent snow/ice                      → other
"""
from __future__ import annotations

import sys
import warnings
import zipfile
from pathlib import Path

warnings.filterwarnings("ignore")

import cdsapi
import numpy as np
import pandas as pd
import rasterio
from rasterstats import zonal_stats

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from src.data.landkreise import load_landkreise  # noqa: E402

RAW_DIR = REPO / "data" / "raw" / "landcover"
RAW_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR = REPO / "data" / "processed" / "static"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ZIP_FILE = RAW_DIR / "cci_lc_2020.zip"

# CCI legend → coarse buckets (per doc string above)
CCI_BUCKETS = {
    "arable":            {10, 11, 12, 20, 30, 40},
    "permanent_crops":   set(),                       # CCI doesn't split perennials from arable
    "pasture":           {110, 120, 121, 122, 130, 140, 150, 151, 152, 153},
    "forest":            {50, 60, 61, 62, 70, 71, 72, 80, 81, 82, 90, 100},
    "urban":             {190},
    "water":             {160, 170, 180, 210},
    "other":             {200, 201, 202, 220},
}


def step_1_download() -> Path:
    if ZIP_FILE.exists() and ZIP_FILE.stat().st_size > 100_000:
        print(f"  ✓ CCI zip already exists at {ZIP_FILE}")
        return ZIP_FILE
    print("  submitting CDS request for satellite-land-cover 2020...")
    c = cdsapi.Client()
    c.retrieve(
        "satellite-land-cover",
        {
            "variable": "all",
            "format": "zip",
            "year": ["2020"],
            "version": "v2_1_1",
            "type_of_record": "cdr",
        },
        str(ZIP_FILE),
    )
    print(f"  ✓ CDS zip retrieved: {ZIP_FILE.stat().st_size / 1e6:.1f} MB")
    return ZIP_FILE


def step_2_extract_tif(zip_path: Path) -> Path:
    """CCI comes as a .nc + companion files inside the zip; we want the raster."""
    extract_dir = zip_path.parent / "unpacked"
    extract_dir.mkdir(exist_ok=True)
    if not any(extract_dir.iterdir()):
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(extract_dir)
        print(f"  ✓ extracted to {extract_dir}")
    # Prefer .nc; if a .tif is present use it directly
    nc_files = list(extract_dir.rglob("*.nc"))
    tif_files = list(extract_dir.rglob("*.tif"))
    if tif_files:
        return tif_files[0]
    if not nc_files:
        raise FileNotFoundError(f"No .nc or .tif inside {zip_path}")
    return nc_files[0]


def step_3_zonal(raster: Path) -> pd.DataFrame:
    print(f"  raster: {raster.name}")
    with rasterio.open(raster) as src:
        print(f"    crs:   {src.crs}")
        print(f"    shape: {src.shape}")
    de = load_landkreise().to_crs(src.crs.to_epsg() or 4326)
    print(f"  {len(de)} districts, running zonal_stats(categorical=True)...")

    # Optional: crop the raster to Germany bbox first to speed up massively.
    # rasterstats does that per-polygon internally, but for a 22-class global
    # raster it can be slow. We pass all_touched=False (per pixel-center).
    stats = zonal_stats(
        de[["NUTS_ID", "geometry"]],
        str(raster),
        categorical=True,
        all_touched=False,
        nodata=0,
    )
    rows = []
    for i, (nuts_id, s) in enumerate(zip(de["NUTS_ID"], stats)):
        s = s or {}
        total = sum(v for v in s.values() if v > 0)
        if total == 0:
            rows.append({"landkreis": nuts_id, **{f"frac_{b}": np.nan for b in CCI_BUCKETS}})
            continue
        row = {"landkreis": nuts_id}
        for bucket, cci_codes in CCI_BUCKETS.items():
            hits = sum(s.get(c, 0) for c in cci_codes)
            row[f"frac_{bucket}"] = hits / total
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    zip_path = step_1_download()
    raster = step_2_extract_tif(zip_path)
    df = step_3_zonal(raster)

    out_file = OUT_DIR / "landcover_features.parquet"
    df.to_parquet(out_file, index=False)
    print()
    print(f"  wrote {out_file}")
    print(f"  {len(df)} districts, {len(df.columns)-1} fraction columns")
    print()

    print("  sanity — fractions sum to ~1.0 per district:")
    frac_cols = [c for c in df.columns if c.startswith("frac_")]
    df["_sum"] = df[frac_cols].sum(axis=1)
    print(f"    _sum stats: min={df['_sum'].min():.3f}, "
          f"mean={df['_sum'].mean():.3f}, max={df['_sum'].max():.3f}")

    print("  sample rows (first 5):")
    print(df.drop(columns=["_sum"]).head(5).to_string(index=False))
    print()
    print("  per-column non-null:")
    for c in df.columns:
        n = df[c].notna().sum()
        print(f"    {c:25s} {n}/{len(df)}")


if __name__ == "__main__":
    main()

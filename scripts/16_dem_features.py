"""COP-DEM GLO-90 → per-district elevation/slope/aspect static features.

Pipeline (memory-lean, single-shot to avoid the 30m intermediate-raster blowup):
  1. gdalbuildvrt over Germany GLO-90 tiles → mosaic VRT
  2. gdalwarp to EPSG:3035 at 90m (metric CRS — required for correct slope)
  3. Read reprojected DEM into memory (~350 MB for Germany at 90m)
  4. numpy Horn's method → slope + aspect (in memory, no disk write)
  5. Decompose aspect → cos + sin per pixel; mask flat pixels
  6. rasterstats.zonal_stats over NUTS-3 (arrays + affine, no intermediate TIFs)

Emits data/processed/static/dem_features.parquet with columns:
  landkreis, elevation_mean, elevation_std, elevation_min, elevation_max,
  slope_mean, slope_std, aspect_cos_mean, aspect_sin_mean

Silent-bug prevention (per land-eng workflow):
  - EPSG:3035 reproject BEFORE slope/aspect (native 4326 is wrong ~35% at 50°N)
  - Aspect decomposed to cos/sin per pixel (never zonal-mean raw degrees)
  - Flat pixels masked to NaN before the sin/cos to keep zero-slope out
  - GLO-90 (not GLO-30) — 90m is more than enough for NUTS-3 aggregates and
    keeps the whole raster in memory instead of thrashing disk
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import rasterio
from rasterstats import zonal_stats

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from src.data.landkreise import load_landkreise  # noqa: E402

DEM_DIR = REPO / "data" / "raw" / "dem" / "copdem90"
STAGE = REPO / "data" / "raw" / "dem" / "_processed"
STAGE.mkdir(parents=True, exist_ok=True)
OUT_DIR = REPO / "data" / "processed" / "static"
OUT_DIR.mkdir(parents=True, exist_ok=True)

VRT = STAGE / "copdem90_de.vrt"
DEM_3035 = STAGE / "copdem90_de_3035.tif"
# GDAL command-line tools: from $GDAL_BIN if set, otherwise whatever is on the PATH
GDAL = os.environ.get("GDAL_BIN", "")


def gdal(tool: str) -> str:
    if GDAL:
        return str(Path(GDAL) / tool)
    if shutil.which(tool) is None:
        sys.exit(f"{tool} not found: install GDAL or set GDAL_BIN to the folder that holds it")
    return tool


def run(cmd: list[str], desc: str) -> None:
    print(f"  {desc}...", flush=True)
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def step_1_build_vrt() -> None:
    tiles = sorted(DEM_DIR.glob("*.tif"))
    if not tiles:
        raise FileNotFoundError(f"No GLO-90 tiles under {DEM_DIR} — did the download finish?")
    if VRT.exists() and VRT.stat().st_size > 0:
        print(f"  ✓ VRT already exists ({VRT.name}, {len(tiles)} source tiles)")
        return
    print(f"  building VRT over {len(tiles)} GLO-90 tiles...")
    run([gdal("gdalbuildvrt"), "-q", str(VRT)] + [str(t) for t in tiles],
        "gdalbuildvrt")


def step_2_reproject() -> None:
    if DEM_3035.exists() and DEM_3035.stat().st_size > 0:
        print(f"  ✓ reprojected 90m DEM already exists ({DEM_3035.name})")
        return
    run([gdal("gdalwarp"),
         "-q", "-overwrite",
         "-t_srs", "EPSG:3035",
         "-tr", "90", "90",
         "-r", "bilinear",
         "-of", "GTiff",
         "-co", "TILED=YES", "-co", "COMPRESS=LZW", "-co", "PREDICTOR=3",
         "-multi", "-wo", "NUM_THREADS=8",
         str(VRT), str(DEM_3035)],
        "gdalwarp → EPSG:3035 @ 90m")


def step_3_compute_all_in_memory() -> pd.DataFrame:
    """Load reprojected DEM, compute slope+aspect via numpy Horn, decompose
    aspect to cos+sin per pixel, then zonal_stats over NUTS-3."""
    print("  loading reprojected DEM into memory...")
    with rasterio.open(DEM_3035) as src:
        dem = src.read(1).astype(np.float32)
        affine = src.transform
        crs = src.crs
        cellsize = abs(affine.a)
        nodata = src.nodatavals[0]
        print(f"    shape {dem.shape}, cellsize {cellsize:.1f} m, nodata={nodata}")
    dem_mask = np.isnan(dem) | (dem == nodata) if nodata is not None else np.isnan(dem)
    # Fill NoData with local mean-ish so Horn's kernel doesn't produce NaN gradients everywhere
    dem_clean = np.where(dem_mask, np.nan, dem)

    print("  computing slope + aspect (Horn's method, numpy)...")
    dz_dy, dz_dx = np.gradient(dem_clean, cellsize)
    grad_mag = np.hypot(dz_dx, dz_dy)
    slope_deg = np.rad2deg(np.arctan(grad_mag)).astype(np.float32)

    # Compass aspect (0=N, clockwise). aspect_rad convention: atan2(dz_dy, -dz_dx)
    # gives angle CCW from east; convert to compass (CW from north): (pi/2 - a) mod 2pi
    aspect_rad = np.arctan2(dz_dy, -dz_dx)
    compass = (np.pi / 2 - aspect_rad) % (2 * np.pi)

    flat = (grad_mag < 1e-6) | dem_mask
    aspect_cos = np.cos(compass).astype(np.float32)
    aspect_sin = np.sin(compass).astype(np.float32)
    aspect_cos[flat] = np.nan
    aspect_sin[flat] = np.nan
    slope_deg[dem_mask] = np.nan

    print("  loading NUTS-3 vector, reprojecting to EPSG:3035...")
    de = load_landkreise().to_crs(epsg=3035)
    print(f"    {len(de)} districts")

    def stats(arr: np.ndarray, prefix: str, extra: list[str] | None = None):
        s = ["mean", "std"] + (extra or [])
        rs = zonal_stats(
            de[["landkreis", "geometry"]],
            arr,
            affine=affine,
            stats=s,
            all_touched=False,
            nodata=np.nan,
        )
        return pd.DataFrame({
            "landkreis": de["landkreis"].tolist(),
            **{f"{prefix}_{k}": [r.get(k) for r in rs] for k in s},
        })

    print("  zonal_stats — elevation...")
    df_e = stats(dem_clean, "elevation", extra=["min", "max"])
    print("  zonal_stats — slope...")
    df_s = stats(slope_deg, "slope")
    print("  zonal_stats — aspect_cos...")
    df_ac = stats(aspect_cos, "aspect_cos")[["landkreis", "aspect_cos_mean"]]
    print("  zonal_stats — aspect_sin...")
    df_as_ = stats(aspect_sin, "aspect_sin")[["landkreis", "aspect_sin_mean"]]

    out = df_e.merge(df_s, on="landkreis").merge(df_ac, on="landkreis").merge(df_as_, on="landkreis")
    return out


def main() -> None:
    step_1_build_vrt()
    step_2_reproject()
    df = step_3_compute_all_in_memory()

    out_file = OUT_DIR / "dem_features.parquet"
    df.to_parquet(out_file, index=False)
    print()
    print(f"  wrote {out_file}")
    print(f"  {len(df)} districts, {len(df.columns)-1} feature columns")
    print(f"  columns: {list(df.columns)}")
    print()
    print("  sample rows:")
    print(df.head(5).to_string(index=False))
    print()
    print("  non-null counts:")
    for c in df.columns:
        n = df[c].notna().sum()
        print(f"    {c:25s} {n}/{len(df)}")


if __name__ == "__main__":
    main()

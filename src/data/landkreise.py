"""German NUTS-3 (Landkreis) geography utilities.

Provides:
  - `load_landkreise()`             — GeoDataFrame of all German Kreise.
  - `landkreis_centroids()`         — DataFrame[landkreis, lon, lat].
  - `build_station_to_landkreis()`  — spatial-join DWD stations → Landkreis,
                                      returns a dict {station_id: nuts3_code}.
  - `save_station_mapping()` / `load_station_mapping()` — pickle helpers.

The NUTS-3 polygons come from Eurostat's public GISCO archive. The first
call downloads ~5 MB to `data/raw/geo/` and caches it.
"""
from __future__ import annotations

import io
import pickle
import zipfile
from pathlib import Path

import pandas as pd
import requests

from ..utils import PROJECT_ROOT, get_logger

log = get_logger("landkreise")

# Eurostat GISCO NUTS-3 polygons (1:1M, 2021 vintage) — small, free, public.
GISCO_URL = (
    "https://gisco-services.ec.europa.eu/distribution/v2/nuts/"
    "download/ref-nuts-2021-01m.shp.zip"
)

DEFAULT_GEO_DIR = PROJECT_ROOT / "data" / "raw" / "geo"


# --------------------------------------------------------------------------- #
# Loading polygons
# --------------------------------------------------------------------------- #
def _ensure_gisco(geo_dir: Path = DEFAULT_GEO_DIR) -> Path:
    """Download + unzip the GISCO NUTS-3 shapefile once."""
    geo_dir.mkdir(parents=True, exist_ok=True)
    shp = geo_dir / "NUTS_RG_01M_2021_4326_LEVL_3.shp"
    if shp.exists():
        return shp

    log.info("Downloading GISCO NUTS-3 polygons (one-time, ~5 MB)…")
    r = requests.get(GISCO_URL, timeout=120)
    r.raise_for_status()

    # The outer zip contains nested zips per resolution. We want the 01M
    # geographic-coords (EPSG:4326) NUTS level-3 shapefile.
    with zipfile.ZipFile(io.BytesIO(r.content)) as outer:
        inner_name = next(
            n for n in outer.namelist()
            if n.endswith("NUTS_RG_01M_2021_4326_LEVL_3.shp.zip")
        )
        with outer.open(inner_name) as f:
            with zipfile.ZipFile(io.BytesIO(f.read())) as inner:
                inner.extractall(geo_dir)
    log.info("Wrote shapefile to %s", shp)
    return shp


def load_landkreise(geo_dir: Path = DEFAULT_GEO_DIR):
    """Return a GeoDataFrame of German NUTS-3 Landkreise (CRS EPSG:4326)."""
    try:
        import geopandas as gpd
    except ImportError as e:
        raise ImportError("Install geopandas: pip install geopandas") from e

    shp = _ensure_gisco(geo_dir)
    gdf = gpd.read_file(shp)
    de = gdf[gdf["CNTR_CODE"] == "DE"].copy()
    de = de.rename(columns={"NUTS_ID": "landkreis", "NUTS_NAME": "landkreis_name"})
    de = de[["landkreis", "landkreis_name", "geometry"]].reset_index(drop=True)
    log.info("Loaded %d Landkreise.", len(de))
    return de


def landkreis_centroids(geo_dir: Path = DEFAULT_GEO_DIR) -> pd.DataFrame:
    """Compute (lon, lat) of each Landkreis centroid."""
    de = load_landkreise(geo_dir)
    # Project to a metric CRS for accurate centroids, then back to lon/lat
    de_m = de.to_crs(3035)
    cent = de_m.geometry.centroid.to_crs(4326)
    out = pd.DataFrame({
        "landkreis": de["landkreis"].values,
        "landkreis_name": de["landkreis_name"].values,
        "lon": cent.x.values,
        "lat": cent.y.values,
    })
    return out


# --------------------------------------------------------------------------- #
# DWD station → Landkreis mapping
# --------------------------------------------------------------------------- #
def build_station_to_landkreis(geo_dir: Path = DEFAULT_GEO_DIR) -> dict[str, str]:
    """Spatial-join DWD daily stations to NUTS-3 polygons.

    Returns
    -------
    dict
        ``{station_id (str): landkreis (NUTS-3 code)}``
    """
    try:
        import geopandas as gpd
        from shapely.geometry import Point
        from wetterdienst.provider.dwd.observation import DwdObservationRequest
    except ImportError as e:
        raise ImportError(
            "Install geopandas + wetterdienst (see requirements.txt)"
        ) from e

    de = load_landkreise(geo_dir)

    log.info("Fetching DWD station list…")
    stations = DwdObservationRequest(
        parameters=("daily", "climate_summary", "temperature_air_max_2m"),
    ).all().df.to_pandas()

    stations = stations.rename(columns={
        "longitude": "lon", "latitude": "lat", "station_id": "station_id"
    })[["station_id", "lon", "lat"]].dropna()

    pts = gpd.GeoDataFrame(
        stations,
        geometry=[Point(xy) for xy in zip(stations["lon"], stations["lat"])],
        crs=4326,
    )
    joined = gpd.sjoin(pts, de[["landkreis", "geometry"]], how="left", predicate="within")
    joined = joined.dropna(subset=["landkreis"])
    mapping = dict(zip(joined["station_id"].astype(str), joined["landkreis"]))
    log.info("Mapped %d stations into %d Landkreise.",
             len(mapping), joined["landkreis"].nunique())
    return mapping


# --------------------------------------------------------------------------- #
# Persistence helpers
# --------------------------------------------------------------------------- #
def save_station_mapping(mapping: dict[str, str], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(mapping, f)
    log.info("Saved mapping → %s", path)
    return path


def load_station_mapping(path: str | Path) -> dict[str, str]:
    with open(path, "rb") as f:
        return pickle.load(f)

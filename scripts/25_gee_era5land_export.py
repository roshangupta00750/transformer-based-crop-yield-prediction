"""Submit Earth Engine batch exports: ERA5-Land daily drivers + MODIS fPAR per NUTS-3.

Adds the water- and energy-stress drivers the models lack (soil moisture at
three depths, solar radiation, actual and potential evaporation, dew point for
VPD) plus fPAR. Exports run on Google's servers and write CSVs into the Google
Drive folder DRIVE_FOLDER (the project has no Earth Engine asset store), so the
laptop is only needed to submit and, later, to download the CSVs.

One task per (source, year). Values are area-weighted district means over the
same GISCO NUTS-3 2021 polygons as the rest of the pipeline.

Usage:
    python scripts/25_gee_era5land_export.py --years 2018            # test one year
    python scripts/25_gee_era5land_export.py --years 2002-2025        # everything
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import ee
import geopandas as gpd

from src.data.acquisition.gee_satellite import DEFAULT_PROJECT, _init

DRIVE_FOLDER = "crop_yield_drivers_v4"
ERA5_BANDS = {
    "volumetric_soil_water_layer_1": "swvl1",          # 0-7 cm, m3/m3
    "volumetric_soil_water_layer_2": "swvl2",          # 7-28 cm
    "volumetric_soil_water_layer_3": "swvl3",          # 28-100 cm
    "surface_solar_radiation_downwards_sum": "ssrd",   # J/m2/day
    "total_evaporation_sum": "evap",                   # m water eq. (negative = upward)
    "potential_evaporation_sum": "pev",                # m water eq.
    "dewpoint_temperature_2m": "d2m",                  # K, daily mean
    "temperature_2m": "t2m",                           # K, daily mean
}


def district_fc() -> ee.FeatureCollection:
    """NUTS-3 Germany, simplified to ~200 m (ERA5-Land is ~9 km, MODIS 500 m)."""
    shp = gpd.read_file(ROOT / "data/raw/geo/NUTS_RG_01M_2021_4326_LEVL_3.shp")
    de = shp[shp["CNTR_CODE"] == "DE"][["NUTS_ID", "geometry"]].copy()
    de["geometry"] = de.geometry.simplify(0.002, preserve_topology=True)
    feats = []
    for nid, geom in zip(de["NUTS_ID"], de["geometry"]):
        gj = geom.__geo_interface__
        feats.append(ee.Feature(ee.Geometry(gj), {"landkreis": nid}))
    return ee.FeatureCollection(feats)


def era5_table(fc, year: int) -> ee.FeatureCollection:
    col = (ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
           .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
           .select(list(ERA5_BANDS), list(ERA5_BANDS.values())))

    def per_day(img):
        date = ee.Date(img.get("system:time_start")).format("YYYY-MM-dd")
        red = img.reduceRegions(collection=fc,
                                reducer=ee.Reducer.mean().forEachBand(img),
                                scale=5000)
        return red.map(lambda f: f.set("date", date).setGeometry(None))

    return col.map(per_day).flatten()


def fpar_table(fc, year: int) -> ee.FeatureCollection:
    col = (ee.ImageCollection("MODIS/061/MOD15A2H")
           .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
           .select(["Fpar_500m"]))

    def per_comp(img):
        date = ee.Date(img.get("system:time_start")).format("YYYY-MM-dd")
        # 0-100 = fPAR %, 248-255 = fill / water / not classified -> mask.
        f = img.updateMask(img.lte(100)).multiply(0.01).rename("fpar")
        red = f.reduceRegions(collection=fc, reducer=ee.Reducer.mean().forEachBand(f),
                              scale=500)
        return red.map(lambda ft: ft.set("date", date).setGeometry(None))

    return col.map(per_comp).flatten()


def parse_years(s: str) -> list[int]:
    if "-" in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",")]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--years", required=True)
    ap.add_argument("--sources", default="era5,fpar")
    args = ap.parse_args()

    _init(None)
    fc = district_fc()
    submitted = {}
    for year in parse_years(args.years):
        for src in args.sources.split(","):
            table = era5_table(fc, year) if src == "era5" else fpar_table(fc, year)
            cols = ["landkreis", "date"] + (list(ERA5_BANDS.values()) if src == "era5" else ["fpar"])
            task = ee.batch.Export.table.toDrive(
                collection=table, description=f"{src}_{year}",
                folder=DRIVE_FOLDER, fileNamePrefix=f"{src}_{year}",
                fileFormat="CSV", selectors=cols)
            task.start()
            submitted[f"{src}_{year}"] = task.id
            print(f"submitted {src}_{year}: {task.id}", flush=True)
    log = ROOT / "results/logs/gee_drivers_v4_tasks.json"
    log.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(log.read_text()) if log.exists() else {}
    log.write_text(json.dumps({**prev, **submitted}, indent=2))
    print(f"{len(submitted)} task(s) submitted; ids in {log}")


if __name__ == "__main__":
    main()

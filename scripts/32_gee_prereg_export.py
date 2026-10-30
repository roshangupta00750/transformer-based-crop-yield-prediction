"""Earth Engine exports for pre-registered arms C and D (paper/preregistration_2026-09-26.md).

Sources (one Drive CSV per source-year, folder DRIVE_FOLDER):
    era5x_<Y>     ERA5-Land daily: soil_temperature_level_1 (K), snow_depth (m),
                  evaporation_from_vegetation_transpiration_sum (m),
                  potential_evaporation_sum (m)                        -> arm C
    vicrop_<Y>    MODIS MOD13Q1 NDVI, EVI (x0.0001), 250 m, cropland only -> arm D
    laicrop_<Y>   MODIS MOD15A2H Lai_500m (x0.1), Fpar_500m (fill>100 masked,
                  x0.01), 500 m, cropland only                          -> arm D
    arable        share of CORINE 2018 class 211 per district (fallback rule)
    era5w_<Y>     ERA5-Land daily over cropland: temperature_2m_max/min, temperature_2m,
                  dewpoint_temperature_2m (K), total_precipitation_sum (m),
                  u/v_component_of_wind_10m (m/s)      -> preregistration v10 §2
    vics_<Y>      MOD13Q1 NDVI/EVI averaged over each crop's own fields (EUCROPMAP 2018 u 2022,
                  pixel weight = crop area fraction)              -> preregistration v13 §2
    laics_<Y>     MOD15A2H LAI/fPAR, same crop weighting
    cropshare     each crop's mapped share of the district area (static)
    flower_<Y>    MODIS MCD43A4 daily NBAR, NDYI = (green - blue) / (green + blue)
                  (bands 4, 3; mandatory quality 0), 500 m, cropland only,
                  1 Mar - 30 Jun                       -> preregistration v8 §2.1

Cropland = CORINE Land Cover 2018 class 211 (non-irrigated arable land),
COPERNICUS/CORINE/V20/100m/2018. Scale factors and gap handling for MODIS are
the same as the existing whole-district pipeline (src/data/acquisition/
gee_satellite.py; fPAR as scripts/25), so the mask is the only difference.

Usage:
    python scripts/32_gee_prereg_export.py --years 2002-2025
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import ee

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("x25", ROOT / "scripts/25_gee_era5land_export.py")
x25 = importlib.util.module_from_spec(spec); spec.loader.exec_module(x25)

DRIVE_FOLDER = "crop_yield_prereg_v1"
ERA5X = {"soil_temperature_level_1": "stl1", "snow_depth": "sd",
         "evaporation_from_vegetation_transpiration_sum": "transp",
         "potential_evaporation_sum": "pev"}


def arable_mask() -> ee.Image:
    return ee.Image("COPERNICUS/CORINE/V20/100m/2018").select("landcover").eq(211)


def per_image(fc, img, scale, date_prop="system:time_start", tile_scale=1):
    date = ee.Date(img.get(date_prop)).format("YYYY-MM-dd")
    red = img.reduceRegions(collection=fc, reducer=ee.Reducer.mean().forEachBand(img), scale=scale,
                            tileScale=tile_scale)
    return red.map(lambda f: f.set("date", date).setGeometry(None))


def era5x(fc, y):
    col = (ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
           .filterDate(f"{y}-01-01", f"{y + 1}-01-01").select(list(ERA5X), list(ERA5X.values())))
    return col.map(lambda i: per_image(fc, i, 5000)).flatten(), ["landkreis", "date"] + list(ERA5X.values())


def vicrop(fc, y):
    m = arable_mask()
    col = (ee.ImageCollection("MODIS/061/MOD13Q1").filterDate(f"{y}-01-01", f"{y + 1}-01-01")
           .select(["NDVI", "EVI"]))
    def f(i):
        s = ee.Image(i.multiply(0.0001).updateMask(m).rename(["ndvi_crop", "evi_crop"])
                     .copyProperties(i, ["system:time_start"]))
        return per_image(fc, s, 250)
    return col.map(f).flatten(), ["landkreis", "date", "ndvi_crop", "evi_crop"]


def laicrop(fc, y):
    m = arable_mask()
    col = (ee.ImageCollection("MODIS/061/MOD15A2H").filterDate(f"{y}-01-01", f"{y + 1}-01-01")
           .select(["Lai_500m", "Fpar_500m"]))
    def f(i):
        lai = i.select("Lai_500m").multiply(0.1).rename("lai_crop")
        fp = i.select("Fpar_500m"); fp = fp.updateMask(fp.lte(100)).multiply(0.01).rename("fpar_crop")
        s = ee.Image(lai.addBands(fp).updateMask(m).copyProperties(i, ["system:time_start"]))
        return per_image(fc, s, 500)
    return col.map(f).flatten(), ["landkreis", "date", "lai_crop", "fpar_crop"]


ERA5W = {"temperature_2m_max": "tmax_k", "temperature_2m_min": "tmin_k", "temperature_2m": "tmean_k",
         "dewpoint_temperature_2m": "dew_k", "total_precipitation_sum": "prec_m",
         "u_component_of_wind_10m": "u10", "v_component_of_wind_10m": "v10"}


def era5w(fc, y):
    m = arable_mask()
    start = "1999-07-01" if y == 1999 else f"{y}-01-01"
    col = (ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR").filterDate(start, f"{y + 1}-01-01")
           .select(list(ERA5W), list(ERA5W.values())))
    return (col.map(lambda i: per_image(fc, ee.Image(i.updateMask(m).copyProperties(i, ["system:time_start"])), 2000))
            .flatten(), ["landkreis", "date"] + list(ERA5W.values()))


# preregistration v13 §2.1: EUCROPMAP class -> mask name
CS_CLASSES = {"wheat": 211, "barley": 213, "rye": 214, "oats": 215, "maize": 216,
              "triticale": 218, "potato": 221, "sugarbeet": 222, "rapeseed": 232}


def crop_mask(cls: int) -> ee.Image:
    c = ee.ImageCollection("JRC/D5/EUCROPMAP/V1")
    m18 = c.filter(ee.Filter.eq("system:index", "EU28_2018")).first().select("classification").eq(cls)
    m22 = c.filter(ee.Filter.eq("system:index", "EU28_2022")).first().select("classification").eq(cls)
    return m18.Or(m22)                      # both maps cover Germany at native 10 m (EPSG:3035)


def crop_fraction(cls: int, proj, max_pixels: int) -> ee.Image:
    """Area fraction of the crop within each pixel of `proj` (10 m mask averaged up)."""
    return (crop_mask(cls)
            .reduceResolution(reducer=ee.Reducer.mean(), maxPixels=max_pixels))   # output grid = the MODIS band it masks


CS_SCALE = 100   # m: the 10 m crop mask is sampled on a 100 m grid, so each MODIS pixel
                 # counts in proportion to its crop-covered area (v13 §2.2)


def _cs_table(fc, img_col, bands, scale, max_pixels, scale_fn):
    masks = {k: crop_mask(v) for k, v in CS_CLASSES.items()}

    def f(i):
        base = scale_fn(i)
        parts = [base.select(b).updateMask(masks[k]).rename(f"{b}_{k}") for k in CS_CLASSES for b in bands]
        return per_image(fc, ee.Image(ee.Image.cat(parts).copyProperties(i, ["system:time_start"])),
                         CS_SCALE, tile_scale=8)
    cols = ["landkreis", "date"] + [f"{b}_{k}" for k in CS_CLASSES for b in bands]
    return img_col.map(f).flatten(), cols


def vics(fc, y):
    col = ee.ImageCollection("MODIS/061/MOD13Q1").filterDate(f"{y}-01-01", f"{y + 1}-01-01").select(["NDVI", "EVI"])
    return _cs_table(fc, col, ["ndvi", "evi"], 250, 1024,
                     lambda i: i.multiply(0.0001).rename(["ndvi", "evi"]))


def laics(fc, y):
    col = (ee.ImageCollection("MODIS/061/MOD15A2H").filterDate(f"{y}-01-01", f"{y + 1}-01-01")
           .select(["Lai_500m", "Fpar_500m"]))

    def sc(i):
        fp = i.select("Fpar_500m")
        return ee.Image.cat([i.select("Lai_500m").multiply(0.1).rename("lai"),
                             fp.updateMask(fp.lte(100)).multiply(0.01).rename("fpar")])
    return _cs_table(fc, col, ["lai", "fpar"], 500, 4096, sc)


def cropshare_table(fc):
    img = ee.Image.cat([crop_mask(v).rename(k) for k, v in CS_CLASSES.items()])
    red = img.reduceRegions(collection=fc, reducer=ee.Reducer.mean().forEachBand(img), scale=100)
    return red.map(lambda f: f.setGeometry(None)), ["landkreis"] + list(CS_CLASSES)


def flower(fc, y):
    m = arable_mask()
    col = (ee.ImageCollection("MODIS/061/MCD43A4").filterDate(f"{y}-03-01", f"{y}-07-01")
           .select(["Nadir_Reflectance_Band3", "Nadir_Reflectance_Band4",
                    "BRDF_Albedo_Band_Mandatory_Quality_Band3", "BRDF_Albedo_Band_Mandatory_Quality_Band4"]))

    def f(i):
        good = i.select("BRDF_Albedo_Band_Mandatory_Quality_Band3").eq(0).And(
            i.select("BRDF_Albedo_Band_Mandatory_Quality_Band4").eq(0))
        b, g = i.select("Nadir_Reflectance_Band3"), i.select("Nadir_Reflectance_Band4")
        ndyi = g.subtract(b).divide(g.add(b)).rename("ndyi_crop")
        return per_image(fc, ee.Image(ndyi.updateMask(good).updateMask(m)
                                      .copyProperties(i, ["system:time_start"])), 500)
    return col.map(f).flatten(), ["landkreis", "date", "ndyi_crop"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--years", required=True)
    ap.add_argument("--sources", default="era5x,vicrop,laicrop,arable")
    args = ap.parse_args()
    x25._init(None)
    fc = x25.district_fc()
    srcs = args.sources.split(",")
    submitted = {}

    def go(name, table, cols):
        t = ee.batch.Export.table.toDrive(collection=table, description=name,
                                          folder=DRIVE_FOLDER, fileNamePrefix=name,
                                          fileFormat="CSV", selectors=cols)
        t.start(); submitted[name] = t.id; print("submitted", name, t.id, flush=True)

    if "arable" in srcs:
        # single-band mean() writes its output as "mean", not the band name
        # (same pitfall as gee_satellite._reduce_image) -> name it explicitly
        red = arable_mask().reduceRegions(
            collection=fc, reducer=ee.Reducer.mean().setOutputs(["arable_share"]), scale=100)
        go("arable", red.map(lambda f: f.setGeometry(None)), ["landkreis", "arable_share"])
    if "cropshare" in srcs:
        tbl, cols = cropshare_table(fc)
        go("cropshare", tbl, cols)
    builders = {"era5x": era5x, "vicrop": vicrop, "laicrop": laicrop, "flower": flower, "era5w": era5w,
                "vics": vics, "laics": laics}
    for y in x25.parse_years(args.years):
        for s in srcs:
            if s in builders:
                tbl, cols = builders[s](fc, y)
                go(f"{s}_{y}", tbl, cols)
    log = ROOT / "results/logs/gee_prereg_tasks.json"
    prev = json.loads(log.read_text()) if log.exists() else {}
    log.write_text(json.dumps({**prev, **submitted}, indent=2))
    print(f"{len(submitted)} task(s) submitted -> {log}")


if __name__ == "__main__":
    main()

"""Download all raw data sources into data/raw/.

Usage:
    python scripts/01_download_data.py [--source dwd|gee|soilgrids|destatis|all]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.acquisition import dwd_weather, destatis_yield, soilgrids, gee_satellite
from src.utils import get_logger, load_data_config, PROJECT_ROOT

log = get_logger("download")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["dwd", "gee", "soilgrids", "destatis", "all"],
                        default="all")
    parser.add_argument("--landkreis-asset", default="",
                        help="GEE asset path for Landkreis polygons (required for --source gee)")
    args = parser.parse_args()

    cfg = load_data_config()
    raw = PROJECT_ROOT / cfg["paths"]["raw"]
    yr = cfg["years"]
    start_date = f"{yr['start']}-01-01"
    end_date = f"{yr['end']}-12-31"

    if args.source in ("dwd", "all"):
        log.info("=== DWD weather ===")
        dwd_weather.download(raw / "dwd", start_date=start_date, end_date=end_date)

    if args.source in ("destatis", "all"):
        log.info("=== Destatis yield ===")
        try:
            destatis_yield.download_via_api(
                raw / "destatis",
                startyear=yr["start"],
                endyear=yr["end"],
            )
        except Exception as e:
            log.error("Destatis API failed (%s). Download the CSV manually "
                      "from https://www.regionalstatistik.de and place it in %s",
                      e, raw / "destatis")

    if args.source in ("soilgrids", "all"):
        log.info("=== SoilGrids ===")
        cent_path = PROJECT_ROOT / cfg["paths"]["interim"] / "landkreis_centroids.parquet"
        if not cent_path.exists():
            log.error("Missing %s — run `python scripts/00_build_geo.py` first.", cent_path)
        else:
            import pandas as pd
            soilgrids.download(raw / "soilgrids",
                               landkreis_centroids=pd.read_parquet(cent_path))

    if args.source in ("gee", "all"):
        log.info("=== Google Earth Engine ===")
        # Empty asset string falls back to an inline FeatureCollection built
        # from the local NUTS-3 shapefile (no asset upload required).
        gee_satellite.download(raw / "gee",
                               landkreis_asset=args.landkreis_asset,
                               start_date=start_date, end_date=end_date,
                               composite_days=cfg["satellite_composite_days"])

    log.info("Done. Raw data at %s", raw)


if __name__ == "__main__":
    main()

"""Build the one-time geo artefacts.

Produces:
  - data/interim/landkreis_centroids.parquet
  - data/interim/station_to_landkreis.pkl

Run this before 01_download_data.py so downstream cleaning has the
station→Landkreis mapping it needs.

Usage:
    python scripts/00_build_geo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.landkreise import (
    build_station_to_landkreis,
    landkreis_centroids,
    save_station_mapping,
)
from src.utils import get_logger, load_data_config, PROJECT_ROOT

log = get_logger("build_geo")


def main() -> None:
    cfg = load_data_config()
    interim = PROJECT_ROOT / cfg["paths"]["interim"]
    interim.mkdir(parents=True, exist_ok=True)

    log.info("Computing Landkreis centroids…")
    cent = landkreis_centroids()
    cent_path = interim / "landkreis_centroids.parquet"
    cent.to_parquet(cent_path, index=False)
    log.info("Wrote %s (%d rows)", cent_path, len(cent))

    log.info("Building DWD station → Landkreis mapping…")
    mapping = build_station_to_landkreis()
    map_path = interim / "station_to_landkreis.pkl"
    save_station_mapping(mapping, map_path)


if __name__ == "__main__":
    main()

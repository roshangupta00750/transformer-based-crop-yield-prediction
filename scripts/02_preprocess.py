"""Clean each raw source → interim parquet → unified dataset.

Usage:
    python scripts/02_preprocess.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.acquisition import dwd_weather, destatis_yield, soilgrids, gee_satellite
from src.data.dataset import build_unified_dataset
from src.data.landkreise import load_station_mapping
from src.utils import get_logger, load_data_config, PROJECT_ROOT

log = get_logger("preprocess")


def main() -> None:
    cfg = load_data_config()
    raw = PROJECT_ROOT / cfg["paths"]["raw"]
    interim = PROJECT_ROOT / cfg["paths"]["interim"]
    out = PROJECT_ROOT / cfg["paths"]["unified_dataset"]

    log.info("Cleaning weather…")
    map_path = interim / "station_to_landkreis.pkl"
    if not map_path.exists():
        log.error("Missing %s — run `python scripts/00_build_geo.py` first.", map_path)
    else:
        try:
            dwd_weather.clean(raw / "dwd/dwd_daily.parquet",
                              interim / "weather.parquet",
                              station_to_landkreis=load_station_mapping(map_path))
        except FileNotFoundError:
            log.warning("Skip weather — raw file missing.")

    log.info("Cleaning satellite…")
    try:
        gee_satellite.clean(raw / "gee/gee_satellite.parquet",
                            interim / "satellite.parquet")
    except FileNotFoundError:
        log.warning("Skip satellite — raw file missing.")

    log.info("Cleaning soil…")
    try:
        soilgrids.clean(raw / "soilgrids/soilgrids.parquet",
                        interim / "soil.parquet")
    except FileNotFoundError:
        log.warning("Skip soil — raw file missing.")

    log.info("Cleaning yield…")
    try:
        destatis_yield.clean(raw / "destatis/destatis_41241-01-03-4.csv",
                             interim / "yield.parquet")
    except FileNotFoundError:
        log.warning("Skip yield — raw file missing.")

    log.info("Building unified dataset → %s", out)
    df = build_unified_dataset(
        interim_dir=interim,
        out_path=out,
        crops=cfg["crops"],
        year_range=(cfg["years"]["start"], cfg["years"]["end"]),
    )
    log.info("Unified dataset: %d rows", len(df))


if __name__ == "__main__":
    main()

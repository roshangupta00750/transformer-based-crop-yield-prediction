"""DWD weather data via the `wetterdienst` library.

Pulls daily climate observations from the Deutscher Wetterdienst (DWD)
historical archive. We grab the nearest N stations per Landkreis centroid
and aggregate to a single daily series per Landkreis.

Reference: https://wetterdienst.readthedocs.io/
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ...utils import get_logger

log = get_logger("dwd")


def download(
    out_dir: str | Path,
    start_date: str = "2003-01-01",
    end_date: str = "2023-12-31",
    parameters: tuple[str, ...] = (
        "temperature_air_max_2m",
        "temperature_air_min_2m",
        "precipitation_height",
        "radiation_global",
        "humidity",
    ),
) -> Path:
    """Download DWD daily observations to `out_dir`.

    Returns the path to the raw parquet file.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "dwd_daily.parquet"

    try:
        from wetterdienst.provider.dwd.observation import DwdObservationRequest
    except ImportError as e:
        raise ImportError(
            "Install wetterdienst: pip install wetterdienst>=0.79"
        ) from e

    # Map parameter names to (resolution, dataset, parameter) tuples
    # radiation_global lives in 'solar'; the others in 'climate_summary'
    _dataset_for = {
        "radiation_global": "solar",
    }
    param_tuples = [
        ("daily", _dataset_for.get(p, "climate_summary"), p) for p in parameters
    ]

    log.info("Fetching DWD daily climate for %s..%s", start_date, end_date)
    request = DwdObservationRequest(
        parameters=param_tuples,
        start_date=start_date,
        end_date=end_date,
    ).all()

    df = request.values.all().df.to_pandas()
    df.to_parquet(out_file, index=False)
    log.info("Wrote %s (%d rows)", out_file, len(df))
    return out_file


def clean(
    raw_path: str | Path,
    out_path: str | Path,
    station_to_landkreis: dict[str, str],
) -> Path:
    """Aggregate station-level DWD data to one daily row per Landkreis.

    `station_to_landkreis` must map each DWD station_id to its containing
    Landkreis (built once from a shapefile join).
    """
    raw_path, out_path = Path(raw_path), Path(out_path)
    df = pd.read_parquet(raw_path)
    df["landkreis"] = df["station_id"].astype(str).map(station_to_landkreis)
    df = df.dropna(subset=["landkreis"])

    # pivot to wide format: one column per parameter
    wide = df.pivot_table(
        index=["landkreis", "date"],
        columns="parameter",
        values="value",
        aggfunc="mean",
    ).reset_index()
    wide.columns.name = None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    wide.to_parquet(out_path, index=False)
    log.info("Cleaned weather → %s (%d Landkreis-days)", out_path, len(wide))
    return out_path

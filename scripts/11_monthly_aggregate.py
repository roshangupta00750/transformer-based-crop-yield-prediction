"""Aggregate the daily observed dataset to monthly.

Produces ``data/processed/dataset_monthly.parquet`` with one row per
(landkreis, crop, year, month). This is the observed-side companion to
the CMIP6 monthly projections in ``cmip6_nuts3.parquet`` — same
temporal resolution + column names so counterfactual features can be
built by column swap.

Column mapping (daily -> monthly):
    temperature_air_max_2m  -> tasmax   (monthly mean of daily max)
    temperature_air_min_2m  -> tasmin   (monthly mean of daily min)
    precipitation_height    -> pr       (monthly sum in mm)
    radiation_global        -> rsds     (monthly mean of daily radiation)
    humidity                -> hurs     (monthly mean; not in CMIP6 pull)
    ndvi / evi / lai        -> monthly mean (no CMIP6 counterpart)
    soil (static)           -> unchanged (constant per district)
    yield_dt_per_ha         -> unchanged (constant per district-crop-year)
    landkreis_id            -> unchanged

Usage:
    python scripts/11_monthly_aggregate.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
IN_PATH = ROOT / "data" / "processed" / "dataset.parquet"
OUT_PATH = ROOT / "data" / "processed" / "dataset_monthly.parquet"


def main() -> None:
    if not IN_PATH.exists():
        sys.exit(f"input missing: {IN_PATH}")
    print(f"reading {IN_PATH}...")
    df = pd.read_parquet(IN_PATH)
    print(f"  {len(df):,} daily rows")

    df["month"] = pd.to_datetime(df["date"]).dt.month
    df["year"] = pd.to_datetime(df["date"]).dt.year

    # Static per (landkreis, crop, year) values that just pass through
    static_cols = ["bulk_density", "clay", "ph", "sand", "soc",
                   "landkreis_id", "yield_dt_per_ha"]

    # Mean-aggregation for temperature / radiation / humidity / satellite
    mean_cols = ["temperature_air_max_2m", "temperature_air_min_2m",
                 "radiation_global", "humidity",
                 "ndvi", "evi", "lai"]

    # Sum-aggregation for precipitation
    sum_cols = ["precipitation_height"]

    group = ["landkreis", "crop", "year", "month"]

    print("aggregating...")
    aggs = {c: "mean" for c in mean_cols}
    aggs.update({c: "sum" for c in sum_cols})
    for c in static_cols:
        aggs[c] = "first"
    monthly = df.groupby(group, as_index=False).agg(aggs)

    # Rename to match CMIP6 short names so counterfactual overlay is trivial
    monthly = monthly.rename(columns={
        "temperature_air_max_2m": "tasmax",
        "temperature_air_min_2m": "tasmin",
        "radiation_global": "rsds",
        "humidity": "hurs",
        "precipitation_height": "pr",
    })
    # Add a derived monthly mean temperature (matches CMIP6 `tas`)
    monthly["tas"] = (monthly["tasmax"] + monthly["tasmin"]) / 2

    # Reorder columns for readability
    id_cols = ["landkreis", "crop", "year", "month", "landkreis_id"]
    weather_cols = ["tas", "tasmax", "tasmin", "pr", "rsds", "hurs"]
    sat_cols = ["ndvi", "evi", "lai"]
    soil_cols = ["bulk_density", "clay", "ph", "sand", "soc"]
    target_col = ["yield_dt_per_ha"]
    monthly = monthly[id_cols + weather_cols + sat_cols + soil_cols + target_col]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    monthly.to_parquet(OUT_PATH, index=False)

    print(f"\nwrote {OUT_PATH} ({len(monthly):,} rows)")
    print(f"  {monthly['landkreis'].nunique()} districts")
    print(f"  {monthly['crop'].nunique()} crops")
    print(f"  {monthly['year'].nunique()} years ({monthly['year'].min()}-{monthly['year'].max()})")
    print(f"\n=== sample ===")
    print(monthly.head(3).to_string(index=False))
    print(f"\n=== monthly weather sanity (Jan 2010 mean across districts) ===")
    q = monthly[(monthly["year"] == 2010) & (monthly["month"] == 1)]
    for c in ["tas", "tasmax", "tasmin", "pr", "rsds"]:
        print(f"  {c:8s} mean {q[c].mean():.2f}  range {q[c].min():.1f} to {q[c].max():.1f}")


if __name__ == "__main__":
    main()

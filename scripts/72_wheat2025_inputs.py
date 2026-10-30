"""Pre-registration v16 §3.4, input side: was 2025 winter wheat unusual in its inputs?

Per harvest year 2010-2025, winter wheat rows of dataset_v5_hybridW (calendar part of the
season, 1 Jan - 31 Jul), averaged over districts:
    coverage   share of non-missing values per input source until harvest
    season     precipitation Mar-Jun (mm), mean daily max temperature May-Jul (°C), days >= 30 °C
               May-Jul, climatic water balance Mar-Jun (mm), top-soil moisture Apr-Jun,
               crop-mask NDVI May-Jun, crop-mask LAI May-Jun
Each season value is also given as an anomaly against the 2010-2024 mean (2025 excluded).

Output: results/metrics/v16_wheat_inputs.csv
Usage:  python scripts/72_wheat2025_inputs.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
COLS = ["landkreis", "date", "year", "precipitation_height", "temperature_air_max_2m", "cwb", "swvl1",
        "ndvi_crop", "lai_crop", "humidity", "soiltemp", "fpar", "yield_dt_per_ha"]


def main() -> None:
    d = pd.read_parquet(ROOT / "data/processed/dataset_v5_hybridW.parquet", columns=COLS,
                        filters=[("crop", "==", "winter_wheat"), ("year", ">=", 2010)])
    d["date"] = pd.to_datetime(d["date"])
    d = d[(d["date"].dt.year == d["year"]) & (d["date"].dt.month <= 7)]
    m = d["date"].dt.month
    g = d.groupby(["year", "landkreis"])
    per = pd.DataFrame({
        "prec_mar_jun": d[m.between(3, 6)].groupby(["year", "landkreis"])["precipitation_height"].sum(),
        "tmax_may_jul": d[m.between(5, 7)].groupby(["year", "landkreis"])["temperature_air_max_2m"].mean(),
        "hot_days_may_jul": (d[m.between(5, 7)]["temperature_air_max_2m"] >= 30)
                            .groupby([d["year"], d["landkreis"]]).sum(),
        "cwb_mar_jun": d[m.between(3, 6)].groupby(["year", "landkreis"])["cwb"].sum(),
        "swvl1_apr_jun": d[m.between(4, 6)].groupby(["year", "landkreis"])["swvl1"].mean(),
        "ndvi_crop_may_jun": d[m.between(5, 6)].groupby(["year", "landkreis"])["ndvi_crop"].mean(),
        "lai_crop_may_jun": d[m.between(5, 6)].groupby(["year", "landkreis"])["lai_crop"].mean(),
    })
    cov = g[["precipitation_height", "temperature_air_max_2m", "swvl1", "ndvi_crop", "lai_crop", "soiltemp",
             "fpar"]].apply(lambda x: x.notna().mean()).add_prefix("coverage_")
    T = per.join(cov).groupby("year").mean()
    T["districts"] = per.groupby("year").size()
    ref = T.loc[2010:2024]
    for c in per.columns:
        T[f"{c}_anom"] = T[c] - ref[c].mean()
        T[f"{c}_z"] = (T[c] - ref[c].mean()) / ref[c].std()
    T.round(4).to_csv(ROOT / "results/metrics/v16_wheat_inputs.csv")
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(T[[c for c in T.columns if c.endswith("_z")] + ["districts"]].round(2).to_string())
        print(T[[c for c in T.columns if c.startswith("coverage_")]].round(3).to_string())


if __name__ == "__main__":
    main()

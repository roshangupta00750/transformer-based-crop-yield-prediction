"""Crop area per district from the agricultural censuses (preregistration v15 §4).

Source: Duden, Nacke & Offermann (2024), Sci. Data 11:95, OpenAgrar DOI
10.3220/DATA20231117103252-0, data/raw/duden2024/Final_data.csv (CC-BY 4.0, 2020 boundaries).
Area (ha) exists only for census years (1999, 2003, 2007, 2010, 2016, 2020 in our period).

Output: data/processed/static/crop_area_census.parquet (landkreis, crop, census_year, area_ha)
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
VAR = {"ww": "winter_wheat", "wb": "winter_barley", "sb": "summer_barley", "rye": "rye", "oats": "oats",
       "triticale": "triticale", "silage_maize": "silage_maize", "potat_tot": "potatoes",
       "sugarbeet": "sugar_beet", "wrape": "winter_rapeseed"}

D = pd.read_csv(ROOT / "data/raw/duden2024/Final_data.csv", dtype={"district_no": str})
A = D[(D.measure == "area") & D["var"].isin(VAR) & D.value.notna() & (D.year >= 1999)]
out = (A.assign(crop=A["var"].map(VAR))
        .rename(columns={"nuts_id": "landkreis", "year": "census_year", "value": "area_ha"})
        [["landkreis", "crop", "census_year", "area_ha"]].sort_values(["crop", "landkreis", "census_year"]))
out.to_parquet(ROOT / "data/processed/static/crop_area_census.parquet", index=False)
print(f"{len(out):,} rows, {out.landkreis.nunique()} districts, census years {sorted(out.census_year.unique())}")
print(out.groupby("crop").area_ha.median().round(0).to_string())

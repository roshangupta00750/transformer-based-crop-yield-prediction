"""Tests for v13 crop-specific satellite inputs (paper/preregistration_2026-10-05_v13.md §2)."""
import numpy as np
import pandas as pd

from src.data.dataset import add_crop_specific_sat


def test_crop_specific_replaces_and_falls_back(tmp_path):
    sat = pd.DataFrame({"landkreis": ["A", "A", "B"], "mask": ["wheat", "maize", "wheat"],
                        "date": pd.to_datetime(["2018-05-01", "2018-05-01", "2018-05-01"]),
                        "ndvi": [0.80, 0.55, 0.70], "evi": [0.5, 0.3, 0.4], "lai": [3.0, 1.5, 2.5], "fpar": [0.7, 0.4, 0.6]})
    sp, sh = tmp_path / "s.parquet", tmp_path / "share.csv"
    sat.to_parquet(sp)
    pd.DataFrame({"landkreis": ["A", "B"], "wheat": [0.3, 0.001], "maize": [0.1, 0.1]}).assign(
        barley=0, rye=0, oats=0, triticale=0, potato=0, sugarbeet=0, rapeseed=0).to_csv(sh, index=False)
    df = pd.DataFrame({"landkreis": ["A", "A", "B", "A"], "crop": ["winter_wheat", "silage_maize", "winter_wheat", "winter_wheat"],
                       "date": pd.to_datetime(["2018-05-05", "2018-05-05", "2018-05-05", "2018-07-01"]),
                       "ndvi_crop": 0.6, "evi_crop": 0.35, "lai_crop": 2.0, "fpar_crop": 0.5})
    out = add_crop_specific_sat(df, sp, sh)
    assert out.loc[0, "ndvi_crop"] == 0.80 and out.loc[0, "cs_fallback"] == 0      # wheat own fields
    assert out.loc[1, "ndvi_crop"] == 0.55 and out.loc[1, "cs_fallback"] == 0      # maize own fields
    assert out.loc[2, "ndvi_crop"] == 0.6 and out.loc[2, "cs_fallback"] == 1       # wheat share < 0.5% -> cropland
    assert out.loc[3, "ndvi_crop"] == 0.6 and out.loc[3, "cs_fallback"] == 1       # value older than 32 days -> cropland
    assert np.isclose(out.loc[0, "lai_crop"], 3.0)

"""Preregistration v15 §4: causal census area -> sqrt weights, normalised per crop, clipped."""
import numpy as np
import pandas as pd

from src.data.dataset import area_weights


def _census(tmp_path):
    c = pd.DataFrame({"landkreis": ["A", "A", "B", "B", "C"], "crop": ["oats"] * 5,
                      "census_year": [1999, 2010, 1999, 2010, 2010], "area_ha": [100., 400., 900., 900., 2500.]})
    p = tmp_path / "c.parquet"; c.to_parquet(p); return p


def test_causal_area_and_normalisation(tmp_path):
    p = _census(tmp_path)
    rows = pd.DataFrame({"landkreis": ["A", "A", "B", "C"], "crop": "oats", "year": [2005, 2012, 2012, 2012]})
    w = area_weights(rows, p)
    # A 2005 uses the 1999 census (100 ha), not 2010; A 2012 uses 2010 (400 ha)
    r = {"A05": 10.0, "A12": 20.0, "B12": 30.0}
    # C 2012 uses its 2010 census (2500 ha); C has no 1999 value
    r["C12"] = 50.0
    m = np.mean(list(r.values()))
    assert np.isclose(w[("A", "oats", 2005)], 10 / m)
    assert np.isclose(w[("A", "oats", 2012)], 20 / m)
    assert np.isclose(np.mean(list(w.values())), 1.0, atol=0.01)


def test_missing_area_uses_crop_low_quantile_and_clip(tmp_path):
    p = _census(tmp_path)
    rows = pd.DataFrame({"landkreis": ["Z", "C"], "crop": "oats", "year": [2005, 2012]})
    w = area_weights(rows, p)                      # Z never reported: 5th pct of the 1999 census
    assert w[("Z", "oats", 2005)] >= 0.25 and w[("C", "oats", 2012)] <= 4.0

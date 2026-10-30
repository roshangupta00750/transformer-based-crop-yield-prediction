"""Tests for the v8 flowering index (paper/preregistration_2026-10-02_v8.md §2.1)."""
from __future__ import annotations

import pandas as pd

from src.data.dataset import add_flowering_index


def test_flowering_index_joins_by_district_and_day(tmp_path):
    nd = pd.DataFrame({"landkreis": ["A", "A", "B"], "date": ["2018-05-01", "2018-05-02", "2018-05-01"],
                       "ndyi_crop": [0.30, 0.35, 0.20]})
    p = tmp_path / "nd.parquet"; nd.to_parquet(p)
    df = pd.DataFrame({"landkreis": ["A", "A", "A", "B"],
                       "date": pd.to_datetime(["2018-05-01", "2018-05-02", "2018-09-01", "2018-05-02"], utc=True),
                       "crop": "winter_rapeseed", "year": 2018})
    out = add_flowering_index(df, p)
    assert out["ndyi_crop"].tolist()[:2] == [0.30, 0.35]
    assert out["ndyi_observed"].tolist() == [1, 1, 0, 0]
    assert len(out) == len(df) and list(out["date"]) == list(df["date"])

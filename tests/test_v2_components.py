"""Tests for the v2 components: observed phenology inputs, national model,
target offset (paper/preregistration_2026-09-28_v2.md)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.data.dataset import (PHENOLOGY_SLOTS, add_phenology_features,
                              phenology_columns)
from src.models.national import NationalModel, season_features


@pytest.fixture
def obs(tmp_path):
    rows = [
        # harvest year 2021 winter wheat: sown in autumn 2020, heads in May 2021
        ("DE111", "winter_wheat", 2021, "full", "sow", "2020-10-05"),
        ("DE111", "winter_wheat", 2021, "full", "head", "2021-05-28"),
        # the previous crop-year's heading must never leak into 2021's autumn rows
        ("DE111", "winter_wheat", 2020, "full", "head", "2020-06-01"),
        ("DE111", "winter_wheat", 2021, "rt", "head", "2021-06-02"),
    ]
    d = pd.DataFrame(rows, columns=["landkreis", "crop", "year", "network", "slot", "date"])
    d["date"] = pd.to_datetime(d["date"])
    p = tmp_path / "ph.parquet"
    d.to_parquet(p)
    return p


def frame(dates, year=2021):
    d = pd.to_datetime(pd.Series(dates))
    return pd.DataFrame({"landkreis": "DE111", "crop": "winter_wheat", "year": year, "date": d})


def test_step_is_causal(obs):
    f = add_phenology_features(frame(["2021-05-27", "2021-05-28", "2021-07-01"]), "full", obs)
    assert f["ph_head"].tolist() == [0, 1, 1]
    assert f["ph_head_missing"].tolist() == [0, 0, 0]


def test_autumn_rows_use_harvest_year_dates(obs):
    # autumn rows carry year = 2021 (relabelled crop-year); sown 5 Oct 2020
    f = add_phenology_features(frame(["2020-10-04", "2020-10-05", "2020-12-31"]), "full", obs)
    assert f["ph_sow"].tolist() == [0, 1, 1]
    # 2020's heading (1 Jun 2020) belongs to crop-year 2020, not to these rows
    assert f["ph_head"].tolist() == [0, 0, 0]


def test_missing_and_network(obs):
    f = add_phenology_features(frame(["2021-05-30"]), "rt", obs)
    assert f["ph_head"].tolist() == [0]            # rt date is 2 Jun
    assert f["ph_sow_missing"].tolist() == [1]     # no rt sowing date
    assert set(phenology_columns()) <= set(f.columns)
    assert len(phenology_columns()) == 2 * len(PHENOLOGY_SLOTS)


def test_national_model_roundtrip():
    rng = np.random.default_rng(0)
    rows = []
    for crop in ("oats", "rye"):
        for yr in range(2000, 2020):
            t = rng.normal()
            for lk in ("A", "B"):
                for day in pd.date_range(f"{yr}-04-01", f"{yr}-07-31", freq="7D"):
                    rows.append((crop, yr, day, lk, t + rng.normal(scale=0.1), 0))
    f = pd.DataFrame(rows, columns=["crop", "year", "date", "landkreis", "tmax", "tmax_missing"])
    X = season_features(f, ["tmax", "tmax_missing"], phenology=False)
    assert not any(c.startswith("tmax_missing") for c in X.columns)
    y = X.filter(like="tmax@").mean(axis=1)            # target driven by the input
    m = NationalModel().fit(X, y)
    p = m.predict(X)
    assert np.corrcoef(p, y.reindex(p.index))[0, 1] > 0.9
    m2 = NationalModel.from_dict(json.loads(json.dumps(m.to_dict())))
    np.testing.assert_allclose(m2.predict(X), p)

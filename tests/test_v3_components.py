"""Tests for the v3 inputs: harvest slot, crop-stage weather, district anomalies
(paper/preregistration_2026-09-29_v3.md)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.dataset import (PHENOLOGY_SLOTS_V3, add_district_anomalies,
                              add_phenology_features, add_stage_features,
                              anomaly_columns, fit_district_climatology)


@pytest.fixture
def obs(tmp_path):
    rows = [("DE111", "winter_wheat", 2021, "full", s, d) for s, d in [
        ("sow", "2021-06-01"), ("head", "2021-06-03"), ("ripe", "2021-06-06"), ("harvest", "2021-06-08")]]
    d = pd.DataFrame(rows, columns=["landkreis", "crop", "year", "network", "slot", "date"])
    d["date"] = pd.to_datetime(d["date"])
    p = tmp_path / "ph3.parquet"
    d.to_parquet(p)
    return p


@pytest.fixture
def days():
    d = pd.date_range("2021-05-30", "2021-06-10")
    return pd.DataFrame({"landkreis": "DE111", "crop": "winter_wheat", "year": 2021, "date": d,
                         "temperature_air_max_2m": 32.0, "temperature_air_min_2m": 12.0,
                         "precipitation_height": 1.0, "cwb": -2.0})


def test_harvest_slot_is_a_causal_step(obs, days):
    f = add_phenology_features(days, "full", obs, PHENOLOGY_SLOTS_V3)
    on = f.set_index("date")["ph_harvest"]
    assert on[:"2021-06-07"].eq(0).all() and on["2021-06-08":].eq(1).all()


def test_stage_totals(obs, days):
    f = add_stage_features(days.sample(frac=1, random_state=0), "full", obs).sort_values("date").set_index("date")
    # season totals start at the observed sowing date (1 Jun): 22 GDD/day
    assert f.loc["2021-05-31", "cum_gdd"] == 0
    assert f.loc["2021-06-02", "cum_gdd"] == 44
    # grain fill from heading (3 Jun) to ripeness (6 Jun, first of ripe/harvest) = 4 days, then frozen
    assert f.loc["2021-06-02", "gf_days"] == 0
    assert f.loc["2021-06-06", "gf_days"] == 4
    assert f.loc["2021-06-10", "gf_days"] == 4 and f.loc["2021-06-10", "gf_heat"] == 4
    assert f.loc["2021-06-10", "gf_cwb"] == -8
    # days since the latest observed stage; -1 before any
    assert f.loc["2021-05-31", "days_since_stage"] == -1
    assert f.loc["2021-06-07", "days_since_stage"] == 1
    assert f.loc["2021-06-10", "days_since_stage"] == 2


def test_stage_totals_do_not_see_future_stages(obs, days):
    # truncating the season at 4 Jun must not change any value up to 4 Jun
    full = add_stage_features(days, "full", obs).set_index("date")
    cut = add_stage_features(days[days.date <= "2021-06-04"], "full", obs).set_index("date")
    cols = ["cum_gdd", "gf_days", "gf_gdd", "days_since_stage"]
    pd.testing.assert_frame_equal(full.loc[:"2021-06-04", cols], cut[cols])


def test_district_climatology_uses_training_rows_only():
    train = pd.DataFrame({"landkreis": ["A", "A", "B"], "date": pd.to_datetime(["2010-05-01", "2011-05-01", "2010-05-01"]),
                          "ndvi_crop": [0.4, 0.6, 0.2], "crop": "oats"})
    clim = fit_district_climatology(train, ["ndvi_crop"])
    assert clim == {"ndvi_crop": {"A|5": 0.5, "B|5": 0.2}}
    test = pd.DataFrame({"landkreis": ["A", "C"], "date": pd.to_datetime(["2020-05-10", "2020-05-10"]),
                         "ndvi_crop": [0.7, 0.3]})
    out = add_district_anomalies(test, clim)
    assert out["ndvi_crop_anom"].iloc[0] == pytest.approx(0.2)
    assert np.isnan(out["ndvi_crop_anom"].iloc[1])          # unknown district -> filled later
    assert anomaly_columns(clim) == ["ndvi_crop_anom"]

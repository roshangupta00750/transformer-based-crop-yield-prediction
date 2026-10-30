"""Tests for the v5 national gate (paper/preregistration_2026-09-30_v5.md §2.1)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.national_gate import (INDICATORS, apply_gate, gate_decisions,
                                      national_anomaly, national_indicators)


def _ind(years, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(size=(len(years), len(INDICATORS))), index=pd.Index(years, name="year"),
                        columns=INDICATORS)


def _yields(crop, anom_by_year, n=30, seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    for y, a in anom_by_year.items():
        for d in range(n):
            base = 60 + d
            rows.append((f"D{d}", crop, y, base * (1 + a / 100) + rng.normal(0, 0.3)))
    return pd.DataFrame(rows, columns=["landkreis", "crop", "year", "yield"])


def test_gate_on_for_noise_off_for_weather_driven_crop():
    years = list(range(2000, 2022))
    ind = _ind(years)
    rng = np.random.default_rng(5)
    maize = dict(zip(years, 6 * ind["rain_jj"].to_numpy()))            # follows summer rain
    wheat = dict(zip(years, rng.normal(0, 5, len(years))))            # unrelated to weather
    y = pd.concat([_yields("silage_maize", maize), _yields("winter_wheat", wheat)])
    g = gate_decisions(y, ind, train_end=2021).set_index("crop")
    assert not g.loc["silage_maize", "gate"] and g.loc["silage_maize", "loyo_r2"] > 0.8
    assert g.loc["winter_wheat", "gate"]


def test_gate_uses_training_years_only():
    years = list(range(2000, 2022))
    ind = _ind(years)
    maize = dict(zip(years, 6 * ind["rain_jj"].to_numpy()))
    y = _yields("silage_maize", maize)
    a = gate_decisions(y, ind, train_end=2015)
    y2 = y.copy(); y2.loc[y2.year > 2015, "yield"] = 1.0                 # garbage after train_end
    b = gate_decisions(y2, ind, train_end=2015)
    assert a.loyo_r2.iloc[0] == b.loyo_r2.iloc[0]


def test_national_anomaly_is_detrended_percent():
    y = _yields("oats", {yr: 2.0 * (yr - 2000) for yr in range(2000, 2010)})
    a = national_anomaly(y[["landkreis", "year", "yield"]])
    assert abs(a).max() < 1.0                                          # a pure trend is removed


def test_apply_gate_removes_only_the_national_mean_of_gated_crops():
    p = pd.DataFrame({"crop": ["winter_wheat"] * 2 + ["silage_maize"] * 2, "year": 2024,
                      "y_hist": [70.0, 80.0, 400.0, 420.0], "y_pred": [74.0, 82.0, 380.0, 410.0]})
    out = apply_gate(p, {"winter_wheat"})
    # wheat: national anomaly +3 removed, district difference (+4 vs +2) kept
    assert np.allclose(out[:2], [71.0, 79.0])
    assert np.allclose(out[2:], [380.0, 410.0])


def test_indicators_from_daily_weather():
    dates = pd.date_range("2000-10-01", "2002-12-31")
    rows = []
    for lk in ("A", "B"):
        rows.append(pd.DataFrame({"landkreis": lk, "date": dates, "precipitation_height": 1.0,
                                  "cwb": -1.0, "temperature_air_max_2m": 31.0, "swroot7": 0.3}))
    ind = national_indicators(pd.concat(rows))
    assert ind.loc[2001, "rain_ond_prev"] == 92                         # Oct-Dec 2000 → 2001
    assert 2000 not in ind.index or np.isnan(ind.loc[2000, "rain_ond_prev"])   # no autumn before 2000
    assert ind.loc[2001, "rain_jj"] == 61 and ind.loc[2001, "heat_jj"] == 61
    assert ind.loc[2001, "rain_harvest"] == 32 and ind.loc[2001, "tmax_jj"] == 31.0

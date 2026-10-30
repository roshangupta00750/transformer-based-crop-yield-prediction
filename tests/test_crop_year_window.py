"""Proofs for the crop-year window (paper/preregistration_2026-09-26.md §4.1, §4.4).

A synthetic frame with a known answer: every daily row's feature ``x`` encodes
its own date and crop, and each crop-year has a distinct yield, so any
mis-ordering, wrong-year yield or wrong-source row is detectable exactly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.dataset import (apply_phenology_window, autumn_complete_keys,
                              prepare_modeling_frame)

PHEN = {"winter_wheat": {"sowing_doy": 280, "harvest_doy": 220},
        "silage_maize": {"sowing_doy": 120, "harvest_doy": 270}}
CFG = {"phenology": PHEN, "crops": list(PHEN), "quality": {"min_sequence_coverage": 0.9}}
CROP_OFFSET = {"winter_wheat": 0.0, "silage_maize": 0.5}


def yield_of(lk, crop, year):
    return 1000 * (lk == "L2") + 100 * (crop == "silage_maize") + year - 2000


def make_frame(drop=()):
    rows = []
    for lk in ("L1", "L2"):
        for crop in PHEN:
            for year in (2019, 2020, 2021):
                if (lk, crop, year) in drop:
                    continue
                for d in pd.date_range(f"{year}-01-01", f"{year}-12-31"):
                    rows.append({"landkreis": lk, "crop": crop, "year": year,
                                 "date": d, "doy": d.dayofyear,
                                 "x": d.toordinal() + CROP_OFFSET[crop],
                                 "yield_dt_per_ha": yield_of(lk, crop, year)})
    return pd.DataFrame(rows)


def sample(df, lk, crop, year):
    return df[(df.landkreis == lk) & (df.crop == crop) & (df.year == year)]


def test_calendar_window_unchanged():
    w = apply_phenology_window(make_frame(), PHEN, window="calendar")
    ww = sample(w, "L1", "winter_wheat", 2020)
    assert ww["doy"].min() == 1 and ww["doy"].max() == 220 and len(ww) == 220
    sm = sample(w, "L1", "silage_maize", 2020)
    assert sm["doy"].min() == 120 and sm["doy"].max() == 270 and len(sm) == 151


def test_crop_year_winter_spans_sowing_to_harvest_in_date_order():
    w = apply_phenology_window(make_frame(), PHEN, window="crop_year")
    s = sample(w, "L1", "winter_wheat", 2020)
    # 2019 is not a leap year: DOY 280 = 7 Oct 2019; harvest DOY 220 of 2020 (leap) = 7 Aug
    assert s["date"].iloc[0] == pd.Timestamp("2019-10-07")
    assert s["date"].iloc[-1] == pd.Timestamp("2020-08-07")
    assert len(s) == (365 - 280 + 1) + 220
    assert s["date"].is_monotonic_increasing
    assert (s["date"].diff().dropna() == pd.Timedelta(days=1)).all()   # no gaps


def test_every_row_carries_the_crop_years_own_yield():
    w = apply_phenology_window(make_frame(), PHEN, window="crop_year")
    for (lk, crop, year), g in w.groupby(["landkreis", "crop", "year"]):
        assert (g["yield_dt_per_ha"] == yield_of(lk, crop, year)).all(), (lk, crop, year)


def test_autumn_rows_come_from_same_crop_previous_year():
    w = apply_phenology_window(make_frame(), PHEN, window="crop_year")
    s = sample(w, "L1", "winter_wheat", 2020)
    autumn = s[s["date"].dt.year == 2019]
    assert np.allclose(autumn["x"], [d.toordinal() + 0.0 for d in autumn["date"]])


def test_fallback_uses_another_crop_same_district_same_dates():
    df = make_frame(drop={("L2", "winter_wheat", 2019)})
    w = apply_phenology_window(df, PHEN, window="crop_year")
    s = sample(w, "L2", "winter_wheat", 2020)
    autumn = s[s["date"].dt.year == 2019]
    assert len(autumn) == 365 - 280 + 1
    assert np.allclose(autumn["x"], [d.toordinal() + 0.5 for d in autumn["date"]])  # maize rows
    assert (s["yield_dt_per_ha"] == yield_of("L2", "winter_wheat", 2020)).all()


def test_no_row_after_harvest_date():
    w = apply_phenology_window(make_frame(), PHEN, window="crop_year")
    for (lk, crop, year), g in w.groupby(["landkreis", "crop", "year"]):
        harvest = pd.Timestamp(f"{year}-01-01") + pd.Timedelta(days=PHEN[crop]["harvest_doy"] - 1)
        assert g["date"].max() <= harvest


def test_spring_crops_identical_in_both_windows():
    df = make_frame()
    a = apply_phenology_window(df, PHEN, window="calendar")
    b = apply_phenology_window(df, PHEN, window="crop_year")
    ka = a[a.crop == "silage_maize"].sort_values(["landkreis", "year", "date"]).reset_index(drop=True)
    kb = b[b.crop == "silage_maize"].sort_values(["landkreis", "year", "date"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(ka, kb)


def test_common_sample_gives_identical_crop_years_in_both_windows():
    df = make_frame()
    a = prepare_modeling_frame(df, CFG, window="calendar", common_sample=True)
    b = prepare_modeling_frame(df, CFG, window="crop_year", common_sample=True)
    keys = lambda f: set(zip(f.landkreis, f.crop, f.year))
    assert keys(a) == keys(b)
    # 2019 winter wheat has no 2018 autumn in the frame -> excluded in BOTH
    assert ("L1", "winter_wheat", 2019) not in keys(a)
    assert ("L1", "silage_maize", 2019) in keys(a)


def test_full_autumn_cannot_hide_a_winter_gap():
    # L1 winter wheat 2020: remove 30 days of Jan-Feb 2020 (190/220 < 90%),
    # autumn 2019 complete (86 + 190 = 276/306 would pass a total-only check)
    df = make_frame()
    gap = ((df.landkreis == "L1") & (df.crop == "winter_wheat") & (df.year == 2020)
           & (df.doy.between(20, 49)))
    df = df.loc[~gap]
    a = prepare_modeling_frame(df, CFG, window="calendar", common_sample=True)
    b = prepare_modeling_frame(df, CFG, window="crop_year", common_sample=True)
    k = ("L1", "winter_wheat", 2020)
    assert k not in set(zip(a.landkreis, a.crop, a.year))
    assert k not in set(zip(b.landkreis, b.crop, b.year))


def test_autumn_completeness_threshold():
    ok = autumn_complete_keys(make_frame(), PHEN, 0.9)
    assert ("L1", "winter_wheat", 2020) in ok and ("L1", "winter_wheat", 2019) not in ok


def test_rejects_unknown_window():
    with pytest.raises(ValueError):
        apply_phenology_window(make_frame(), PHEN, window="weekly")

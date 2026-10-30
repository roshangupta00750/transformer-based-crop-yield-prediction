"""Tests for v7 recency (paper/preregistration_2026-10-01_v7.md §2)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.dataset import (YIELD_LAG_COLUMNS, FeaturePipeline, FeatureSpec,
                              add_yield_lags, build_yield_history)


def _yields():
    rows = [("A", "winter_wheat", y, 70.0 + (y - 2000)) for y in range(2000, 2012)]   # +1 dt/ha per year
    rows += [("B", "winter_wheat", y, 80.0) for y in range(2009, 2012)]               # short record
    return pd.DataFrame(rows, columns=["landkreis", "crop", "year", "yield_dt_per_ha"])


def _pipe(hist_end=2010):
    y = _yields()
    p = FeaturePipeline(FeatureSpec(past_observed=[], static=[], known_future=[]), trend_test="yearly")
    p.recent_baseline_ = True
    p.yield_hist_ = build_yield_history(y[y.year <= hist_end])
    return p.fit(y[y.year <= hist_end - 1])


def test_recent_baseline_uses_only_previous_five_years():
    p = _pipe()
    # 2011: window 2006-2010 -> mean 78 (+ trend x (2011 - 2008) if a trend is kept)
    slope = p.target_trend_by_crop_.get("winter_wheat", 0.0)
    assert p.baseline_yield("A", "winter_wheat", 2011) == pytest.approx(78.0 + slope * 3)
    # the 2011 actual (81) is not in the history, so it cannot move the baseline
    assert 2011 not in p.yield_hist_[("A", "winter_wheat")]


def test_short_record_falls_back_to_long_term_baseline():
    p = _pipe()
    # B has only 2009-2010 before 2011 (< 3 years) -> v4 baseline (training mean)
    assert p.baseline_yield("B", "winter_wheat", 2011) == pytest.approx(
        p.target_mean_.get(("B", "winter_wheat"), p.target_mean_by_crop_["winter_wheat"])
        + p.target_trend_by_crop_.get("winter_wheat", 0.0) * (2011 - p.trend_ref_year_))


def test_lag_features():
    hist = build_yield_history(_yields())
    df = pd.DataFrame({"landkreis": ["A", "A"], "crop": "winter_wheat", "year": 2008, "date": ["d1", "d2"]})
    out = add_yield_lags(df, hist)
    m = np.mean([73, 74, 75, 76, 77])                       # 2003-2007
    assert out["ylag1_rel"].iloc[0] == pytest.approx(77 / m - 1)
    assert out["ylag3_rel"].iloc[0] == pytest.approx(75 / m - 1)
    assert out["yrecent_vs_all"].iloc[0] == pytest.approx(m / np.mean(range(70, 78)) - 1)
    assert len(out) == 2 and set(YIELD_LAG_COLUMNS) <= set(out.columns)


def test_lags_ignore_the_target_year_and_later():
    y = _yields(); hist = build_yield_history(y)
    hist_cut = build_yield_history(y[y.year < 2008])
    df = pd.DataFrame({"landkreis": ["A"], "crop": "winter_wheat", "year": 2008, "date": ["d"]})
    pd.testing.assert_frame_equal(add_yield_lags(df, hist), add_yield_lags(df, hist_cut))


def test_save_load_roundtrip(tmp_path):
    p = _pipe()
    p.save(tmp_path / "p.json")
    q = FeaturePipeline.load(tmp_path / "p.json")
    assert q.recent_baseline_ and q.yield_hist_ == p.yield_hist_
    assert q.baseline_yield("A", "winter_wheat", 2011) == p.baseline_yield("A", "winter_wheat", 2011)

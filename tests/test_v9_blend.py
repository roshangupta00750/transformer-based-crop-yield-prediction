"""Tests for the v9 blended baseline (paper/preregistration_2026-10-03_v9.md §2)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.dataset import FeaturePipeline, FeatureSpec, baseline_components, build_yield_history


def _pipe(df, hist_end):
    p = FeaturePipeline(FeatureSpec(past_observed=[], static=[], known_future=[]), trend_test="yearly")
    p.recent_baseline_ = True; p.baseline_blend_ = True
    p.yield_hist_ = build_yield_history(df[df.year <= hist_end])
    return p.fit(df[df.year <= hist_end - 1])


def _panel(kind, seed=0):
    rng = np.random.default_rng(seed); rows = []
    for d in range(60):
        level = 70.0
        for y in range(2000, 2021):
            if kind == "shift" and y % 6 == 0:
                level += rng.normal(0, 8)          # lasting level shifts -> recent level persists
            yv = level + rng.normal(0, 2) if kind == "shift" else 70 + rng.normal(0, 6)  # iid noise
            rows.append((f"D{d}", "winter_wheat", y, yv))
    return pd.DataFrame(rows, columns=["landkreis", "crop", "year", "yield_dt_per_ha"])


def test_weight_high_when_levels_persist_low_when_noise():
    w_shift = _pipe(_panel("shift"), 2020).recent_weight_by_crop_["winter_wheat"]
    w_noise = _pipe(_panel("noise"), 2020).recent_weight_by_crop_["winter_wheat"]
    assert w_shift > 0.6 and w_noise < 0.3


def test_baseline_is_the_blend():
    df = _panel("noise"); p = _pipe(df, 2020)
    h = p.yield_hist_[("D0", "winter_wheat")]; w = p.recent_weight_by_crop_["winter_wheat"]
    lt, b5 = baseline_components(h, 2021, p.target_trend_by_crop_.get("winter_wheat", 0.0))
    assert p.baseline_yield("D0", "winter_wheat", 2021) == pytest.approx(w * b5 + (1 - w) * lt)


def test_components_use_only_earlier_years():
    h = {2000: 60.0, 2001: 70.0, 2002: 80.0, 2003: 90.0, 2004: 1000.0}
    assert baseline_components(h, 2004) == (pytest.approx(75.0), pytest.approx(75.0))


def test_weight_ignores_years_after_training():
    df = _panel("shift"); a = _pipe(df, 2016).recent_weight_by_crop_
    df2 = df.copy(); df2.loc[df2.year > 2016, "yield_dt_per_ha"] = 1.0
    assert _pipe(df2, 2016).recent_weight_by_crop_ == a


def test_roundtrip(tmp_path):
    p = _pipe(_panel("shift"), 2020); p.save(tmp_path / "p.json")
    q = FeaturePipeline.load(tmp_path / "p.json")
    assert q.recent_weight_by_crop_ == p.recent_weight_by_crop_
    assert q.baseline_yield("D3", "winter_wheat", 2021) == pytest.approx(p.baseline_yield("D3", "winter_wheat", 2021))


def test_reliable_series_filter():
    from src.data.dataset import reliable_series
    rows = [("A", "oats", y) for y in range(2000, 2010)] + [("B", "oats", y) for y in (2000, 2001, 2005)]
    df = pd.DataFrame(rows, columns=["landkreis", "crop", "year"])
    out = reliable_series(df)                      # n = 10 years -> need max(5, 6) = 6
    assert set(out.landkreis) == {"A"}

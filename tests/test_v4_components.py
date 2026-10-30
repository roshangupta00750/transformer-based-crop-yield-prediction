"""Tests for v4: trend test on yearly means, fine-tuning from saved weights
(paper/preregistration_2026-09-29_v4.md)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from src.data.dataset import FeaturePipeline, FeatureSpec


def _yields(shift_early: float, n_districts: int = 200, seed: int = 0) -> pd.DataFrame:
    """Flat yields; the first three years are low in EVERY district together
    (a shared national dip, like wheat 2000-2002), plus district noise."""
    rng = np.random.default_rng(seed)
    rows = []
    for y in range(2000, 2018):
        common = rng.normal(0, 3) + (shift_early if y <= 2002 else 0.0)
        for d in range(n_districts):
            rows.append((f"D{d}", "winter_wheat", y, 75 + common + rng.normal(0, 4)))
    return pd.DataFrame(rows, columns=["landkreis", "crop", "year", "yield_dt_per_ha"])


def _fit(df, test):
    spec = FeatureSpec(past_observed=[], static=[], known_future=[])
    return FeaturePipeline(spec, trend_test=test).fit(df)


def test_yearly_test_ignores_a_shared_early_dip():
    df = _yields(shift_early=-7.0)
    assert _fit(df, "district_year").target_trend_by_crop_["winter_wheat"] > 0   # old rule: spurious trend
    assert _fit(df, "yearly").target_trend_by_crop_["winter_wheat"] == 0.0      # one point per year


def test_yearly_test_keeps_a_real_trend():
    df = _yields(shift_early=0.0)
    df["yield_dt_per_ha"] += 2.0 * (df["year"] - 2000)        # +2 dt/ha/yr, like sugar beet
    slope = _fit(df, "yearly").target_trend_by_crop_["winter_wheat"]
    assert 1.5 < slope < 2.5


def test_finetune_starts_from_given_weights():
    from src.models.itransformer import ITransformerModel
    cfg = dict(d_model=16, n_heads=2, n_layers=1, d_ff=32, dropout=0.0, learning_rate=1e-3)
    batch = {"x_past": torch.randn(8, 20, 3), "x_future": torch.randn(8, 20, 2),
             "x_static": torch.randn(8, 4), "y": torch.randn(8)}
    loader = [batch]
    a = ITransformerModel(cfg).fit(loader, None, max_epochs=1)
    init = {k: v.clone() for k, v in a.net.state_dict().items()}
    b = ITransformerModel(cfg).fit(loader, None, max_epochs=0, init_state=init)
    for k, v in b.net.state_dict().items():
        assert torch.equal(v, init[k]), k

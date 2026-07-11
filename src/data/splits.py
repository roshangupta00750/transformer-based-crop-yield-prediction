"""Train/val/test split logic — temporal and spatial holdouts."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SplitIndices:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def make_temporal_split(
    df: pd.DataFrame,
    train_end_year: int,
    val_years: list[int],
    test_years: list[int],
    year_col: str = "year",
) -> SplitIndices:
    """Hold out future years for testing — the only honest split for yield."""
    train = df[df[year_col] <= train_end_year].index.to_numpy()
    val = df[df[year_col].isin(val_years)].index.to_numpy()
    test = df[df[year_col].isin(test_years)].index.to_numpy()
    return SplitIndices(train=train, val=val, test=test)


def make_spatial_split(
    df: pd.DataFrame,
    held_out_landkreise: list[str],
    landkreis_col: str = "landkreis",
) -> SplitIndices:
    """
    Hold out specific Landkreise entirely (leave-one-out generalization test).
    Caller is responsible for choosing which to hold out per fold.
    """
    held_out = set(held_out_landkreise)
    test_mask = df[landkreis_col].isin(held_out)
    test = df[test_mask].index.to_numpy()
    train_pool = df[~test_mask]
    # Use last available year as val by default
    val_year = train_pool["year"].max()
    val = train_pool[train_pool["year"] == val_year].index.to_numpy()
    train = train_pool[train_pool["year"] != val_year].index.to_numpy()
    return SplitIndices(train=train, val=val, test=test)


def loo_landkreis_iter(df: pd.DataFrame, sample_size: int | None, rng_seed: int = 42):
    """Yield SplitIndices one Landkreis at a time for spatial LOO sweep."""
    landkreise = df["landkreis"].unique().tolist()
    if sample_size is not None and sample_size < len(landkreise):
        rng = np.random.default_rng(rng_seed)
        landkreise = list(rng.choice(landkreise, size=sample_size, replace=False))
    for lk in landkreise:
        yield lk, make_spatial_split(df, [lk])

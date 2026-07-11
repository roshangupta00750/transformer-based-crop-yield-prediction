"""Adapter from our unified parquet → pytorch-forecasting TimeSeriesDataSet.

Series are one per ``(landkreis, crop)`` spanning all years. ``time_idx``
is a cumulative day counter across years within each series. This keeps
the categorical encoder vocabulary closed (no unknown IDs at test time)
and lets the model leverage multi-year history.

The yield target is broadcast across all days of a given year, so within
a series ``yield_dt_per_ha`` is piecewise-constant: one value for all
2003 days, another for 2004, etc. The model learns to recover that yearly
constant from the preceding ``max_encoder_length`` days of weather +
satellite + soil features.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class TFTColumns:
    """Defines column roles for pytorch-forecasting."""
    target: str = "yield_dt_per_ha"
    group_ids: tuple[str, ...] = ("series_id",)
    time_idx: str = "time_idx"
    static_categoricals: tuple[str, ...] = ("landkreis", "crop")
    static_reals: tuple[str, ...] = ()                # filled from soil columns
    time_varying_known_reals: tuple[str, ...] = (
        "time_idx", "sin_doy", "cos_doy",
    )
    time_varying_unknown_reals: tuple[str, ...] = ()  # filled from weather + sat


def prepare_tft_dataframe(
    df: pd.DataFrame,
    soil_cols: list[str],
    weather_sat_cols: list[str],
    target_col: str = "yield_dt_per_ha",
) -> tuple[pd.DataFrame, TFTColumns]:
    """Return (annotated_df, TFTColumns) ready to feed ``TimeSeriesDataSet``.

    Mutations
      - ``series_id`` (string) — one per ``(landkreis, crop)``, spans all years
      - ``time_idx`` (int)     — cumulative day counter within each series
      - ``year_idx`` (int)     — 0..N year ordinal within each series, used to
                                  align yearly yield values
    """
    df = df.copy()
    df["series_id"] = (
        df["landkreis"].astype(str) + "__" + df["crop"].astype(str)
    )
    df = df.sort_values(["series_id", "year", "doy"]).reset_index(drop=True)
    df["time_idx"] = df.groupby("series_id").cumcount().astype(np.int64) + 1

    # Broadcast yield within each (series, year) — was already done upstream
    # but keep the invariant explicit here too.
    df[target_col] = (df.groupby(["series_id", "year"])[target_col]
                        .transform("first"))

    cols = TFTColumns(
        target=target_col,
        static_reals=tuple(soil_cols),
        time_varying_unknown_reals=tuple(weather_sat_cols),
    )
    return df, cols


def build_timeseries_dataset(
    df: pd.DataFrame,
    cols: TFTColumns,
    max_encoder_length: int = 180,
    max_prediction_length: int = 1,
):
    """Wrap into a `pytorch_forecasting.TimeSeriesDataSet`."""
    from pytorch_forecasting import TimeSeriesDataSet
    from pytorch_forecasting.data import GroupNormalizer
    from pytorch_forecasting.data.encoders import NaNLabelEncoder

    # add_nan=True on each categorical encoder so test-time series IDs that
    # didn't appear during training don't blow up with KeyError.
    categorical_encoders = {
        c: NaNLabelEncoder(add_nan=True)
        for c in (list(cols.group_ids) + list(cols.static_categoricals))
    }

    return TimeSeriesDataSet(
        df,
        time_idx=cols.time_idx,
        target=cols.target,
        group_ids=list(cols.group_ids),
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        static_categoricals=list(cols.static_categoricals),
        static_reals=list(cols.static_reals),
        time_varying_known_reals=list(cols.time_varying_known_reals),
        time_varying_unknown_reals=list(cols.time_varying_unknown_reals),
        target_normalizer=GroupNormalizer(groups=list(cols.group_ids), transformation="softplus"),
        categorical_encoders=categorical_encoders,
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,
    )

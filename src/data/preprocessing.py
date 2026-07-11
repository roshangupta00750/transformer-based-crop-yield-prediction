"""Project-wide preprocessing — gap fill, scaling, sliding windows."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def gap_fill_timeseries(
    df: pd.DataFrame,
    by: list[str],
    date_col: str = "date",
    method: str = "linear",
) -> pd.DataFrame:
    """Reindex to a contiguous daily index per group and interpolate."""
    out = []
    for keys, g in df.groupby(by):
        g = g.set_index(date_col).sort_index()
        full = pd.date_range(g.index.min(), g.index.max(), freq="D")
        g = g.reindex(full).interpolate(method=method, limit_direction="both")
        g.index.name = date_col
        g = g.reset_index()
        for k, v in zip(by, keys if isinstance(keys, tuple) else (keys,)):
            g[k] = v
        out.append(g)
    return pd.concat(out, ignore_index=True)


def fit_scaler(df: pd.DataFrame, columns: list[str]) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(df[columns].to_numpy(np.float32))
    return scaler


def apply_scaler(
    df: pd.DataFrame, columns: list[str], scaler: StandardScaler
) -> pd.DataFrame:
    df = df.copy()
    df[columns] = scaler.transform(df[columns].to_numpy(np.float32))
    return df

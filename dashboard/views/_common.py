"""Shared loaders, labels, colours and map helper for the six-page dashboard.

Colours follow one rule across every page: a model keeps the same colour
everywhere (transformers take the first three categorical slots), the
historical average is always a neutral grey reference, yield magnitude is a
single-hue blue ramp, and "better / worse than the historical average" is a
blue <-> red diverging scale with a grey midpoint.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pyarrow.parquet as pq
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import METRICS_DIR, PROCESSED, PROJECT_ROOT  # noqa: E402

CROPS = ["winter_wheat", "winter_barley", "rye", "triticale",
         "winter_rapeseed", "summer_barley", "oats", "silage_maize",
         "potatoes", "sugar_beet"]
CROP_LABEL = {c: c.replace("_", " ").capitalize() for c in CROPS}

TRANSFORMERS = ["itransformer", "patchtst", "tft"]
BASELINES = ["xgboost", "lstm", "dlinear"]
ALL_MODELS = TRANSFORMERS + BASELINES
MODEL_LABEL = {"itransformer": "iTransformer", "patchtst": "PatchTST",
               "tft": "TFT", "xgboost": "XGBoost", "lstm": "LSTM",
               "dlinear": "DLinear"}
MODEL_COLOR = {"itransformer": "#2a78d6", "patchtst": "#eb6834",
               "tft": "#1baf7a", "xgboost": "#eda100", "lstm": "#e87ba4",
               "dlinear": "#008300"}

MUTED = "#898781"
NO_DATA = "#e1e0d9"
SEQ_BLUE = [[0.0, "#cde2fb"], [0.17, "#9ec5f4"], [0.33, "#6da7ec"],
            [0.5, "#3987e5"], [0.67, "#256abf"], [0.83, "#184f95"],
            [1.0, "#0d366b"]]
# Negative = lower error than the historical average = better = blue.
DIVERGING = [[0.0, "#2a78d6"], [0.5, "#f0efec"], [1.0, "#e34948"]]

PREDICTIONS_PATH = METRICS_DIR / "test_predictions.csv"
SHAPEFILE = PROJECT_ROOT / "data" / "raw" / "geo" / "NUTS_RG_01M_2021_4326_LEVL_3.shp"


@st.cache_data(show_spinner=False)
def load_yields() -> pd.DataFrame:
    """One row per (crop, district, year) with the observed yield."""
    t = pq.read_table(
        PROCESSED, columns=["crop", "landkreis", "year", "yield_dt_per_ha"]
    ).to_pandas()
    t = t.dropna(subset=["yield_dt_per_ha"])
    return t.groupby(["crop", "landkreis", "year"], as_index=False)[
        "yield_dt_per_ha"].first()


@st.cache_data(show_spinner=False)
def load_test_predictions() -> pd.DataFrame:
    """Per-district test predictions from the current checkpoints.

    Written by scripts/23_export_test_predictions.py. Empty if not yet run.
    """
    if not PREDICTIONS_PATH.exists():
        return pd.DataFrame()
    d = pd.read_csv(PREDICTIONS_PATH)
    d["error"] = d["y_pred"] - d["y_true"]
    d["abs_error"] = d["error"].abs()
    d["hist_abs_error"] = (d["y_hist"] - d["y_true"]).abs()
    return d


def reproducible_pairs(preds: pd.DataFrame, mc: pd.DataFrame,
                       tol: float = 0.01) -> set[tuple[str, str]]:
    """(model, crop) pairs whose exported predictions reproduce the official RMSE.

    The official per-crop results were computed on the GPU cluster (XGBoost 3.3,
    pandas 3.0, numpy 2.5, torch 2.6). A pair whose re-prediction on this
    machine misses that RMSE by more than ``tol`` (relative) is left off the
    per-district views, so a map can never disagree with the summary charts.
    """
    if preds.empty or mc.empty:
        return set()
    local = (preds.assign(se=(preds["y_pred"] - preds["y_true"]) ** 2)
                  .groupby(["model", "crop"])["se"].mean() ** 0.5)
    official = (mc.drop_duplicates(["model", "crop"])
                  .set_index(["model", "crop"])["rmse"])
    both = pd.concat([local.rename("local"), official.rename("official")],
                     axis=1).dropna()
    ok = (both["local"] / both["official"] - 1).abs() <= tol
    return set(both.index[ok])


def error_vs_history(mc: pd.DataFrame, clim: pd.DataFrame) -> pd.DataFrame:
    """Per (model, crop): RMSE in dt/ha and % difference vs historical average."""
    if mc.empty or clim.empty:
        return pd.DataFrame()
    m = mc.drop_duplicates(["model", "crop"])[["model", "crop", "rmse"]]
    h = clim[["crop", "rmse"]].rename(columns={"rmse": "hist_rmse"})
    d = m.merge(h, on="crop")
    d["pct_vs_hist"] = (d["rmse"] / d["hist_rmse"] - 1) * 100
    return d


@st.cache_data(show_spinner=False)
def load_polygons():
    """German NUTS-3 polygons as GeoJSON plus a (landkreis, name) table."""
    if not SHAPEFILE.exists():
        return None, pd.DataFrame(columns=["landkreis", "name"])
    import geopandas as gpd
    g = gpd.read_file(SHAPEFILE)
    g = g[g["CNTR_CODE"] == "DE"][["NUTS_ID", "NUTS_NAME", "geometry"]].copy()
    g["geometry"] = g.geometry.simplify(0.005, preserve_topology=True)
    g = g.rename(columns={"NUTS_ID": "landkreis", "NUTS_NAME": "name"})
    return json.loads(g.to_json()), g[["landkreis", "name"]].reset_index(drop=True)


def district_map(df: pd.DataFrame, value_col: str, *, label: str,
                 colorscale, zmid: float | None = None,
                 zrange: tuple[float, float] | None = None,
                 height: int = 560, value_fmt: str = ".1f") -> go.Figure | None:
    """Choropleth of one value per district. Districts without data stay grey."""
    gj, names = load_polygons()
    if gj is None:
        return None
    d = names.merge(df[["landkreis", value_col]], on="landkreis", how="left")
    has = d[value_col].notna()

    fig = go.Figure()
    # Base layer: every district in neutral grey, so gaps read as "no data".
    fig.add_trace(go.Choropleth(
        geojson=gj, featureidkey="properties.landkreis",
        locations=d["landkreis"], z=np.zeros(len(d)),
        colorscale=[[0, NO_DATA], [1, NO_DATA]], showscale=False,
        marker_line_width=0.3, marker_line_color="#fcfcfb",
        text=d["name"], hovertemplate="%{text}<br>no data<extra></extra>",
    ))
    kw = {}
    if zrange is not None:
        kw.update(zmin=zrange[0], zmax=zrange[1])
    if zmid is not None:
        kw.update(zmid=zmid)
    fig.add_trace(go.Choropleth(
        geojson=gj, featureidkey="properties.landkreis",
        locations=d.loc[has, "landkreis"], z=d.loc[has, value_col],
        colorscale=colorscale, marker_line_width=0.3,
        marker_line_color="#fcfcfb",
        colorbar=dict(title=dict(text=label, side="right"), thickness=10,
                      len=0.6, x=0.92),
        text=d.loc[has, "name"],
        hovertemplate="%{text}<br>" + label + ": %{z:" + value_fmt
                      + "}<extra></extra>",
        **kw,
    ))
    fig.update_geos(visible=False, projection_type="mercator",
                    lonaxis_range=[5.7, 15.2], lataxis_range=[47.2, 55.1])
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=0, b=0),
                      paper_bgcolor="rgba(0,0,0,0)", geo_bgcolor="rgba(0,0,0,0)")
    return fig

"""Geographic page — choropleth of actual vs. predicted yield per Landkreis."""
from __future__ import annotations

import sys
from pathlib import Path

import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_folium import st_folium

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _model_template import _predict_test_set
from utils import (CHECKPOINTS_DIR, MODELS, PROJECT_ROOT, divider,
                   landkreis_name_map, load_unified, page_header)

st.set_page_config(page_title="Geographic — Crop Yield", layout="wide")
page_header("Geographic view",
            "Yield and model predictions across the ~400 German Landkreise.")

# --------------------------------------------------------------------------- #
# Load polygons
# --------------------------------------------------------------------------- #
@st.cache_data(ttl=3600)
def _load_polygons() -> gpd.GeoDataFrame:
    shp = PROJECT_ROOT / "data" / "raw" / "geo" / "NUTS_RG_01M_2021_4326_LEVL_3.shp"
    g = gpd.read_file(shp)
    g = g[g["CNTR_CODE"] == "DE"][["NUTS_ID", "NUTS_NAME", "geometry"]]
    return g.rename(columns={"NUTS_ID": "landkreis", "NUTS_NAME": "name"})


polys = _load_polygons()
if polys.empty:
    st.error("NUTS-3 shapefile missing. Run `scripts/00_build_geo.py`.")
    st.stop()

# --------------------------------------------------------------------------- #
# Controls
# --------------------------------------------------------------------------- #
df = load_unified()
yrs = sorted(df["year"].unique().tolist()) if not df.empty else []
crops = sorted(df["crop"].unique().tolist()) if not df.empty else ["winter_wheat"]

c1, c2, c3 = st.columns(3)
with c1:
    crop = st.selectbox("Crop", crops)
with c2:
    year = st.select_slider("Year", yrs, value=int(yrs[-1])) if yrs else 2023
with c3:
    overlay = st.selectbox(
        "Map overlay",
        ["Actual yield", "Model prediction", "Residual (pred − actual)"],
    )

available_models = [m for m in MODELS
                    if (CHECKPOINTS_DIR / f"{m}.pipeline.json").exists()]
model = st.selectbox("Model (used for prediction / residual overlays)",
                     available_models or MODELS,
                     index=available_models.index("xgboost")
                     if "xgboost" in available_models else 0)

# --------------------------------------------------------------------------- #
# Build per-Landkreis value column for the chosen overlay
# --------------------------------------------------------------------------- #
@st.cache_data(ttl=120)
def _yield_year(crop: str, year: int) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["landkreis", "value"])
    y = (df[(df["crop"] == crop) & (df["year"] == year)]
         .groupby("landkreis")["yield_dt_per_ha"].first()
         .reset_index().rename(columns={"yield_dt_per_ha": "value"}))
    return y


if overlay == "Actual yield":
    values = _yield_year(crop, year)
    label = f"Actual yield (dt/ha), {crop}, {year}"
else:
    # Need predictions for the requested year. Our test set covers 2022/2023.
    if year not in {2022, 2023}:
        st.warning("Predictions are only computed on the 2022–2023 test split. "
                   "Pick one of those years for prediction / residual overlays.")
        st.stop()
    preds, err = _predict_test_set(model, crop)
    if err:
        st.error(err["error"])
        st.stop()
    sub = preds[preds["year"] == year]
    col = "y_pred" if overlay == "Model prediction" else "residual"
    values = sub.groupby("landkreis")[col].first().reset_index().rename(
        columns={col: "value"})
    label = (f"{model.upper()} {('prediction' if overlay == 'Model prediction' else 'residual')}"
             f" (dt/ha), {crop}, {year}")

joined = polys.merge(values, on="landkreis", how="left")

# --------------------------------------------------------------------------- #
# Folium choropleth
# --------------------------------------------------------------------------- #
center = [51.1657, 10.4515]
m = folium.Map(location=center, zoom_start=6, tiles="cartodbpositron")

# Color scale: diverging if residual, sequential otherwise
if overlay.startswith("Residual"):
    abs_max = float(np.nanmax(np.abs(joined["value"]))) if joined["value"].notna().any() else 1.0
    abs_max = max(abs_max, 1e-3)
    vmin, vmax = -abs_max, abs_max
    fill_color = "RdYlGn_r"
else:
    vmin = float(joined["value"].min(skipna=True)) if joined["value"].notna().any() else 0.0
    vmax = float(joined["value"].max(skipna=True)) if joined["value"].notna().any() else 100.0
    fill_color = "YlGn"

if joined["value"].notna().any():
    folium.Choropleth(
        geo_data=joined.__geo_interface__,
        data=joined,
        columns=["landkreis", "value"],
        key_on="feature.properties.landkreis",
        fill_color=fill_color, fill_opacity=0.75, line_opacity=0.2,
        legend_name=label, nan_fill_color="lightgrey", nan_fill_opacity=0.4,
        bins=8 if not overlay.startswith("Residual") else 7,
    ).add_to(m)

# Tooltip layer
folium.GeoJson(
    joined.__geo_interface__,
    style_function=lambda f: {"color": "white", "weight": 0.4, "fillOpacity": 0},
    tooltip=folium.features.GeoJsonTooltip(
        fields=["name", "landkreis", "value"],
        aliases=["Landkreis", "NUTS-3", label.split('(')[0].strip()],
        localize=True, sticky=False, labels=True,
        max_width=300,
    ),
).add_to(m)

st_folium(m, width=None, height=620, returned_objects=[])

# --------------------------------------------------------------------------- #
# Top / bottom 10 districts
# --------------------------------------------------------------------------- #
divider("Top and bottom 10 Landkreise")
non_null = joined.dropna(subset=["value"])[["landkreis", "name", "value"]]
if not non_null.empty:
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown(f"**Highest** ({label})")
        st.dataframe(non_null.nlargest(10, "value").reset_index(drop=True),
                     hide_index=True, width="stretch")
    with col_r:
        st.markdown(f"**Lowest**")
        st.dataframe(non_null.nsmallest(10, "value").reset_index(drop=True),
                     hide_index=True, width="stretch")

# --------------------------------------------------------------------------- #
# Year trend for selected district
# --------------------------------------------------------------------------- #
divider("Per-district year trend")
name_map = landkreis_name_map()
codes = sorted(df["landkreis"].unique().tolist()) if not df.empty else []
# Show "Name (CODE)" in the dropdown but keep the code as the actual value.
dist = st.selectbox(
    "Landkreis", codes,
    format_func=lambda c: f"{name_map.get(c, c)} ({c})",
)
if dist and not df.empty:
    dist_label = f"{name_map.get(dist, dist)} ({dist})"
    trend = (df[(df["crop"] == crop) & (df["landkreis"] == dist)]
             .groupby("year")["yield_dt_per_ha"].first().reset_index())
    fig = px.line(trend, x="year", y="yield_dt_per_ha", markers=True,
                  title=f"{dist_label} — {crop} yield (dt/ha)")
    fig.add_hline(y=trend["yield_dt_per_ha"].mean(), line_dash="dash",
                  line_color="grey",
                  annotation_text=f"mean = {trend['yield_dt_per_ha'].mean():.1f}")
    fig.update_layout(height=350)
    st.plotly_chart(fig, width="stretch")

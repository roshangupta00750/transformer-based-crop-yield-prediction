"""Geographic: one map, three layers -- actual yield, predicted yield, error.

The error layer compares the model with the historical average district by
district: blue where the model lands closer to the actual yield, red where it
lands further away. Only (model, crop) pairs whose predictions on this machine
reproduce the official results are offered, so the map never contradicts the
summary charts on the other pages.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import (ALL_MODELS, CROP_LABEL, CROPS, DIVERGING,  # noqa: E402
                     MODEL_LABEL, SEQ_BLUE, district_map, load_polygons,
                     load_test_predictions, load_yields, reproducible_pairs)
from utils import HELP, load_multicrop_summary, page_header  # noqa: E402

st.set_page_config(page_title="Geographic — Crop Yield Germany", layout="wide")
page_header("Geographic",
            "Actual yields, model predictions and model error for every district.")

yields = load_yields()
preds = load_test_predictions()
ok_pairs = reproducible_pairs(preds, load_multicrop_summary())
_, names = load_polygons()

LAYERS = ["Actual yield", "Predicted yield", "Error vs historical average"]
c1, c2, c3, c4 = st.columns([1.4, 1, 1, 1])
layer = c1.selectbox("Show", LAYERS, key="geo_layer", help=HELP["layers"])
crop = c2.selectbox("Crop", CROPS, format_func=CROP_LABEL.get, key="geo_crop")

crop_yields = yields[yields["crop"] == crop]["yield_dt_per_ha"]
yield_range = (float(np.nanpercentile(crop_yields, 2)),
               float(np.nanpercentile(crop_yields, 98))) if len(crop_yields) else None

if layer == "Actual yield":
    years = sorted(yields.loc[yields["crop"] == crop, "year"].unique())[::-1]
    year = c3.selectbox("Year", years, key="geo_year_a")
    data = yields[(yields["crop"] == crop) & (yields["year"] == year)]
    col, label, scale, zmid, zrange = "yield_dt_per_ha", "dt/ha", SEQ_BLUE, None, yield_range
    caption = (f"{CROP_LABEL[crop]}, {year}: {len(data)} districts with data. "
               "Grey = no data.")
else:
    models = [m for m in ALL_MODELS if (m, crop) in ok_pairs]
    missing = [MODEL_LABEL[m] for m in ALL_MODELS if (m, crop) not in ok_pairs]
    if not models:
        st.info("No per-district predictions for this crop yet. Run "
                "`scripts/23_export_test_predictions.py`.")
        st.stop()
    model = c4.selectbox("Model", models, format_func=MODEL_LABEL.get, key="geo_model",
                         help=HELP["models_listed"])
    p = preds[(preds["model"] == model) & (preds["crop"] == crop)]
    years = sorted(p["year"].unique())[::-1]
    year = c3.selectbox("Year", years, key="geo_year_p")
    data = p[p["year"] == year].copy()

    if layer == "Predicted yield":
        col, label, scale, zmid, zrange = "y_pred", "dt/ha", SEQ_BLUE, None, yield_range
        caption = (f"{MODEL_LABEL[model]}'s predicted {CROP_LABEL[crop].lower()} "
                   f"yield, {year}. Same colour scale as the actual-yield layer.")
    else:
        # Negative = model closer to the actual yield than the historical average.
        data["gain"] = data["abs_error"] - data["hist_abs_error"]
        p_gain = (p["abs_error"] - p["hist_abs_error"]).abs()
        lim = float(np.nanpercentile(p_gain, 95)) if len(p_gain) else 1.0
        col, label, scale, zmid, zrange = "gain", "dt/ha", DIVERGING, 0.0, (-lim, lim)
        share = float((data["gain"] < 0).mean() * 100) if len(data) else 0.0
        caption = (f"Blue = {MODEL_LABEL[model]} closer to the actual yield than "
                   f"the historical average; red = further away. Blue in "
                   f"{share:.0f}% of districts ({year}).")
    if missing:
        caption += (" Not shown per district yet: " + ", ".join(missing) +
                    " (predictions on this machine don't match the official "
                    "results).")

map_col, side_col = st.columns([1.7, 1], gap="large")
fig = district_map(data, col, label=label, colorscale=scale, zmid=zmid,
                   zrange=zrange, height=640)
if fig is None:
    st.info("District boundaries not found (data/raw/geo).")
    st.stop()
with map_col:
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.caption(caption)

# --------------------------------------------------------------------------- #
# Best and worst districts for the layer on screen
# --------------------------------------------------------------------------- #
t = data[["landkreis", col]].merge(names, on="landkreis", how="left")
t["District"] = t["name"].fillna(t["landkreis"])
t = t.dropna(subset=[col])
if col == "gain":
    top_title, bottom_title = "Model helps most", "Model hurts most"
    top, bottom = t.nsmallest(5, col), t.nlargest(5, col)
    value_name = "vs historical (dt/ha)"
else:
    top_title, bottom_title = "Highest", "Lowest"
    top, bottom = t.nlargest(5, col), t.nsmallest(5, col)
    value_name = "dt/ha"

for title, frame in ((top_title, top), (bottom_title, bottom)):
    with side_col:
        st.markdown(f"##### {title}",
                    help=HELP["gain"] if col == "gain" else HELP["dt_ha"])
        st.dataframe(frame[["District", col]].rename(columns={col: value_name})
                     .style.format({value_name: "{:+.1f}" if col == "gain" else "{:.1f}"}),
                     width="stretch", hide_index=True)

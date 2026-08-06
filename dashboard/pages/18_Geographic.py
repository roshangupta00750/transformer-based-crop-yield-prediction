"""Retired — merged into the Spatial view.

This page and `11_Spatial.py` loaded the same NUTS-3 shapefile through a
byte-identical `_load_polygons()` and rendered the same choropleth. Everything
that lived here (actual yield / prediction / residual overlays, the top-bottom
tables and the per-district year trend) is now an overlay on the single map in
`11_Spatial.py`, alongside the per-district error and LOO views.

Kept as a stub only because `dashboard/nav.py` still registers it; removing the
entry there makes this file deletable.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import page_header

st.set_page_config(page_title="Geographic — Crop Yield", layout="wide")
page_header("Geographic view — merged",
            "Moved into the Spatial view under Findings.")

st.info(
    "Actual yield, model predictions and residuals are now overlays on the "
    "single map in **Spatial**, together with per-district RMSE/MAE and the "
    "leave-one-Landkreis-out views."
)
try:
    st.page_link("pages/11_Spatial.py", label="Open the Spatial view", icon=":material/map:")
except Exception:  # noqa: BLE001 — page_link path form varies with the nav entrypoint
    st.write("Open **Findings -> Spatial Holdout** in the sidebar.")

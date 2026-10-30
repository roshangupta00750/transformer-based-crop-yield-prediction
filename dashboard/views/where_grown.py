"""Where crops are grown: crop area, share of farmland and yield level per district (agricultural census)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _results import (CROPS, HELP, MODEL, SEQ_BLUE, crop_label, district_map, header, load_table,  # noqa: E402
                      missing, plain_layout)

A = load_table("crop_area.csv")
if A.empty:
    missing("crop_area.csv")
    st.stop()

tot = A.groupby("crop").area_ha.sum().sort_values(ascending=False)
header("Where crops are grown",
       f"Germany's 10 study crops cover about <b>{tot.sum() / 1e6:.1f} million hectares</b>. Winter wheat alone covers "
       f"{tot.get('winter_wheat', 0) / 1e6:.1f} million. Each crop has its own heartland: sugar beet and wheat on the "
       "fertile loess soils, rye on the sandy soils of the north-east, silage maize near livestock regions. Choose a "
       "crop to see where it grows and how much it yields.")

k = st.columns(5)
for i, c in enumerate(tot.index[:5]):
    k[i].metric(crop_label(c), f"{tot[c] / 1e3:,.0f}k ha", help="Crop area in the latest agricultural census "
                                                                "(2020; earlier census where 2020 is missing).")
crop = st.selectbox("Crop", [c for c in CROPS if c in set(A.crop)], format_func=crop_label, key="wg_crop")
a = A[A.crop == crop].copy()

c1, c2 = st.columns(2)
with c1:
    st.markdown("##### Crop area by district")
    fig = district_map(a.assign(v=a.area_ha / 1e3), "v", label="1,000 ha", colorscale=SEQ_BLUE,
                       zrange=(0, float(np.nanpercentile(a.area_ha / 1e3, 97))), height=480, value_fmt=".1f")
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.caption("Hectares of the crop per district (agricultural census). Grey = not reported.")
with c2:
    st.markdown("##### Share of farmland")
    fig = district_map(a, "share_of_farmland", label="% of farmland", colorscale=SEQ_BLUE,
                       zrange=(0, float(np.nanpercentile(a.share_of_farmland, 97))), height=480, value_fmt=".1f")
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.caption("Crop area as a share of the district's arable land. Large shares = the crop's heartland.")

c1, c2 = st.columns(2)
with c1:
    st.markdown("##### Average yield by district", help=HELP["dt_ha"])
    y = a.dropna(subset=["yield_2016_25"])
    fig = district_map(y, "yield_2016_25", label="dt/ha", colorscale=SEQ_BLUE,
                       zrange=(float(np.nanpercentile(y.yield_2016_25, 3)), float(np.nanpercentile(y.yield_2016_25, 97))),
                       height=480, value_fmt=".1f")
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.caption("Mean official yield 2016–2025. " + HELP["dt_ha"])
with c2:
    st.markdown("##### Area by state")
    s = a.groupby("state").area_ha.sum().sort_values()
    fig = go.Figure(go.Bar(y=s.index, x=s.values / 1e3, orientation="h", marker_color=MODEL,
                           text=[f"{v / 1e3:,.0f}k" for v in s.values], textposition="outside", cliponaxis=False))
    plain_layout(fig, 480, xaxis=dict(title="1,000 ha"), showlegend=False)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

st.markdown("##### Top districts")
t = a.sort_values("area_ha", ascending=False).head(15).copy()
t["area_ha"] = t.area_ha.map(lambda x: f"{x:,.0f}")
t["production_t"] = t.production_t.map(lambda x: "–" if x != x else f"{x:,.0f}")
st.dataframe(t[["district", "state", "area_ha", "share_of_farmland", "yield_2016_25", "production_t", "census_year"]],
             hide_index=True, width="stretch", height=38 * 16 + 3, column_config={
                 "district": "District", "state": "State",
                 "area_ha": st.column_config.TextColumn("Crop area (ha)"),
                 "share_of_farmland": st.column_config.NumberColumn("Share of farmland", format="%.1f%%"),
                 "yield_2016_25": st.column_config.NumberColumn("Avg yield 2016–25 (dt/ha)", format="%.1f"),
                 "production_t": st.column_config.TextColumn("Production (t/year)"),
                 "census_year": st.column_config.NumberColumn("Census", format="%d")})
st.caption("Source: Duden, Nacke & Offermann (2024), Scientific Data 11:95 — district crop areas from the German "
           "agricultural census, harmonised to 2020 district boundaries (CC-BY 4.0). Production = area × average "
           "yield 2016–2025, an approximation.")

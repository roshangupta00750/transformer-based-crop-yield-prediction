"""Weather and yield: which weather drives which crop (national, 2003-2025, descriptive)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _results import (BAD, CROPS, MODEL, ROOT, WEATHER_LABEL, crop_label, header, missing,  # noqa: E402
                      plain_layout)

MET = ROOT / "results" / "metrics"
if not (MET / "rca_national_weather.csv").exists():
    missing("../metrics/rca_national_weather.csv")
    st.stop()
W = pd.read_csv(MET / "rca_national_weather.csv").set_index("year")
A = pd.read_csv(MET / "rca_national_anomaly.csv").set_index("year")
crops = [c for c in CROPS if c in A.columns]
corr = pd.DataFrame({c: {w: W[w].corr(A[c]) for w in W.columns} for c in crops})
CFG = {"displayModeBar": False}

header("Weather and yield",
       "Which weather makes a good or a bad harvest? Each crop's national yield swing (how far a year was above or below "
       "its trend, 2003–2025) compared with seasonal weather averaged over Germany. <b>Summer rain dominates the "
       "summer crops</b> (silage maize, potatoes, sugar beet). The cereals and rapeseed react more weakly and to other "
       "seasons, such as heat in June–July or rain in the autumn before, which is why they are harder to estimate.")

st.markdown("##### Which weather matters for which crop")
c1, c2 = st.columns([1, 1])
with c1:
    best = pd.DataFrame({"crop": crops, "signal": [corr[c].abs().idxmax() for c in crops],
                         "r": [corr[c][corr[c].abs().idxmax()] for c in crops]}).sort_values("r")
    fig = go.Figure(go.Bar(y=[crop_label(c) for c in best.crop], x=best.r, orientation="h",
                           marker_color=[MODEL if v > 0 else BAD for v in best.r],
                           text=[WEATHER_LABEL.get(w, w) for w in best.signal], textposition="auto",
                           hovertemplate="%{y}: r = %{x:+.2f}<br>%{text}<extra></extra>"))
    fig.add_vline(x=0, line_color="#b9b8b2", line_width=1)
    plain_layout(fig, 420, xaxis=dict(range=[-1, 1], title="correlation with the crop's strongest weather signal"),
                 showlegend=False)
    st.plotly_chart(fig, use_container_width=True, config=CFG)
    st.caption("For each crop, the one weather signal that moves its national yield most, 2003–2025.")
with c2:
    pick = st.selectbox("All signals for one crop", crops, format_func=crop_label, key="we_all")
    r = corr[pick].sort_values()
    fig = go.Figure(go.Bar(y=[WEATHER_LABEL.get(w, w) for w in r.index], x=r.values, orientation="h",
                           marker_color=[MODEL if v > 0 else BAD for v in r.values],
                           text=[f"{v:+.2f}" for v in r.values], textposition="outside", cliponaxis=False))
    fig.add_vline(x=0, line_color="#b9b8b2", line_width=1)
    plain_layout(fig, 370, xaxis=dict(range=[-1, 1], title="correlation r"), showlegend=False)
    st.plotly_chart(fig, use_container_width=True, config=CFG)
st.caption("Blue = more of this weather goes with better yields, red = with worse yields. |r| above about 0.4 is "
           "unlikely to be chance with 23 years; small values mean weather alone does not explain the crop.")

st.markdown("---")
st.markdown("##### One crop, one weather signal")
c1, c2 = st.columns(2)
crop = c1.selectbox("Crop", crops, format_func=crop_label, key="we_crop")
best = corr[crop].abs().idxmax()
sig = c2.selectbox("Weather signal", list(W.columns), index=list(W.columns).index(best),
                   format_func=lambda c: WEATHER_LABEL.get(c, c), key="we_sig")
d = pd.DataFrame({"x": W[sig], "y": A[crop]}).dropna()
fig = go.Figure(go.Scatter(x=d.x, y=d.y, mode="markers+text", text=d.index.astype(str), textposition="top center",
                           marker=dict(size=10, color=[BAD if v <= -10 else MODEL for v in d.y])))
if len(d) > 2:
    b = np.polyfit(d.x, d.y, 1)
    xs = np.linspace(d.x.min(), d.x.max(), 20)
    fig.add_trace(go.Scatter(x=xs, y=b[0] * xs + b[1], mode="lines", line=dict(color="#222", dash="dash")))
fig.add_hline(y=0, line_color="#bbb")
plain_layout(fig, 440, xaxis=dict(title=WEATHER_LABEL.get(sig, sig)),
             yaxis=dict(title=f"{crop_label(crop)} national yield vs trend (%)"), showlegend=False)
st.plotly_chart(fig, use_container_width=True, config=CFG)
st.caption(f"Each dot is a year. r = {corr.at[sig, crop]:+.2f}. Red dots = yield at least 10% below trend. The strongest "
           f"signal for {crop_label(crop).lower()} is preselected: {WEATHER_LABEL.get(best, best).lower()}.")

st.markdown("---")
st.markdown("##### Drought years in the weather")
fig = go.Figure()
fig.add_trace(go.Bar(x=W.index, y=W["rain_summer(JJ)"], name="Rain June–July (mm)", marker_color=MODEL, opacity=0.8))
fig.add_trace(go.Scatter(x=W.index, y=W["heatdays_JJ"], name="Hot days June–July", yaxis="y2", mode="lines+markers",
                         line=dict(color=BAD, width=2.5)))
for y in (2003, 2006, 2018, 2019, 2022):
    if y in W.index:
        fig.add_vrect(x0=y - 0.5, x1=y + 0.5, fillcolor=BAD, opacity=0.08, line_width=0)
plain_layout(fig, 380, yaxis=dict(title="rain (mm)"),
             yaxis2=dict(title="hot days", overlaying="y", side="right", showgrid=False))
st.plotly_chart(fig, use_container_width=True, config=CFG)
st.caption("Germany-wide summer rain (bars) and hot days (line). The shaded summers 2003, 2006, 2018, 2019 and 2022 "
           "combine little rain with many hot days. These are the years in which the summer crops fell sharply and in "
           "which the model gains most over simple averages.")
st.caption("Descriptive analysis on national averages; the model itself uses daily district weather for each crop's "
           "own season.")

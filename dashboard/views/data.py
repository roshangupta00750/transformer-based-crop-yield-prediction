"""Data: the German crop dataset at a glance.

Three views: how each crop's yield moved over 2003-2025, the daily weather and
satellite series of one district's season (what the models actually see), and
where the yield records exist.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pyarrow.parquet as pq
import streamlit as st
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import (CROP_LABEL, CROPS, SEQ_BLUE, load_polygons,  # noqa: E402
                     load_yields)
from utils import HELP, PROCESSED  # noqa: E402
from _results import header  # noqa: E402

BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#008300"
GRID = "#e1e0d9"

st.set_page_config(page_title="Data — Crop Yield Germany", layout="wide")
header("Data", "What goes into the model: official district yields for 10 crops (what it learns to estimate), and "
       "for every season the daily weather and satellite greenness it reads. <b>Three takeaways:</b> yields rise "
       "slowly over the years and dip in drought years, which is what the model must catch; a single season's "
       "weather is very bumpy, so the model needs many years to learn from it; and for rye, oats, potatoes and sugar "
       "beet, far fewer districts report yields since 2016, so those crops rest on less data.")


@st.cache_data(show_spinner=False)
def load_corrected_yields() -> pd.DataFrame:
    """Yields matched by official district code (the data the final model uses); falls back to the old panel."""
    f = PROCESSED.parents[1] / "interim" / "v5" / "yield.parquet"
    return pd.read_parquet(f) if f.exists() else load_yields()


yields = load_corrected_yields()
_, names = load_polygons()
name_of = dict(zip(names["landkreis"], names["name"]))


@st.cache_data(show_spinner=False)
def load_season(landkreis: str, crop: str, year: int) -> pd.DataFrame:
    """Daily series for one (district, crop, year), as the models see it."""
    t = pq.read_table(
        PROCESSED,
        columns=["date", "temperature_air_max_2m", "temperature_air_min_2m",
                 "precipitation_height", "ndvi"],
        filters=[("landkreis", "=", landkreis), ("crop", "=", crop),
                 ("year", "=", int(year))],
    ).to_pandas()
    return t.sort_values("date")


# --------------------------------------------------------------------------- #
# Yield trend per crop -- small multiples, each crop on its own scale
# --------------------------------------------------------------------------- #
st.markdown("##### Yield trend per crop", help=HELP["trend"])
g = (yields.groupby(["crop", "year"])["yield_dt_per_ha"]
           .quantile([0.25, 0.5, 0.75]).unstack().reset_index())
fig = make_subplots(rows=2, cols=5, subplot_titles=[CROP_LABEL[c] for c in CROPS],
                    vertical_spacing=0.2, horizontal_spacing=0.045)
for i, c in enumerate(CROPS):
    r, k = i // 5 + 1, i % 5 + 1
    s = g[g["crop"] == c]
    fig.add_trace(go.Scatter(
        x=np.concatenate([s["year"], s["year"][::-1]]),
        y=np.concatenate([s[0.75], s[0.25][::-1]]),
        fill="toself", fillcolor="rgba(42,120,214,0.15)", line=dict(width=0),
        hoverinfo="skip", showlegend=False), row=r, col=k)
    fig.add_trace(go.Scatter(
        x=s["year"], y=s[0.5], mode="lines", line=dict(color=BLUE, width=2),
        showlegend=False,
        hovertemplate=f"{CROP_LABEL[c]} %{{x}}: %{{y:.1f}} dt/ha<extra></extra>"),
        row=r, col=k)
fig.update_xaxes(showgrid=False, nticks=4, tickfont=dict(size=10))
fig.update_yaxes(gridcolor=GRID, tickfont=dict(size=10))
fig.update_annotations(font=dict(size=12))
fig.update_layout(height=440, margin=dict(l=10, r=10, t=30, b=10),
                  paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
st.caption("Median yield across districts (dt/ha); band = middle 50% of districts. Each crop has its own scale. "
           "Look for the dips, e.g. 2018 (drought) in most crops: those year-to-year swings are what the model "
           "estimates; the slow upward trend is already in the 'normal' it starts from.")

# --------------------------------------------------------------------------- #
# One district, one season
# --------------------------------------------------------------------------- #
st.markdown("---")
st.markdown("##### One district, one season",
            help=HELP["weather"] + "\n\n" + HELP["ndvi"])
districts = sorted(yields["landkreis"].unique(),
                   key=lambda lk: name_of.get(lk, lk))
# Default to a district with the full record, preferring the Würzburg region
# (Unterfranken, DE26*): Würzburg Landkreis itself has one winter-wheat year.
n_years = (yields[yields["crop"] == CROPS[0]]
           .groupby("landkreis")["year"].nunique())
full = set(n_years[n_years == n_years.max()].index)
default = next((i for i, lk in enumerate(districts)
                if lk in full and lk.startswith("DE26")),
               next((i for i, lk in enumerate(districts) if lk in full), 0))
c1, c2, c3 = st.columns(3)
lk = c1.selectbox("District", districts, index=default,
                  format_func=lambda x: f"{name_of.get(x, x)} ({x})", key="da_lk")
crop = c2.selectbox("Crop", CROPS, format_func=CROP_LABEL.get, key="da_crop")
yrs = sorted(yields.loc[(yields["landkreis"] == lk) & (yields["crop"] == crop),
                        "year"].unique())[::-1]
if not yrs:
    st.info("No yield records for this district and crop.")
else:
    year = c3.selectbox("Year", yrs, key="da_year")
    hist = yields[(yields["landkreis"] == lk) & (yields["crop"] == crop)]
    this = float(hist.loc[hist["year"] == year, "yield_dt_per_ha"].iloc[0])
    avg = float(hist["yield_dt_per_ha"].mean())

    m1, m2, _ = st.columns([1, 1, 2])
    m1.metric(f"Yield {year}", f"{this:.1f} dt/ha",
              delta=f"{(this / avg - 1) * 100:+.1f}% vs district average",
              help=HELP["dt_ha"])
    m2.metric("District average", f"{avg:.1f} dt/ha",
              help=f"Mean yield of this district over its {len(hist)} recorded "
                   "years. " + HELP["dt_ha"])

    s = load_season(lk, crop, int(year))
    if s.empty:
        st.info("No daily series stored for this season (this view uses the original data set, which has fewer "
                "districts).")
    else:
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                            vertical_spacing=0.07,
                            subplot_titles=["Temperature (°C)",
                                            "Precipitation (mm/day)", "NDVI"])
        fig.add_trace(go.Scatter(x=s["date"], y=s["temperature_air_max_2m"],
                                 name="max", line=dict(color=ORANGE, width=1.5)),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=s["date"], y=s["temperature_air_min_2m"],
                                 name="min", line=dict(color=BLUE, width=1.5)),
                      row=1, col=1)
        fig.add_trace(go.Bar(x=s["date"], y=s["precipitation_height"],
                             name="precipitation", marker_color=BLUE,
                             showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter(x=s["date"], y=s["ndvi"], name="NDVI",
                                 line=dict(color=GREEN, width=2),
                                 showlegend=False), row=3, col=1)
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(gridcolor=GRID)
        fig.update_annotations(font=dict(size=12))
        fig.update_layout(height=560, margin=dict(l=10, r=10, t=30, b=10),
                          legend=dict(orientation="h", y=1.08, x=0),
                          hovermode="x unified", bargap=0.1,
                          paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.caption(f"Daily weather and greenness in {name_of.get(lk, lk)}, calendar year {year}. The model "
                   f"reads each crop's own season from it: for {CROP_LABEL[crop].lower()} from sowing "
                   "(for autumn-sown crops, the previous autumn) up to harvest. NDVI (greenness) moves in steps "
                   "because the satellite delivers one cloud-free picture every 16 days; sudden low values in "
                   "winter usually mean snow, clouds or bare fields.")

# --------------------------------------------------------------------------- #
# Coverage
# --------------------------------------------------------------------------- #
st.markdown("---")
st.markdown("##### Data coverage", help=HELP["coverage"])
cov = (yields.groupby(["crop", "year"])["landkreis"].nunique().unstack().reindex(CROPS))
y0, y1, ym = int(cov.columns.min()), int(cov.columns.max()), 2010
cv = cov[[c for c in (y0, ym, y1) if c in cov.columns]].sort_values(y1)
fig = go.Figure()
for crop_, r in cv.iterrows():
    fig.add_trace(go.Scatter(x=[r.max(), r.min()], y=[CROP_LABEL[crop_]] * 2, mode="lines", showlegend=False,
                             line=dict(color="#d4d3cd", width=2), hoverinfo="skip"))
for yr, color, sym in ((y0, "#9ec5f4", "circle-open"), (ym, BLUE, "circle"), (y1, "#e34948", "circle")):
    if yr in cv.columns:
        fig.add_trace(go.Scatter(x=cv[yr], y=[CROP_LABEL[c] for c in cv.index], mode="markers", name=str(yr),
                                 marker=dict(color=color, size=11, symbol=sym, line=dict(width=2, color=color)),
                                 hovertemplate="%{y}: %{x} districts<extra>" + str(yr) + "</extra>"))
fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)",
                  plot_bgcolor="rgba(0,0,0,0)", legend=dict(orientation="h", y=1.08, x=0),
                  xaxis=dict(title="districts with an official yield", gridcolor=GRID))
st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
st.caption(f"Number of districts with an official yield record in {y0}, {ym} and {y1}. Since about 2016 far fewer "
           "districts publish yields for rye, oats, potatoes and sugar beet (e.g. potatoes: 308 districts in 2010, "
           "114 in 2025). Wheat, barley, maize and rapeseed stay well covered.")

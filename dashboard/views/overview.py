"""Overview: what the dataset covers and how well the transformers predict.

One screen: four tiles, a yield map, and one chart answering "is each
transformer more accurate than simply using each district's historical
average?" Accuracy is error in dt/ha, not R2 -- R2 here credits the model with
the district average it was handed for free.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import (CROP_LABEL, CROPS, MODEL_COLOR, MODEL_LABEL, MUTED,  # noqa: E402
                     SEQ_BLUE, TRANSFORMERS, district_map, error_vs_history,
                     load_yields)
from utils import (HELP, load_climatology_baseline,  # noqa: E402
                   load_multicrop_summary, page_header)

st.set_page_config(page_title="Transformer-based Crop Yield Prediction", layout="wide")
page_header("Transformer-based Crop Yield Prediction using Multivariate "
            "Time Series Datasets",
            "10 major German crops · 282 districts · 2003–2025. Transformer "
            "models predict district yields from daily weather and satellite data.")

yields = load_yields()
ev = error_vs_history(load_multicrop_summary(), load_climatology_baseline())

# --------------------------------------------------------------------------- #
# Tiles
# --------------------------------------------------------------------------- #
k1, k2, k3, k4 = st.columns(4)
k1.metric("Districts", f"{yields['landkreis'].nunique()}", help=HELP["districts"])
k2.metric("Crops", f"{yields['crop'].nunique()}", help=HELP["crops"])
k3.metric("Years", f"{int(yields['year'].min())}–{int(yields['year'].max())}",
          help=HELP["years"])
if not ev.empty:
    per_model = ev.groupby("model")["pct_vs_hist"].mean().sort_values()
    best = per_model.index[0]
    k4.metric("Best model vs historical average", MODEL_LABEL.get(best, best),
              delta=f"{per_model.iloc[0]:+.1f}% error", delta_color="inverse",
              help=HELP["pct"] + " Averaged over the 10 crops.")

st.markdown("")

# --------------------------------------------------------------------------- #
# Yield map  |  transformer accuracy by crop
# --------------------------------------------------------------------------- #
left, right = st.columns([1.05, 1], gap="large")

with left:
    st.markdown("##### Yield by district", help=HELP["dt_ha"])
    c1, c2 = st.columns(2)
    crop = c1.selectbox("Crop", CROPS, format_func=CROP_LABEL.get, key="ov_crop")
    years = sorted(yields.loc[yields["crop"] == crop, "year"].unique())
    year = c2.selectbox("Year", years[::-1], key="ov_year")
    sel = yields[(yields["crop"] == crop) & (yields["year"] == year)]
    fig = district_map(sel, "yield_dt_per_ha", label="dt/ha",
                       colorscale=SEQ_BLUE, height=520)
    if fig is None:
        st.info("District boundaries not found (data/raw/geo).")
    else:
        st.plotly_chart(fig, use_container_width=True,
                        config={"displayModeBar": False})
        st.caption(f"{CROP_LABEL[crop]}, {year}: {len(sel)} districts with data. "
                   "Grey = no data.")

with right:
    st.markdown("##### Transformer error vs historical average",
                help=HELP["error"] + "\n\n" + HELP["historical"])
    if ev.empty:
        st.info("Model results not found.")
    else:
        d = ev[ev["model"].isin(TRANSFORMERS)].copy()
        order = [c for c in CROPS if c in set(d["crop"])][::-1]
        fig = go.Figure()
        fig.add_vline(x=0, line_width=2, line_color=MUTED)
        for m in TRANSFORMERS:
            s = d[d["model"] == m].set_index("crop").reindex(order)
            fig.add_trace(go.Scatter(
                x=s["pct_vs_hist"], y=[CROP_LABEL[c] for c in order],
                mode="markers", name=MODEL_LABEL[m],
                marker=dict(size=11, color=MODEL_COLOR[m],
                            line=dict(width=2, color="#fcfcfb")),
                customdata=s[["rmse", "hist_rmse"]].to_numpy(),
                hovertemplate=(f"<b>{MODEL_LABEL[m]}</b> · %{{y}}<br>"
                               "%{x:+.1f}% vs historical average<br>"
                               "error %{customdata[0]:.1f} dt/ha "
                               "(historical %{customdata[1]:.1f})<extra></extra>"),
            ))
        fig.add_annotation(x=0, y=1.02, yref="paper", text="historical average",
                           showarrow=False, font=dict(size=11, color=MUTED))
        fig.update_layout(
            height=560, margin=dict(l=10, r=10, t=30, b=10),
            xaxis=dict(title="% difference in error (left = more accurate)",
                       ticksuffix="%", zeroline=False, gridcolor="#e1e0d9"),
            yaxis=dict(showgrid=False),
            legend=dict(orientation="h", y=-0.12, x=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True,
                        config={"displayModeBar": False})
        st.caption("Test years 2024–25. Dots left of the grey line beat simply "
                   "using each district's historical average.")

        with st.expander("Table", expanded=False):
            t = (d.pivot(index="crop", columns="model", values="rmse")
                   .reindex(CROPS)[TRANSFORMERS])
            t.insert(0, "historical average",
                     d.drop_duplicates("crop").set_index("crop")["hist_rmse"])
            t = t.rename(index=CROP_LABEL, columns=MODEL_LABEL)
            st.dataframe(t.style.format("{:.1f}"), width="stretch", column_config={
                "historical average": st.column_config.NumberColumn(help=HELP["historical"]),
                **{MODEL_LABEL[m]: st.column_config.NumberColumn(help=HELP["error"])
                   for m in TRANSFORMERS}})
            st.caption("Error (RMSE) in dt/ha, lower is better.")

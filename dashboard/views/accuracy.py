"""Accuracy: how close the forecasts are, by crop, year and district."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _results import (AVG5, AVG_LONG, CROPS, DIVERGING, HELP, MAIN_LABEL, MODEL, crop_label,  # noqa: E402
                      district_map, header, load_summary, load_table, missing, plain_layout)

S = load_summary()
R = load_table("rows.parquet")
C = load_table("accuracy_by_crop.csv")
Y = load_table("accuracy_by_year.csv")
if R.empty:
    missing("rows.parquet")
    st.stop()

_gain = (C.accuracy_model - C.accuracy_avg5).set_axis(C.crop)
_ybetter = int((Y.accuracy_model > Y.accuracy_avg5).sum())
header("Accuracy",
       f"Over {S['rows']:,} district estimates from 2010 to 2025, the final model {MAIN_LABEL} is typically "
       f"<b>{S['mape_model']:.1f}% away</b> from the real yield (accuracy {S['accuracy_district']:.1f}%), against "
       f"{S['mape_avg5']:.1f}% for the 5-year average: about {S['error_reduction_vs_avg5']:.0f}% smaller errors. "
       f"<b>{S['within10']:.0f}%</b> of forecasts land within ±10% of the real yield "
       f"({S['within10_avg5']:.0f}% for the 5-year average). It is more accurate for "
       f"<b>{int((_gain > 0).sum())} of {len(_gain)} crops</b> (gains from {_gain.min():+.1f} to {_gain.max():+.1f} "
       f"points) and in "
       f"{_ybetter} of {len(Y)} years. In calm years the 5-year average is about as good; in drought "
       "years it fails and the model is far better.")

t1, t2, t3, t4, t5 = st.tabs(["By crop", "By year", "Map", "Forecast vs real", "One district"])

with t1:
    st.markdown("##### Accuracy per crop", help=HELP["accuracy"])
    st.caption("Dots show each crop's accuracy; the further right, the better. The gap between the blue dot and the "
               "grey tick is the model's gain over the 5-year average.")
    d = C.set_index("crop").reindex([c for c in CROPS if c in set(C.crop)]).reset_index()
    d = d.iloc[::-1]                       # fixed crop order, first crop at the top
    fig = go.Figure()
    for col, name, color in (("accuracy_avg_long", "Long-term average", AVG_LONG),
                             ("accuracy_avg5", "5-year average", AVG5),
                             ("accuracy_model", MAIN_LABEL, MODEL)):
        fig.add_trace(go.Scatter(y=[crop_label(c) for c in d.crop], x=d[col], name=name, mode="markers",
                                 marker=dict(color=color, size=13 if col == "accuracy_model" else 10,
                                             symbol="circle" if col == "accuracy_model" else "line-ns-open",
                                             line=dict(width=3, color=color)),
                                 hovertemplate="%{y}: %{x:.1f}%<extra>" + name + "</extra>"))
    plain_layout(fig, 480, xaxis=dict(ticksuffix="%", title="accuracy (100% − typical error)"))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    t = d.assign(crop=d.crop.map(crop_label), gain=d.accuracy_model - d.accuracy_avg5)[
        ["crop", "accuracy_model", "accuracy_avg5", "gain", "within10_model", "within10_avg5", "n"]]
    st.dataframe(t, hide_index=True, width="stretch", column_config={
        "crop": "Crop",
        "accuracy_model": st.column_config.NumberColumn(MAIN_LABEL, format="%.1f%%", help=HELP["accuracy"]),
        "accuracy_avg5": st.column_config.NumberColumn("5-year average", format="%.1f%%", help=HELP["avg5"]),
        "gain": st.column_config.NumberColumn("Gain (pts)", format="%+.1f"),
        "within10_model": st.column_config.NumberColumn("Within ±10% (model)", format="%.0f%%", help=HELP["within10"]),
        "within10_avg5": st.column_config.NumberColumn("Within ±10% (5-yr avg)", format="%.0f%%"),
        "n": st.column_config.NumberColumn("Forecasts", format="%d")})
    st.caption("Accuracy differs by crop mainly because some crops vary more from year to year: "
               "oats, potatoes and summer barley swing most with spring and summer weather. "
               "Winter wheat is the most stable, so the 5-year average is already hard to beat there.")

with t2:
    st.markdown("##### Accuracy per year", help=HELP["unseen"])
    fig = go.Figure()
    fig.add_vrect(x0=2009.5, x1=2015.5, fillcolor=MODEL, opacity=0.06, line_width=0,
                  annotation_text="clean test 2010–2015", annotation_position="top left")
    for col, name, color, dash in (("accuracy_avg5", "5-year average", AVG5, "dot"),
                                   ("accuracy_model", MAIN_LABEL, MODEL, "solid")):
        fig.add_trace(go.Scatter(x=Y.year, y=Y[col], name=name, mode="lines+markers",
                                 line=dict(color=color, width=2.5, dash=dash), marker=dict(size=7)))
    plain_layout(fig, 420, yaxis=dict(ticksuffix="%", title="accuracy"), hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    worst = Y.loc[Y.accuracy_model.idxmin()]
    best = Y.loc[Y.accuracy_model.idxmax()]
    st.caption(f"Best year {int(best.year)} ({best.accuracy_model:.1f}%), hardest year {int(worst.year)} "
               f"({worst.accuracy_model:.1f}%, 5-year average {worst.accuracy_avg5:.1f}%). In drought "
               "years the averages fail badly, which is where the model earns its keep. "
               + HELP["hindcast"])

with t3:
    st.markdown("##### Where is the model better than the 5-year average?")
    a, b = st.columns([1, 1])
    crop = a.selectbox("Crop", ["all"] + CROPS, format_func=lambda c: "All crops" if c == "all" else crop_label(c),
                       key="acc_map_crop")
    years = b.select_slider("Years", options=list(range(2010, 2026)), value=(2010, 2025), key="acc_map_years")
    m = R[R.year.between(*years)]
    if crop != "all":
        m = m[m.crop == crop]
    g = m.groupby("landkreis").agg(model=("err_predicted", lambda e: 100 - e.abs().mean()),
                                   avg5=("err_avg5", lambda e: 100 - e.abs().mean()),
                                   n=("actual", "size")).reset_index()
    g["gain"] = g.model - g.avg5
    view = st.radio("Show", ["Gain over 5-year average (points)", "Accuracy of our model (%)"],
                    horizontal=True, key="acc_map_view")
    if view.startswith("Gain"):
        lim = float(np.nanpercentile(g.gain.abs(), 95)) if len(g) else 5
        fig = district_map(g.assign(v=-g.gain), "v", label="pts", colorscale=DIVERGING, zmid=0,
                           zrange=(-lim, lim), value_fmt="+.1f")
        if fig is not None:
            fig.data[-1].hovertemplate = "%{text}<br>gain %{customdata:+.1f} pts<extra></extra>"
            fig.data[-1].customdata = g.set_index("landkreis").reindex(fig.data[-1].locations).gain.values
            fig.data[-1].colorbar.title.text = "blue = model better"
            fig.data[-1].showscale = False
        caption = (f"Blue = model more accurate than the 5-year average; red = less. "
                   f"Model better in {int((g.gain > 0).sum())} of {len(g)} districts.")
    else:
        from _results import SEQ_BLUE
        fig = district_map(g, "model", label="%", colorscale=SEQ_BLUE,
                           zrange=(float(np.nanpercentile(g.model, 5)), float(np.nanpercentile(g.model, 95))))
        caption = "Darker = more accurate. Grey = no data."
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.caption(caption)

with t4:
    st.markdown("##### Every forecast against the real yield")
    crop = st.selectbox("Crop", CROPS, format_func=crop_label, key="acc_sc_crop")
    d = R[R.crop == crop]
    lo, hi = float(min(d.actual.min(), d.predicted.min())), float(max(d.actual.max(), d.predicted.max()))
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=d.actual, y=d.predicted, mode="markers", name="district-year",
                               marker=dict(size=5, color=MODEL, opacity=0.35),
                               text=d.district + " " + d.year.astype(str),
                               hovertemplate="%{text}<br>real %{x:.1f}, forecast %{y:.1f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="perfect forecast",
                             line=dict(color="#222", dash="dash", width=1)))
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo * 1.1, hi * 1.1], mode="lines", name="±10%",
                             line=dict(color=AVG5, dash="dot", width=1)))
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo * 0.9, hi * 0.9], mode="lines", showlegend=False,
                             line=dict(color=AVG5, dash="dot", width=1)))
    plain_layout(fig, 520, xaxis_title="real yield (dt/ha)", yaxis_title="forecast (dt/ha)")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    r = np.corrcoef(d.actual, d.predicted)[0, 1]
    st.caption(f"{len(d):,} forecasts for {crop_label(crop)}; correlation {r:.2f}; "
               f"{(d.err_predicted.abs() <= 10).mean():.0%} within ±10%. Points on the dashed line "
               "would be perfect. Forecasts are pulled towards the middle: very high and very low "
               "yields are predicted less extreme than they turn out.")

with t5:
    st.markdown("##### One district, year by year")
    a, b = st.columns(2)
    names = R.drop_duplicates("landkreis").sort_values("district")
    lk = a.selectbox("District", names.landkreis, format_func=lambda k: f"{names.set_index('landkreis').district[k]} ({k})",
                     key="acc_dist")
    crops = [c for c in CROPS if c in set(R[R.landkreis == lk].crop)]
    crop = b.selectbox("Crop", crops, format_func=crop_label, key="acc_dist_crop")
    d = R[(R.landkreis == lk) & (R.crop == crop)].sort_values("year")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d.year, y=d.avg5, name="5-year average", line=dict(color=AVG5, dash="dot")))
    fig.add_trace(go.Scatter(x=d.year, y=d.actual, name="Real yield", mode="lines+markers",
                             line=dict(color="#222", width=2.5)))
    fig.add_trace(go.Scatter(x=d.year, y=d.predicted, name="Our forecast", mode="lines+markers",
                             line=dict(color=MODEL, width=2.5)))
    plain_layout(fig, 400, yaxis_title="dt/ha", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    if len(d):
        st.caption(f"{len(d)} years. Accuracy here: model {100 - d.err_predicted.abs().mean():.1f}%, "
                   f"5-year average {100 - d.err_avg5.abs().mean():.1f}%.")
    with st.expander("Table"):
        st.dataframe(d[["year", "actual", "predicted", "avg5", "err_predicted"]].round(1), hide_index=True,
                     width="stretch", column_config={"err_predicted": st.column_config.NumberColumn(
                         "Error", format="%+.1f%%")})

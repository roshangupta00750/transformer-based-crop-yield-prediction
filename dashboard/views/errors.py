"""Error analysis: where, when and why the final model is wrong."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _results import (AVG5, BAD, CROPS, DASH, MODEL, MODEL_NAME, crop_label, header, load_summary,  # noqa: E402
                      load_table, missing, plain_layout)

S = load_summary()
R = load_table("rows.parquet")
if R.empty:
    missing("rows.parquet")
    st.stop()
I = json.loads((DASH / "insights.json").read_text()) if (DASH / "insights.json").exists() else {}
ST = load_table("error_by_state.csv")
AB = load_table("error_by_area.csv")
BY = load_table("bias_by_year.csv")
DEC = load_table("error_decomposition.csv")
PER = load_table("error_persistence_sample.csv")
CAL = load_table("calibration_sample.csv")
WM = load_table("largest_misses.csv")
CFG = {"displayModeBar": False}

header("Error analysis",
       f"Where, when and why <b>{MODEL_NAME}</b> is wrong. The typical district error is {S['mape_model']:.1f}%. Errors "
       f"are larger where a crop is rare, in a few extreme years, and in single districts that differ from their "
       f"neighbours. The worst 10% of estimates account for {I.get('worst10_share', float('nan')):.0f}% of all error.")

crop = st.selectbox("Crop (applies to the first four charts)", ["all"] + CROPS,
                    format_func=lambda c: "All crops" if c == "all" else crop_label(c), key="er_crop")
d = R if crop == "all" else R[R.crop == crop]

c1, c2 = st.columns(2)
with c1:
    st.markdown("##### Error distribution")
    e = d.err_predicted.clip(-50, 50)
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=d.err_avg5.clip(-50, 50), xbins=dict(size=2), name="5-year average",
                               marker_color=AVG5, opacity=0.5))
    fig.add_trace(go.Histogram(x=e, xbins=dict(size=2), name=MODEL_NAME, marker_color=MODEL, opacity=0.7))
    fig.add_vline(x=0, line_color="#222")
    plain_layout(fig, 330, barmode="overlay", xaxis=dict(title="error (%)  — left: too low, right: too high"),
                 yaxis=dict(title="estimates"))
    st.plotly_chart(fig, use_container_width=True, config=CFG)
    st.caption(f"Median error {np.median(d.err_predicted):+.1f}%; half of all estimates are within "
               f"±{np.median(d.err_predicted.abs()):.1f}%. The model's errors are more tightly packed around zero than "
               "the 5-year average's.")
with c2:
    st.markdown("##### Error by state")
    s = d.groupby("state").agg(m=("err_predicted", lambda x: x.abs().mean()),
                               a=("err_avg5", lambda x: x.abs().mean())).sort_index(ascending=False)
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=s.index, x=s.a, mode="markers", name="5-year average",
                             marker=dict(color=AVG5, size=10, symbol="line-ns-open", line=dict(width=3, color=AVG5))))
    fig.add_trace(go.Scatter(y=s.index, x=s.m, mode="markers", name=MODEL_NAME, marker=dict(color=MODEL, size=11)))
    plain_layout(fig, 330, xaxis=dict(title="typical error (%)"))
    st.plotly_chart(fig, use_container_width=True, config=CFG)
    st.caption("Errors are largest in Brandenburg and Saxony-Anhalt, where the 5-year average is also worst: yields "
               "there swing most from year to year (dry, sandy soils). Lower Saxony, Hesse and Saxony are the most "
               "predictable.")

c1, c2 = st.columns(2)
with c1:
    st.markdown("##### Error vs crop area")
    if not AB.empty:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=AB["bin"], y=AB.error_avg5, name="5-year average", marker_color=AVG5))
        fig.add_trace(go.Bar(x=AB["bin"], y=AB.error_model, name=MODEL_NAME, marker_color=MODEL))
        plain_layout(fig, 330, barmode="group", yaxis=dict(title="typical error (%)"),
                     xaxis=dict(title="crop's share of the district's farmland"))
        st.plotly_chart(fig, use_container_width=True, config=CFG)
        st.caption(f"All crops. Where the crop is rare (median share {AB.share_median.iloc[0]:.2f}%), errors are "
                   f"{AB.error_model.iloc[0]:.1f}%; where it dominates ({AB.share_median.iloc[-1]:.0f}%), "
                   f"{AB.error_model.iloc[-1]:.1f}%. Official yields for small areas rest on few farms and are noisy.")
with c2:
    st.markdown("##### Over- or under-estimate by year")
    b = d.groupby("year").err_predicted.mean()
    fig = go.Figure(go.Bar(x=b.index, y=b.values, marker_color=[BAD if v > 0 else MODEL for v in b.values],
                           text=[f"{v:+.1f}" for v in b.values], textposition="outside", cliponaxis=False))
    fig.add_hline(y=0, line_color="#222")
    plain_layout(fig, 330, yaxis=dict(title="average error (%)"), showlegend=False)
    st.plotly_chart(fig, use_container_width=True, config=CFG)
    big = b.abs().sort_values(ascending=False).index[:2]
    st.caption("Red = estimates too high on average (the year was worse than estimated), blue = too low (the year was "
               "better than estimated). The largest: " + ", ".join(f"{int(y)} ({b[y]:+.1f}%)" for y in sorted(big)) +
               ". Estimates are cautious, so very good years come out too low and bad years too high.")

st.markdown("---")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("##### Where the error comes from")
    if not DEC.empty:
        dd = DEC.set_index("component").share
        fig = go.Figure(go.Pie(labels=dd.index, values=dd.values, hole=0.55,
                               marker=dict(colors=[AVG5, "#6da7ec", MODEL]), textinfo="percent"))
        plain_layout(fig, 300)
        st.plotly_chart(fig, use_container_width=True, config=CFG)
        st.caption("Germany-wide year effect = the whole country misjudged in a year; state effect = one state; "
                   "single district = what neighbours don't share. The last part cannot come from weather data, which "
                   "is smooth across neighbours.")
with c2:
    st.markdown("##### Errors repeat")
    if not PER.empty:
        sp = PER.sample(min(2000, len(PER)), random_state=0)
        fig = go.Figure(go.Scattergl(x=100 * sp.prev, y=100 * sp.e_dist, mode="markers",
                                     marker=dict(size=4, color=MODEL, opacity=0.3), hoverinfo="skip"))
        fig.update_xaxes(range=[-40, 40]); fig.update_yaxes(range=[-40, 40])
        plain_layout(fig, 300, xaxis=dict(title="district error last year (%)"),
                     yaxis=dict(title="district error this year (%)"))
        st.plotly_chart(fig, use_container_width=True, config=CFG)
        st.caption(f"Correlation {I.get('persistence_r', float('nan')):+.2f} after the error memory. The memory "
                   "already removes part of this repetition; a small part remains.")
with c3:
    st.markdown("##### Cautious estimates")
    if not CAL.empty:
        sc = CAL.sample(min(2000, len(CAL)), random_state=0)
        fig = go.Figure(go.Scattergl(x=100 * sc.a_true, y=100 * sc.a_pred, mode="markers",
                                     marker=dict(size=4, color=MODEL, opacity=0.3), hoverinfo="skip"))
        fig.update_xaxes(range=[-50, 50]); fig.update_yaxes(range=[-50, 50])
        fig.add_trace(go.Scatter(x=[-50, 50], y=[-50, 50], mode="lines", line=dict(color="#222", dash="dash")))
        plain_layout(fig, 300, xaxis=dict(title="real vs normal (%)"), yaxis=dict(title="estimate vs normal (%)"),
                     showlegend=False)
        st.plotly_chart(fig, use_container_width=True, config=CFG)
        st.caption(f"The cloud is flatter than the dashed line: estimated swings are about "
                   f"{100 * I.get('sd_ratio', 0):.0f}% of the real ones. The model is right on direction more often "
                   "than on size.")

st.markdown("---")
st.markdown("##### Largest misses")
if not WM.empty:
    w = WM.assign(crop=WM.crop.map(crop_label))
    st.dataframe(w, hide_index=True, width="stretch", height=38 * 16 + 3, column_config={
        "year": st.column_config.NumberColumn("Year", format="%d"), "crop": "Crop", "district": "District",
        "state": "State", "actual": st.column_config.NumberColumn("Real (dt/ha)", format="%.1f"),
        "predicted": st.column_config.NumberColumn(f"{MODEL_NAME} (dt/ha)", format="%.1f"),
        "avg5": st.column_config.NumberColumn("5-yr avg (dt/ha)", format="%.1f"),
        "err_predicted": st.column_config.NumberColumn("Error", format="%+.0f%%")})
    st.caption("The 30 largest district errors. Most are not model failures but implausible official figures, usually "
               "for crops grown on a tiny area: e.g. silage maize in Köln reported at 45 dt/ha where 400–500 is normal, "
               "or wheat in Cottbus at 10.7 dt/ha. Filtering such records is a data-cleaning step still to do.")

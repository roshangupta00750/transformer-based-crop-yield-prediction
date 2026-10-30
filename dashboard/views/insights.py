"""Key insights: the ten most important findings, each with its number and a chart."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _results import (AVG5, BAD, CROPS, DASH, GOOD, MAIN, MODEL, MODEL_NAME, ROOT, WEATHER_LABEL, crop_label,  # noqa: E402
                      header, load_summary, load_table, missing, plain_layout)

S = load_summary()
if not S:
    missing("summary.json")
    st.stop()
I = json.loads((DASH / "insights.json").read_text()) if (DASH / "insights.json").exists() else {}
C = load_table("accuracy_by_crop.csv")
Y = load_table("accuracy_by_year.csv")
AB = load_table("error_by_area.csv")
DEC = load_table("error_decomposition.csv")
PER = load_table("error_persistence_sample.csv")
CAL = load_table("calibration_sample.csv")
WC = load_table("weather_correlation.csv")
VC = pd.read_csv(ROOT / "results/metrics/version_comparison_common.csv") \
    if (ROOT / "results/metrics/version_comparison_common.csv").exists() else pd.DataFrame()

header("Key insights",
       f"Ten things we learned from 37,383 district estimates by <b>{MODEL_NAME}</b> (the final model, {MAIN}: three transformers averaged) and from "
       "building 18 versions. Each insight states one number and shows the chart behind it.")

CFG = {"displayModeBar": False}


def insight(n: int, title: str, number: str, text: str) -> None:
    st.markdown("---")
    st.markdown(f"#### {n}. {title}")
    st.markdown(f"<div style='font-size:1.6rem;font-weight:700;color:{MODEL}'>{number}</div>", unsafe_allow_html=True)
    st.markdown(text)


# 1 ---------------------------------------------------------------------------
gain = (C.accuracy_model - C.accuracy_avg5).set_axis(C.crop)
gain = gain.reindex([c for c in CROPS if c in gain.index][::-1])       # fixed crop order, no ranking
insight(1, "It beats the simple 5-year average for every crop", f"{int((gain > 0).sum())} of {len(gain)} crops",
        f"Typical error {S['mape_model']:.1f}% vs {S['mape_avg5']:.1f}%. The gain differs by crop, from "
        f"{gain.min():+.1f} to {gain.max():+.1f} points: it is small where yields are steady and a simple average is "
        "already close, and larger where the weather moves yields a lot.")
fig = go.Figure(go.Bar(y=[crop_label(c) for c in gain.index], x=gain.values, orientation="h", marker_color=MODEL,
                       text=[f"{v:+.1f}" for v in gain.values], textposition="outside", cliponaxis=False))
plain_layout(fig, 340, xaxis=dict(title="accuracy gain over the 5-year average (percentage points)"), showlegend=False)
st.plotly_chart(fig, use_container_width=True, config=CFG)

# 2 ---------------------------------------------------------------------------
w = Y.assign(gap=Y.accuracy_model - Y.accuracy_avg5)
top = w.loc[w.gap.idxmax()]
insight(2, "The gain is biggest in drought years", f"{int(top.year)}: {top.accuracy_model:.1f}% vs {top.accuracy_avg5:.1f}%",
        "In calm years the 5-year average is about as good. When the weather breaks the pattern, the average fails "
        f"and the model holds: in {int(top.year)} the gap was {top.gap:.1f} points.")
fig = go.Figure()
fig.add_trace(go.Scatter(x=Y.year, y=Y.accuracy_avg5, name="5-year average", mode="lines+markers",
                         line=dict(color=AVG5, dash="dot")))
fig.add_trace(go.Scatter(x=Y.year, y=Y.accuracy_model, name=MODEL_NAME, mode="lines+markers", line=dict(color=MODEL)))
plain_layout(fig, 320, yaxis=dict(title="accuracy", ticksuffix="%"))
st.plotly_chart(fig, use_container_width=True, config=CFG)

# 3 ---------------------------------------------------------------------------
insight(3, "Germany-wide totals are far more reliable than single districts",
        f"{S['accuracy_national']:.1f}% national vs {S['accuracy_district']:.1f}% district",
        f"District errors partly cancel out when added up. Only {S['within10']:.0f}% of district estimates are within "
        "±10% of the real yield, but national estimates almost always are.")
fig = go.Figure(go.Bar(x=["District", "National"], y=[S["accuracy_district"], S["accuracy_national"]],
                       marker_color=[MODEL, "#184f95"], text=[f"{S['accuracy_district']:.1f}%", f"{S['accuracy_national']:.1f}%"],
                       textposition="outside", cliponaxis=False))
plain_layout(fig, 280, yaxis=dict(range=[0, 105], ticksuffix="%", title="accuracy"), showlegend=False)
st.plotly_chart(fig, use_container_width=True, config=CFG)

# 4 ---------------------------------------------------------------------------
if not DEC.empty:
    dd = DEC.set_index("component").share
    insight(4, "Half of the remaining error belongs to single districts", f"{dd.get('Single district', 0):.0f}% of the error",
            "We split the error into a Germany-wide part (a whole year misjudged), a state part and a single-district "
            "part. Weather and satellite inputs are smooth across neighbours, so they cannot explain the district part. "
            "This is why better input data (versions 13 and 14) did not improve the model.")
    fig = go.Figure(go.Bar(x=dd.values, y=dd.index, orientation="h", marker_color=[AVG5, "#6da7ec", MODEL],
                           text=[f"{v:.0f}%" for v in dd.values], textposition="outside", cliponaxis=False))
    plain_layout(fig, 240, xaxis=dict(range=[0, 70], ticksuffix="%", title="share of the squared error"), showlegend=False)
    st.plotly_chart(fig, use_container_width=True, config=CFG)

# 5 ---------------------------------------------------------------------------
if not AB.empty:
    insight(5, "Where a crop is rare, the official numbers are noisy and errors are larger",
            f"{AB.error_model.iloc[0]:.1f}% vs {AB.error_model.iloc[-1]:.1f}% error",
            "Districts where the crop covers the least farmland have much larger errors than where it dominates. The "
            "simple 5-year average shows the same pattern, so part of this is noise in the statistics themselves.")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=AB["bin"], y=AB.error_avg5, name="5-year average", marker_color=AVG5))
    fig.add_trace(go.Bar(x=AB["bin"], y=AB.error_model, name=MODEL_NAME, marker_color=MODEL))
    plain_layout(fig, 300, barmode="group", yaxis=dict(title="typical error", ticksuffix="%"),
                 xaxis=dict(title="districts grouped by the crop's share of farmland"))
    st.plotly_chart(fig, use_container_width=True, config=CFG)

# 6 ---------------------------------------------------------------------------
if not PER.empty:
    insight(6, "Errors repeat from one year to the next, which is what the error memory uses",
            f"correlation {I.get('persistence_r', float('nan')):+.2f}",
            f"A district the model over-estimated last year tends to be over-estimated again. {MODEL_NAME} corrects "
            "each estimate by part of the district's recent errors; this was the first change to pass our strict test "
            "(even after the correction, a small correlation remains).")
    fig = go.Figure(go.Scattergl(x=100 * PER.prev, y=100 * PER.e_dist, mode="markers",
                                 marker=dict(size=4, color=MODEL, opacity=0.25)))
    plain_layout(fig, 320, xaxis=dict(title="district's own error last year (%)", range=[-40, 40]),
                 yaxis=dict(title="district's own error this year (%)", range=[-40, 40]), showlegend=False)
    st.plotly_chart(fig, use_container_width=True, config=CFG)

# 7 ---------------------------------------------------------------------------
if not WC.empty:
    W = WC.set_index(WC.columns[0])
    drv = {c: (W[c].abs().idxmax(), W[c][W[c].abs().idxmax()]) for c in W.columns}
    insight(7, "Summer rain decides maize, potatoes and sugar beet; cereals depend on other things",
            "r ≈ 0.8–0.9 for summer rain",
            "Correlation of each crop's national yield swing (2003–2025) with seasonal weather. Rain in June–July "
            "explains most of the good and bad years of the summer crops. Wheat, barley, rye and rapeseed react more "
            "weakly and to different seasons (heat, the autumn before), which is why they are harder to estimate.")
    order = [c for c in CROPS if c in drv]
    fig = go.Figure(go.Bar(y=[crop_label(c) for c in order], x=[drv[c][1] for c in order], orientation="h",
                           marker_color=[GOOD if drv[c][1] > 0 else BAD for c in order],
                           text=[WEATHER_LABEL.get(drv[c][0], drv[c][0]) for c in order], textposition="auto"))
    plain_layout(fig, 380, xaxis=dict(range=[-1, 1], title="correlation with the strongest weather signal"),
                 showlegend=False)
    st.plotly_chart(fig, use_container_width=True, config=CFG)

# 8 ---------------------------------------------------------------------------
if not CAL.empty:
    insight(8, "Estimates are cautious: they are pulled towards the normal yield",
            f"about {100 * I.get('sd_ratio', 0):.0f}% of the real swing",
            f"The estimated deviations from normal are on average {100 * I.get('sd_ratio', 0):.0f}% as large as the real "
            f"ones. In bad years the model captures about {S['bad_share_captured_median']:.0f}% of the drop. This is "
            "expected from a model that minimises average error, but it means extreme years are underestimated.")
    fig = go.Figure(go.Scattergl(x=100 * CAL.a_true, y=100 * CAL.a_pred, mode="markers",
                                 marker=dict(size=4, color=MODEL, opacity=0.25)))
    fig.add_trace(go.Scatter(x=[-50, 50], y=[-50, 50], mode="lines", line=dict(color="#222", dash="dash")))
    plain_layout(fig, 340, xaxis=dict(title="real yield vs normal (%)", range=[-50, 50]),
                 yaxis=dict(title="estimate vs normal (%)", range=[-50, 50]), showlegend=False)
    st.plotly_chart(fig, use_container_width=True, config=CFG)

# 9 ---------------------------------------------------------------------------
tr = pd.Series(I.get("trend_per_decade", {}))
tr = tr.reindex([c for c in CROPS if c in tr.index][::-1])
if len(tr):
    insight(9, "Yields of winter barley and sugar beet are rising; dry-summer crops are not",
            f"{crop_label(tr.idxmax())} {tr.max():+.0f}% per decade",
            "Within-district yield trend 2010–2025. Breeding and management raise some crops; others (oats, "
            "potatoes, silage maize) lost ground in the dry summers of 2018–2022. The model starts from each "
            "district's trend, so it does not need to learn this from the weather.")
    fig = go.Figure(go.Bar(y=[crop_label(c) for c in tr.index], x=tr.values, orientation="h",
                           marker_color=[GOOD if v > 0 else BAD for v in tr.values],
                           text=[f"{v:+.1f}%" for v in tr.values], textposition="outside", cliponaxis=False))
    plain_layout(fig, 340, xaxis=dict(title="yield change per decade (%)"), showlegend=False)
    st.plotly_chart(fig, use_container_width=True, config=CFG)

# 10 --------------------------------------------------------------------------
if not VC.empty:
    v = VC.set_index("model")
    insight(10, "Better input data stopped helping; fixing the data, the district bias and combining models did",
            "18 versions, 2 passed the strict test",
            "Finer weather (v14) and crop-specific satellite data (v13) did not improve the estimates. What helped: "
            "correct district matching (v12), the error memory (v15) and averaging three transformers (v18). All "
            "versions lie within about one percentage point of each other; the simple averages are further below.")
    order = [m for m in ["v4", "v7", "v9", "v10", "v12", "v13", "v14", "v15b", "v15", "v17 TFT", "v17 PatchTST", MAIN]
             if m in v.index]
    fig = go.Figure(go.Scatter(x=order, y=[v.at[m, "2016-25"] for m in order], mode="markers+text",
                               marker=dict(size=13, color=[MODEL if m == MAIN else "#9ec5f4" for m in order]),
                               text=[f"{v.at[m, '2016-25']:.2f}" for m in order], textposition="top center"))
    fig.add_hline(y=v.at["b5", "2016-25"], line_dash="dot", line_color=AVG5, annotation_text="5-year average")
    plain_layout(fig, 300, yaxis=dict(title="accuracy 2016–2025 (%)", ticksuffix="%"), showlegend=False)
    st.plotly_chart(fig, use_container_width=True, config=CFG)

st.markdown("---")
st.caption("All insights are descriptive: they come from the evaluation of the final model, not from new "
           "pre-registered tests. Weather correlations use national averages 2003–2025.")

"""Bad years: does the model see droughts and failed harvests coming?"""
from __future__ import annotations

import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _results import (BAD, GOOD, HELP, MODEL, crop_label, header, load_summary, load_table,  # noqa: E402
                      missing, plain_layout)

S = load_summary()
N = load_table("national.csv")
if N.empty:
    missing("national.csv")
    st.stop()

W = load_table("bad_year_warnings.csv")
w = {r.warning_rule: r for r in W.itertuples()} if not W.empty else {}
w5, w10 = w.get("predicted ≥5% below normal"), w.get("predicted ≥10% below normal")
others = int(w5.warnings_in_other_years + w5.other_years_quiet) if w5 else 0

header("Bad years: droughts and failed harvests",
       f"Between 2010 and 2025 there were <b>{S['bad_years']} bad harvests</b>: a crop's national yield at least "
       f"10% below its normal level. With the rule \"warn when the estimate is at least 5% below normal\", the model "
       f"warned in <b>{w5.bad_years_warned if w5 else '–'} of them</b>, and gave {w5.warnings_in_other_years if w5 else '–'} "
       f"warnings in the {others} other crop-years. It typically estimated about <b>{S['bad_share_captured_median']:.0f}% "
       "of the real drop</b>: it sees the damage coming but underestimates its size. It cannot prevent a drought; it "
       "estimates the damage per district right after harvest.")

k = st.columns(4)
k[0].metric("Bad years warned", f"{w5.bad_years_warned if w5 else '–'} of {S['bad_years']}",
            "rule: estimate ≥5% below normal", delta_color="off", help=HELP["bad_year"])
k[1].metric("False warnings", f"{w5.warnings_in_other_years if w5 else '–'} of {others}",
            "same rule, in years that were not bad", delta_color="off")
k[2].metric("With a strict rule (≥10%)", f"{w10.bad_years_warned if w10 else '–'} of {S['bad_years']} warned",
            f"{w10.warnings_in_other_years if w10 else '–'} false warnings", delta_color="off",
            help="Warn only when the estimate is at least 10% below normal, the same threshold that defines a bad year.")
k[3].metric("Drop captured", f"~{S['bad_share_captured_median']:.0f}%", "median over the 15 bad years",
            delta_color="off", help="Estimated drop ÷ real drop. 100% = exact size of the loss.")

st.markdown("##### How often the model warns: three rules side by side")
if not W.empty:
    st.table(W.rename(columns={"warning_rule": "The model warns when its estimate is…",
                               "bad_years_warned": "Bad years warned", "bad_years_missed": "Bad years missed",
                               "warnings_in_other_years": "False warnings", "other_years_quiet": "Other years, no warning"})
             .set_index("The model warns when its estimate is…"))
    st.caption("A looser rule catches more bad years but raises more false warnings; a stricter rule the opposite. "
               "'Normal' = the district's long-term average with its trend, fitted on earlier years only. Example: "
               "winter wheat 2018 (a well-known drought year) was 5.9% below normal nationally, so it does not count "
               "as a bad year under the 10% definition; the model estimated −6.7%.")

st.markdown("---")
st.markdown("##### Every bad harvest: real drop vs predicted drop")
b = N[N.kind == "bad"].sort_values("actual_vs_normal").copy()
b["label"] = b.crop.map(crop_label) + " " + b.year.astype(str)
fig = go.Figure()
for _, r in b.iterrows():
    fig.add_trace(go.Scatter(x=[r.actual_vs_normal, r.predicted_vs_normal], y=[r.label, r.label], mode="lines",
                             line=dict(color="#c9c8c2", width=3), showlegend=False, hoverinfo="skip"))
fig.add_trace(go.Scatter(x=b.actual_vs_normal, y=b.label, mode="markers", name="Real drop",
                         marker=dict(color=BAD, size=12), customdata=b.event,
                         hovertemplate="%{y} · %{customdata}<br>real %{x:+.1f}%<extra></extra>"))
fig.add_trace(go.Scatter(x=b.predicted_vs_normal, y=b.label, mode="markers", name="Estimated",
                         marker=dict(color=MODEL, size=12, symbol="diamond"),
                         hovertemplate="%{y}<br>predicted %{x:+.1f}%<extra></extra>"))
fig.add_vline(x=0, line_color="#222", line_width=1)
fig.add_vline(x=-10, line_color=BAD, line_width=1, line_dash="dot")
plain_layout(fig, 40 * len(b) + 120, xaxis=dict(title="national yield vs normal", ticksuffix="%"),
             yaxis=dict(autorange="reversed"))
st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
st.caption("Red dot = what really happened, blue diamond = what the model estimated, both relative to the "
           "normal level. A diamond left of the black line means the estimate was below normal; the closer it is to "
           "the red dot, the better the size of the loss was estimated.")

t = b[["crop", "year", "event", "actual_vs_normal", "predicted_vs_normal", "share_captured", "detected"]].copy()
t["crop"] = t.crop.map(crop_label)
t.loc[~t.detected, "share_captured"] = None          # missed: no share of the drop to speak of
t["detected"] = t.predicted_vs_normal <= -5
t["share_captured"] = t.share_captured.map(lambda x: "missed" if x != x else f"{x:.0f}%")
st.dataframe(t, hide_index=True, width="stretch", height=38 * (len(t) + 1) + 3, column_config={
    "crop": "Crop", "year": st.column_config.NumberColumn("Year", format="%d"), "event": "Event",
    "actual_vs_normal": st.column_config.NumberColumn("Real", format="%+.1f%%"),
    "predicted_vs_normal": st.column_config.NumberColumn("Predicted", format="%+.1f%%"),
    "share_captured": st.column_config.TextColumn("Drop captured",
                                                     help="Empty = missed (forecast above normal)."),
    "detected": st.column_config.CheckboxColumn("Warned (≥5%)")})

st.markdown("---")
st.markdown("##### All years at a glance")
st.caption("Each dot is one crop in one year: real national yield vs normal (x) against the estimate (y). "
           "Points near the diagonal were estimated well; the shaded sides are the bad (left) and good (right) years. "
           "Most dots sit closer to the horizontal line than to the diagonal: the estimates are cautious in both "
           "directions, in good years as well as bad ones.")
fig = go.Figure()
fig.add_shape(type="rect", x0=-35, x1=-10, y0=-35, y1=35, fillcolor=BAD, opacity=0.06, line_width=0)
fig.add_shape(type="rect", x0=10, x1=35, y0=-35, y1=35, fillcolor=GOOD, opacity=0.06, line_width=0)
for kind, color in (("normal", "#9e9d97"), ("bad", BAD), ("good", GOOD)):
    d = N[N.kind == kind]
    fig.add_trace(go.Scatter(x=d.actual_vs_normal, y=d.predicted_vs_normal, mode="markers",
                             name={"normal": "normal year", "bad": "bad year", "good": "good year"}[kind],
                             marker=dict(color=color, size=9, opacity=0.85),
                             text=d.crop.map(crop_label) + " " + d.year.astype(str),
                             hovertemplate="%{text}<br>real %{x:+.1f}%, estimate %{y:+.1f}%<extra></extra>"))
fig.add_trace(go.Scatter(x=[-30, 30], y=[-30, 30], mode="lines", name="perfect",
                         line=dict(color="#222", dash="dash", width=1)))
plain_layout(fig, 520, xaxis=dict(title="real national yield vs normal", ticksuffix="%", range=[-32, 32]),
             yaxis=dict(title="estimate vs normal", ticksuffix="%", range=[-32, 32]))
st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

with st.expander("Limits, stated plainly", expanded=True):
    st.markdown(
        "- **Timing:** the model uses the observed harvest date and the weather up to harvest, so it is an "
        "**at-harvest damage estimate**, about 3 months before the official statistics, not a warning "
        "months ahead.\n"
        "- **Size of the loss:** it sees droughts coming but usually estimates about two-thirds of the drop, and "
        "good years are underestimated too. Estimates are cautious by design: they are pulled towards the normal.\n"
        "- **Only national events are counted here** (15 crop-years); a district-level check is still to be done.\n"
        "- **Blind spots:** rapeseed in a spring drought (2011, missed), and local events such as hail, "
        "floods or pests, which weather data at district scale cannot see.")

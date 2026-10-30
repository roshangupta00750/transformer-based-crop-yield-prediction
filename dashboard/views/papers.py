"""Compared with papers: the same tests as three published German studies."""
from __future__ import annotations

import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _results import (AVG5, MAIN, MODEL, PAPER, PAPERS_MODEL, crop_label, header, load_summary, load_table,  # noqa: E402
                      missing, plain_layout)

S = load_summary()
P = load_table("papers.csv")
V = load_table("versions.csv")
if P.empty:
    missing("papers.csv")
    st.stop()

ENS = bool((P.version == MAIN).any())          # final model's own runs on the papers' splits exist
SHOWN = MAIN if ENS else PAPERS_MODEL
OURS = f"{MAIN} (final model, without memory)" if ENS else "v12 (iTransformer, without memory)"
M = P[P.version == SHOWN].copy()
M["better_when"] = M.metric.map(lambda m: "higher" if m == "r2" else "lower")
M["label"] = M.crop.map(crop_label) + " · " + M.years.astype(str).str.replace(r" \(.*\)", "", regex=True) \
    .str.replace("2019-24 pooled incl. our validation years 2019-21, as their code", "2019-24 pooled", regex=False)
M["metric_name"] = M.metric.map({"r2": "R²", "rmse": "RMSE (dt/ha)", "mae": "MAE (dt/ha)", "mape": "MAPE (%)"})

header("Compared with published papers",
       f"We repeated the tests of three published studies on German crop yields, each time using "
       f"<b>that paper's own training and test years</b>, so the comparison is fair. "
       + (f"Shown: the three transformers of <b>YieldFormer-DE ({MAIN})</b> averaged and retrained on each paper's "
          "split, without the error memory, because these test rules freeze information at the end of training. "
          "Counted by case (paper, crop and test years) it <b>wins 5 of 8, loses 2 and ties 1</b>; counted by single "
          f"rows it is better in {int((M.verdict == 'ours better').sum())} of {len(M)}. Not a blind test: the papers' "
          "test years were development years of this project."
          if ENS else
          f"<b>The final model ({MAIN}, three transformers averaged) is still being retrained on the papers' splits;</b> "
          f"until then this page shows the iTransformer versions: v12 wins <b>{S['papers_won']} of {S['papers_total']}</b> "
          f"comparisons, and v15 (v12 + the error memory, allowed only in Srivastava's single-year tests) wins "
          f"<b>{S.get('papers_won_v15', '–')} of {S.get('papers_total_v15', '–')}</b> there."))

_won = M.groupby("paper").apply(lambda g: f"{int((g.verdict == 'ours better').sum())} of {len(g)}")
st.info("**What this means in one line:** "
        + "; ".join(f"{p.replace(' (same districts)', '').replace(' paper', '')}: {w} won" for p, w in _won.items())
        + f" ({OURS}). For winter wheat, a plain historical average is still hard to beat for everyone.", icon="💡")
with st.expander("How to read the measures (R², RMSE, MAE, MAPE)", expanded=False):
    st.markdown(
        "- **R²** (0 to 1, higher is better): how much of the difference between districts and years the estimate "
        "explains. 1 = perfect.\n"
        "- **RMSE / MAE** (dt/ha, lower is better): typical size of the error in yield units. Silage maize errors look "
        "bigger (around 55) only because maize yields are much larger (around 400 dt/ha) than wheat (around 75).\n"
        "- **MAPE** (%, lower is better): typical error in percent.\n"
        "- **Pooled**: several test years scored together.\n"
        "- **v12 vs v15**: v15 = v12 + a correction from last year's real yields. Most papers' test rules don't "
        "allow using those, so v12 is shown for them; v15 is shown where allowed (Srivastava's single-year tests).")

INFO = {
    "AgroForecast-DE (same districts)": "AgroForecast-DE: the CropFusionNet team's public follow-up benchmark for "
                                        "German districts, with two strictly unseen test years (2023–24); scored "
                                        "here on exactly their published district-years.",
    "CropFusionNet paper": "CropFusionNet (2026, Srivastava et al. at ZALF; a different study from Srivastava 2022): transformer-style model on daily climate, "
                           "soil and terrain data for German districts; test years 2022–24. Their headline score most "
                           "likely also counts years used for tuning (2019–21), so we show both versions.",
    "Srivastava 2022 CNN": "Srivastava et al. (2022, Scientific Reports): convolutional network for winter wheat in "
                           "271 German counties, one test year at a time (2017, 2018, 2019). Their yields appear to be "
                           "dry matter (~14% lower), which makes their errors look ~14% smaller than ours.",
}

for paper, g in M.groupby("paper", sort=False):
    won = int((g.verdict == "ours better").sum())
    st.markdown(f"#### {paper.replace(' (same districts)', '').replace(' paper', '')} — "
                f"{OURS} better in {won} of {len(g)}")
    st.caption(INFO.get(paper, ""))
    lost_hist = g[~g.beats_hist_avg.astype(bool)]
    if len(lost_hist):
        rows = ", ".join(sorted(set(lost_hist.crop.map(crop_label) + " " + lost_hist.years.astype(str).str[:7])))
        st.markdown(f"<span style='color:#b3261e;font-size:0.9rem'>Note: on {rows} the plain historical "
                    f"average is better than our model ({len(lost_hist)} of {len(g)} rows), even where we beat "
                    f"the paper.</span>", unsafe_allow_html=True)
    for metric, h in g.groupby("metric", sort=False):
        h = h.copy()
        key = ["paper", "crop", "years", "metric"]
        for v in ("v15", "v15b"):
            o = P[P.version == v].set_index(key)["ours_equal_protocol"]
            h[v] = [o.get(tuple(r)) for r in h[key].itertuples(index=False)]
        fig = go.Figure()
        series = [("theirs", "Published", PAPER), ("ours_equal_protocol", OURS, MODEL if ENS else "#9ec5f4")]
        if not ENS and h["v15"].notna().any():
            series.append(("v15", "YieldFormer-DE-i (v15)", MODEL))
        series += [("v15b", "v15b", "#6da7ec"), ("hist_avg", "Historical average", AVG5)]
        for col, name, color in series:
            fig.add_trace(go.Bar(x=h.label, y=h[col], name=name, marker_color=color,
                                 text=h[col].map(lambda x: "" if x != x else (f"{x:.2f}" if metric == "r2" else f"{x:.1f}")),
                                 textposition="outside"))
        better = "higher is better" if metric == "r2" else "lower is better"
        top = float(h[[c for c, _, _ in series]].max().max())
        fig.update_traces(cliponaxis=False)
        plain_layout(fig, 330, barmode="group", yaxis=dict(range=[0, top * 1.18]),
                     title=dict(text=f"{h.metric_name.iloc[0]} · {better}", font=dict(size=13)))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

st.markdown("---")
st.markdown("##### Every version, every comparison")
key = ["paper", "crop", "years", "metric"]
W = P.pivot_table(index=key, columns="version", values="ours_equal_protocol", aggfunc="first")
base = M.set_index(key)[["theirs", "hist_avg"]]
order = [v for v in ("v1", "v3", "v4", "v7", "v9", "v10", "v12", "v13", "v14", "v15b", "v15", MAIN) if v in W.columns]
T = base.join(W[order]).reset_index()
T["paper"] = T.paper.str.replace(" (same districts)", "", regex=False).str.replace(" paper", "", regex=False)
T["crop"] = T.crop.map(crop_label)
T["years"] = T.years.astype(str).str.replace(r" \(.*\)", "", regex=True).str.replace(
    "2019-24 pooled incl. our validation years 2019-21, as their code", "2019-24 pooled", regex=False)
T["metric"] = T.metric.map({"r2": "R²", "rmse": "RMSE", "mae": "MAE", "mape": "MAPE"})
T = T[["paper", "crop", "years", "metric", "theirs"] + order + ["hist_avg"]]
for c in ["theirs"] + order + ["hist_avg"]:
    T[c] = T[c].map(lambda x: "n/a" if x != x else f"{x:.3f}")
st.dataframe(T, hide_index=True, width="stretch", height=38 * (len(T) + 1) + 3, column_config={
    "paper": "Paper", "crop": "Crop", "years": "Test years", "metric": "Measure",
    "theirs": st.column_config.TextColumn("Published"),
    "hist_avg": st.column_config.TextColumn("Historical avg"),
    **{v: st.column_config.TextColumn(v + (" (final)" if v == MAIN else "")) for v in order}})
st.caption("R² higher is better; RMSE, MAE (dt/ha) and MAPE (%) lower is better. n/a in the v15 column: not allowed "
           "under that paper's protocol (multi-year test periods freeze information at the end of training). "
           "v15's Srivastava values were computed after the results were in; single-year tests make them fair.")

if not V.empty:
    st.markdown("##### How the comparison changed across our versions")
    w = V.dropna(subset=["papers_won"]).copy()
    w["tot"] = w.papers_rows.fillna(18).astype(int)
    w["share"] = 100 * w.papers_won / w.tot
    fig = go.Figure(go.Bar(x=w.version, y=w.share, marker_color=[MODEL if v == MAIN else "#9ec5f4" for v in w.version],
                           text=w.papers_won.astype(int).astype(str) + " / " + w.tot.astype(str), textposition="outside",
                           hovertext=w.title, hovertemplate="%{x}: %{hovertext}<br>%{text}<extra></extra>"))
    plain_layout(fig, 320, yaxis=dict(range=[0, 105], ticksuffix="%", title="comparisons won"))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.caption("Share of comparisons won. v4 and v15b win the most of all 18 (12). v15 is shown on the 6 "
               "Srivastava rows only, where its error memory is allowed (4 of 6).")

# --------------------------------------------------------------------------- #
# Operational benchmark: JRC MARS 2025
# --------------------------------------------------------------------------- #
MA = load_table("mars_2025.csv")
if not MA.empty:
    st.markdown("---")
    st.markdown("##### The EU's operational forecast: JRC MARS, Germany 2025")
    grp = {"winter_wheat": 0, "winter_barley": 0, "summer_barley": 0, "rye": 0, "triticale": 0, "winter_rapeseed": 0}
    cer = MA[MA.crop.isin(list(grp))]
    root = MA[~MA.crop.isin(list(grp))]
    k = st.columns(3)
    k[0].metric("Mean error, all 9 crops", f"{MA.ens_err_official_pct.abs().mean():.1f}% vs {MA.mars_err_pct.abs().mean():.1f}%",
                help="YieldFormer-DE vs MARS, against the official national yield.")
    k[1].metric("Cereals and rapeseed", f"{cer.ens_err_official_pct.abs().mean():.1f}% vs {cer.mars_err_pct.abs().mean():.1f}%",
                help="MARS is closer for these six crops.")
    k[2].metric("Maize, sugar beet, potatoes", f"{root.ens_err_official_pct.abs().mean():.1f}% vs {root.mars_err_pct.abs().mean():.1f}%",
                help="YieldFormer-DE is closer for these three crops.")
    crops = [c for c in ["winter_wheat", "winter_barley", "rye", "triticale", "winter_rapeseed", "summer_barley",
                         "silage_maize", "potatoes", "sugar_beet"] if c in set(MA.crop)]
    D = MA.set_index("crop").reindex(crops)
    fig = go.Figure()
    for col, name, color, sym in (("mars_err_pct", "JRC MARS", "#7a4fb3", "circle"),
                                  ("ens_err_official_pct", f"YieldFormer-DE ({MAIN})", MODEL, "circle"),
                                  ("b5_err_official_pct", "5-year average", AVG5, "line-ns-open")):
        fig.add_trace(go.Scatter(x=D[col], y=[crop_label(c) for c in crops], mode="markers", name=name,
                                 marker=dict(color=color, size=11, symbol=sym, line=dict(width=2, color=color)),
                                 hovertemplate="%{y}: %{x:+.1f}%<extra>" + name + "</extra>"))
    fig.add_vline(x=0, line_color="#222", line_width=1)
    plain_layout(fig, 360, xaxis=dict(ticksuffix="%", title="error of the national 2025 yield vs official"),
                 yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key="mars_2025")
    st.caption("MARS: last bulletin before harvest (August 2025 for winter crops, September 2025 for the others; no "
               "oats forecast). YieldFormer-DE: rolling backtest with weather up to harvest, district forecasts "
               "weighted by 2020 census crop area. The September bulletins came before the maize, beet and potato "
               "harvests, which favours YieldFormer-DE for those crops; for winter barley, rapeseed and spring barley "
               "they came after the harvest, which favours MARS. One season cannot rank the two systems.")

with st.expander("Why we do not copy the papers' scoring shortcuts"):
    st.markdown(
        "Some published scores include years that were also used to tune the model. That makes the "
        "number look better without making forecasts better. For silage maize, CropFusionNet reports R² 0.75, "
        "while the same team's AgroForecast-DE, tested on two strictly unseen years, reaches about 0.4. "
        "We report every result on years the model never saw, with the rule for adopting a change written "
        "down before the test is run.")

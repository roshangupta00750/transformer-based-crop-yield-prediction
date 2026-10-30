"""Model journey: every version we tried, what it changed, its values and what we learned."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _results import (AVG5, AVG_LONG, BAD, HELP, MAIN, MODEL, ROOT, header, load_table, missing,  # noqa: E402
                      plain_layout)

V = load_table("versions.csv")
if V.empty:
    missing("versions.csv")
    st.stop()


@st.cache_data(show_spinner=False)
def comparison(name: str) -> pd.DataFrame:
    f = ROOT / "results" / "metrics" / name
    return pd.read_csv(f) if f.exists() else pd.DataFrame()


C = comparison("version_comparison_common.csv")
FULL = comparison("version_comparison_full.csv")

header("Model journey",
       f"We built and tested <b>{len(V)} versions</b>. Every version was planned in writing <b>before</b> it was "
       "run, including the rule for keeping it. From v6 on, a change is adopted only if it is better on the "
       "test years 2010–2015 with at least 95% certainty. Most ideas did <b>not</b> pass. "
       f"The <b>final model is {MAIN}, named YieldFormer-DE</b>: the average of three transformers (iTransformer, "
       "TFT, PatchTST) plus a correction for each district's recent errors. Before it, v15 (the iTransformer alone "
       "plus the correction) was the main model. <b>All differences between versions are small</b>, within about "
       "1 percentage point of accuracy.")

COLOR = {"Final model": MODEL, "Was base": "#6da7ec", "Was main": "#6da7ec", "Not adopted": BAD,
         "Rules": AVG5, "Diagnostic": AVG5}

# --------------------------------------------------------------------------- #
# All versions on the same forecasts
# --------------------------------------------------------------------------- #
if not C.empty:
    n_rows = C.iloc[0]["set"].split()[1]
    st.markdown("##### All versions on the same forecasts", help=HELP["accuracy"] + " " + HELP["unseen"])
    st.caption(f"Every version scored on exactly the same {int(n_rows):,} district-crop-years with the same official "
               "yields, so the numbers compare directly. v1–v3, v5, v6, v8 and v11 were not run on the 2010–2025 "
               "rolling test (earlier test design, rules only, rapeseed-only, or a diagnostic).")
    d = C.set_index("model")
    order = [m for m in ["v4", "v7", "v9", "v10", "v12", "v13", "v14", "v15b", "v15", "v17 TFT", "v17 PatchTST", MAIN]
             if m in d.index]
    PC = [c for c in d.columns if c.startswith("Pmain>_") or c.startswith("P15>_")]
    fig = go.Figure()
    for col, name, color, sym in (("2010-15", "Clean test 2010–2015", "#9ec5f4", "circle-open"),
                                  ("2016-25", "2016–2025", MODEL, "circle")):
        vals = [d.at[m, col] for m in order]
        fig.add_trace(go.Scatter(x=order, y=vals, name=name, mode="markers+text",
                                 marker=dict(color=color, size=13, symbol=sym, line=dict(width=2.5, color=color)),
                                 text=[f"{v:.2f}" for v in vals], textfont=dict(size=10),
                                 textposition="top center" if col == "2010-15" else "bottom center"))
    for ref, name, color in (("b5", "5-year average, 2016–2025", AVG5),
                             ("y_hist", "long-term average, 2016–2025", AVG_LONG)):
        fig.add_hline(y=d.at[ref, "2016-25"], line_dash="dot", line_color=color,
                      annotation_text=name, annotation_position="bottom left", annotation_font_size=10)
    plain_layout(fig, 400, yaxis=dict(ticksuffix="%", title="accuracy (100% − typical error)"))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.caption("Dots, not bars, because the axis does not start at 0: the whole range shown is only about 3 "
               "percentage points. Versions differ by tenths of a point; the gap to the simple averages "
               "(dotted lines) is larger.")

    t = d.reindex(order[::-1] + ["b5", "y_hist"]).reset_index()
    t["key"] = t["model"].str.split().str[0]                       # "v17 TFT" -> v17
    t = t.merge(V[["version", "title", "decision"]].rename(columns={"version": "key"}), on="key", how="left")
    t.loc[t.model == "v17 TFT", "title"] = "YieldFormer-DE-T: Temporal Fusion Transformer + memory"
    t.loc[t.model == "v17 PatchTST", "title"] = "YieldFormer-DE-P: PatchTST + memory"
    t.loc[t.model == "v15", "title"] = "YieldFormer-DE-i: iTransformer + district error memory"
    t["model"] = t["model"].replace({"b5": "5-year average", "y_hist": "Long-term average"})
    t[["title", "decision"]] = t[["title", "decision"]].fillna({"title": "reference", "decision": "–"})
    for c in ("2010-15", "2016-25", "2010-25", "national"):
        t[c] = t[c].map(lambda x: f"{x:.2f}%")
    t["within10"] = t["within10"].map(lambda x: f"{x:.1f}%")
    for c in PC:
        t[c] = t[c].map(lambda x: "–" if x != x else f"{x:.0f}%")
    st.dataframe(t[["model", "title", "decision", "2010-15", "2016-25", "2010-25", "national", "within10"] + PC],
                 hide_index=True, width="stretch", column_config={
        "model": "Version", "title": "What it is", "decision": "Status",
        "2010-15": st.column_config.TextColumn("Clean test 2010–15", help=HELP["hindcast"]),
        "2016-25": st.column_config.TextColumn("2016–25"),
        "2010-25": st.column_config.TextColumn("All 2010–25"),
        "national": st.column_config.TextColumn("National", help=HELP["national"]),
        "within10": st.column_config.TextColumn("Within ±10%", help=HELP["within10"]),
        **{c: st.column_config.TextColumn(
            f"{MAIN} better, {'2010–15' if c.endswith('10-15') else '2016–25'}",
            help=f"Share of 2,000 resamplings of the years in which {MAIN} has the higher skill. 95%+ = clearly better.")
           for c in PC}})
    best = d.loc[order, "2016-25"].idxmax()
    st.caption(f"Reading the table: on 2016–2025 the most accurate version is {best}; {MAIN} is the final model "
               "because it passed the pre-registered test on 2010–2015, where single versions such as v7, v9 or TFT "
               "score close to it but did not pass their own tests. v13, v14 and v4 are marginally better at the "
               "national level in some periods.")
    IV, AB = load_table("v18_intervals.csv"), load_table("v16_ablation.csv")
    CL, SC = load_table("v18_cleaned.csv"), load_table("v18_comparison.csv")
    st.markdown("##### How sure is the gain?",
                help="90% ranges from resampling whole years (2,000 times): the year-to-year luck of the test.")
    if not IV.empty:
        lab = {"XE18": f"YieldFormer-DE ({MAIN})", "E15": "YieldFormer-DE-i (v15)", "b5": "5-year average",
               "y_hist": "Long-term average"}
        win = {"hindcast_2010_2015": "Clean test 2010–2015", "dev_2016_2025": "2016–2025"}
        d = IV[(IV.level == "district") & IV.window.isin(win) & IV.quantity.str.startswith("accuracy")].copy()
        d["model"] = d.quantity.str.replace("accuracy ", "", regex=False).map(lab)
        fig = go.Figure()
        for w, color, sym in (("hindcast_2010_2015", "#9ec5f4", "circle-open"), ("dev_2016_2025", MODEL, "circle")):
            g = d[d.window == w].set_index("model").reindex(list(lab.values()))
            fig.add_trace(go.Scatter(x=g.index, y=g.estimate, name=win[w], mode="markers",
                                     marker=dict(color=color, size=12, symbol=sym, line=dict(width=2.5, color=color)),
                                     error_y=dict(type="data", symmetric=False, array=g.hi90 - g.estimate,
                                                  arrayminus=g.estimate - g.lo90, color=color, thickness=1.5, width=6),
                                     hovertemplate="%{x}<br>%{y:.2f}% (90% range %{customdata[0]:.2f}–%{customdata[1]:.2f})"
                                                   "<extra></extra>", customdata=g[["lo90", "hi90"]].to_numpy()))
        plain_layout(fig, 360, yaxis=dict(ticksuffix="%", title="accuracy, with 90% range"))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        def gap(q, w):
            r = IV[(IV.level == "district") & (IV.window == w) & (IV.quantity == q)].iloc[0]
            return f"{r.estimate:+.2f} points (90% range {r.lo90:+.2f} to {r.hi90:+.2f})"
        st.markdown(
            f"- **{MAIN} vs v15** (what averaging three transformers adds): {gap('XE18 − E15 (pp)', 'hindcast_2010_2015')} "
            f"on 2010–2015, {gap('XE18 − E15 (pp)', 'dev_2016_2025')} on 2016–2025.\n"
            f"- **{MAIN} vs the 5-year average**: {gap('XE18 − b5 (pp)', 'hindcast_2010_2015')} on 2010–2015, "
            f"{gap('XE18 − b5 (pp)', 'dev_2016_2025')} on 2016–2025.\n"
            "- The ranges of single models overlap because all of them do better or worse in the same years "
            "(good and bad seasons); the gaps between models are measured on the same years and are much tighter.")
    if not CL.empty:
        c = CL[(CL.forecast == "XE18") & (CL.window == "all_2010_2025")].iloc[0]
        st.markdown(f"- **Doubtful official figures:** {int(c.flagged)} of {int(c.rows):,} official yields look like "
                    "reporting errors (less than half or more than double the district's neighbouring years, while "
                    "neighbouring districts had a normal year). Leaving them out moves accuracy only from "
                    f"{c.accuracy_all:.2f}% to {c.accuracy_clean:.2f}%; no ranking changes.")
    if not SC.empty:
        h = SC[(SC.incumbent == "E15") & (SC.window == "hindcast_2010_2015")].iloc[0]
        dv = SC[(SC.incumbent == "E15") & (SC.window == "dev_2016_2025")].iloc[0]
        st.warning(f"**Many decisions, same test years.** 12 adoption decisions had been taken on 2010–2015 when v18 was adopted (14 by v20). "
                   f"Corrected for that (Bonferroni), a pass needs 99.6% certainty; {MAIN} reached "
                   f"{100 * h.p_delta_gt_0:.1f}%, so on 2010–2015 alone it could be a lucky pass. On 2016–2025 it is "
                   f"{100 * dv.p_delta_gt_0:.1f}% certain, but those years were seen while developing the models. "
                   f"Read {MAIN} as *better than v15 by a small margin*; the frozen 2026 forecasts are the clean test.",
                   icon="⚖️")

    st.markdown("##### Is the transformer needed?",
                help="Ablation planned in writing before it was run (pre-registration v16): remove the network and keep "
                     "only the normal, the 5-year average and the error memory, on exactly the same rows.")
    if not AB.empty:
        arm = {"E15": "YieldFormer-DE-i (v15)", "Q12": "v12 (no memory)", "A16a": "Trend + memory",
               "A16b": "5-yr average + memory", "b5": "5-year average", "y_hist": "Long-term average",
               "GE16": "XGBoost + memory (reference)"}
        ens = ({w: SC[(SC.incumbent == "E15") & (SC.window == w)].acc_challenger.iloc[0]
                for w in ("hindcast_2010_2015", "dev_2016_2025")} if not SC.empty else {})
        fig = go.Figure()
        for w, name, color, sym in (("hindcast_2010_2015", "Clean test 2010–2015", "#9ec5f4", "circle-open"),
                                    ("dev_2016_2025", "2016–2025", MODEL, "circle")):
            g = AB[AB.window == w].set_index("arm")
            labels = ([f"YieldFormer-DE ({MAIN})"] if ens else []) + list(arm.values())
            vals = ([ens[w]] if ens else []) + [g.acc_E15.iloc[0]] + [g.at[a, "acc_arm"] for a in list(arm)[1:]]
            fig.add_trace(go.Scatter(x=labels, y=vals, name=name, mode="markers+text",
                                     marker=dict(color=color, size=13, symbol=sym, line=dict(width=2.5, color=color)),
                                     text=[f"{v:.2f}" for v in vals], textfont=dict(size=10),
                                     textposition="top center" if w.startswith("hindcast") else "bottom center"))
        plain_layout(fig, 380, yaxis=dict(ticksuffix="%", title="accuracy (100% − typical error)"))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        h = AB[(AB.window == "hindcast_2010_2015")].set_index("arm")
        dv = AB[(AB.window == "dev_2016_2025")].set_index("arm")
        ref = ""
        if not SC.empty:
            rh = SC[(SC.incumbent == "GE16") & (SC.window == "hindcast_2010_2015")].iloc[0]
            rd = SC[(SC.incumbent == "GE16") & (SC.window == "dev_2016_2025")].iloc[0]
            ref = (f"- **Reference, not a candidate:** XGBoost, a strong tree model, given the same inputs and the same "
                   f"memory. {MAIN} is ahead in both periods: {rh.acc_challenger:.2f}% vs {rh.acc_incumbent:.2f}% on "
                   f"2010–2015 ({100 * rh.p_delta_gt_0:.1f}% certain) and {rd.acc_challenger:.2f}% vs "
                   f"{rd.acc_incumbent:.2f}% on 2016–2025 ({100 * rd.p_delta_gt_0:.0f}% certain). The iTransformer "
                   f"alone (v15) was behind XGBoost on 2016–2025 ({dv.at['GE16', 'acc_E15']:.2f}% vs "
                   f"{dv.at['GE16', 'acc_arm']:.2f}%). So the transformers are *competitive with, and as an ensemble "
                   "slightly ahead of,* a strong tree model, and clearly better than every simple method.")
        st.markdown(
            "Same years, same rows. *Trend + memory* and *5-yr average + memory* are the method without any "
            "transformer: the district normal (or the 5-year average) corrected by the same district error memory.\n"
            f"- **2010–2015 (the years all decisions use):** YieldFormer-DE-i (the iTransformer alone, v15) is already ahead of every "
            f"version without a network, at least {100 * min(h.loc[['A16a', 'A16b', 'b5', 'y_hist'], 'p_delta_gt_0']):.1f}% "
            "certain in each comparison.\n"
            f"- **2016–2025:** the gap is larger: v15 {dv.at['A16a', 'acc_E15']:.2f}% vs trend + memory "
            f"{dv.at['A16a', 'acc_arm']:.2f}% and 5-yr average + memory {dv.at['A16b', 'acc_arm']:.2f}%"
            + (f"; the three-transformer average {ens['dev_2016_2025']:.2f}%.\n" if ens else ".\n")
            + "- **Plain reading:** the transformers carry most of the gain over the simple averages; the district "
            "error memory adds a smaller, separate gain on top.\n" + ref)
    if not FULL.empty:
        with st.expander("Recent versions on the full corrected panel (all districts)"):
            f = FULL.copy()
            f["model"] = f["model"].replace({"b5": "5-year average", "y_hist": "Long-term average"})
            st.dataframe(f[["model", "rows", "districts", "2010-15", "2016-25", "2010-25", "national", "within10"]],
                         hide_index=True, width="stretch", column_config={
                             c: st.column_config.NumberColumn(c, format="%.2f%%") for c in
                             ("2010-15", "2016-25", "2010-25", "national", "within10")})
else:
    st.info("Run `python scripts/67_version_comparison.py` to add the all-versions comparison.")

st.markdown("---")
st.markdown("##### Every version")
cm = C.set_index("model") if not C.empty else pd.DataFrame()
for _, r in V.iterrows():
    c1, c2 = st.columns([1, 7])
    color = COLOR.get(r.decision, AVG5)
    c1.markdown(f"<div style='font-size:1.4rem;font-weight:700;color:{color}'>{r.version}</div>",
                unsafe_allow_html=True)
    badge = (f"<span style='background:{color};color:white;padding:1px 8px;border-radius:9px;"
             f"font-size:0.78rem'>{r.decision}</span>")
    vals = []
    if r.version in cm.index:
        vals.append(f"rows shared by all versions: 2010–15 {cm.at[r.version, '2010-15']:.2f}% · "
                    f"2016–25 {cm.at[r.version, '2016-25']:.2f}%")
    if r.papers_won == r.papers_won:
        tot = int(r.papers_rows) if r.papers_rows == r.papers_rows else 18
        vals.append(f"papers {int(r.papers_won)}/{tot}" + (" (Srivastava only)" if tot == 6 else ""))
    extra = (" · " + " · ".join(vals)) if vals else ""
    c2.markdown(f"**{r.title}** {badge}<span style='color:#898781;font-size:0.85rem'>{extra}</span>  \n"
                f"{r.change}.  \n<span style='color:#555'>→ {r.lesson}</span>", unsafe_allow_html=True)

st.markdown("---")
st.markdown("##### What we learned")
c = st.columns(2)
c[0].markdown(
    "**What helped**\n"
    "- Averaging satellite data over farmland only: the one idea confirmed on its own (v1)\n"
    "- Harvest date + growth-stage weather + district anomalies, kept together by the old count rule, not confirmed (v3)\n"
    "- Trend rule + per-crop fine-tuning, kept by the old count rule, not confirmed (v4: 12 of 18 paper comparisons)\n"
    "- Correct district matching + station weather: fixes the data without losing accuracy (v12)\n"
    "- District error memory: corrects lagging district normals (v15)\n"
    "- The transformer itself: clearly ahead of averages and trend + memory in both periods (v16)\n"
    "- Averaging three transformers (iTransformer, TFT, PatchTST): steadier than any one of them (v18, the final "
    "model)")
c[1].markdown(
    "**What did not**\n"
    "- More history, national-level switches, splitting autumn/spring crops (v2, v5)\n"
    "- Recent-yield inputs and blended baselines: promising but not certain (v7, v9)\n"
    "- ERA5 reanalysis weather instead of stations (v10, v11)\n"
    "- Crop-specific satellite averaging: pixels bigger than fields (v13)\n"
    "- DWD 1 km gridded weather: better data, no better forecasts (v14)\n"
    "- Area-weighted training: shifts accuracy between crops, no net gain (v15b)\n"
    "- Removing doubtful official figures: changes accuracy by < 0.1 point (v16)\n"
    "- TFT alone: more accurate on average but uneven across years, so not certain enough (v17); PatchTST alone: "
    "weaker than the iTransformer (v17)\n"
    "- Scaling up the good/bad-year swing, recalibrated ranges (v19)\n"
    "- Crop stress inputs at critical stages: help TFT and PatchTST, hurt the iTransformer (v20)")
st.caption("2025 winter wheat (mean over districts): the iTransformer version (v15) estimated Germany 15% too low, "
           "the TFT 2% and YieldFormer-DE (v18) 8.8% (5-year average 5.6%, long-term average 5.0%). Spring 2025 was "
           "the driest of 2010–2025, yet the crop canopy was dense and the harvest good; the iTransformer had learned "
           "that a dry spring means a poor wheat harvest and over-reacted. Its inputs were complete.")
st.caption("Why better inputs stopped helping: over half of the remaining error is specific to single districts "
           "and not shared with their neighbours, while weather and satellite data are spatially smooth. Much of "
           "it is noise in the official statistics where a crop is rare.")

with st.expander("Full table"):
    st.dataframe(V.drop(columns=[c for c in ("planned", "result", "hindcast_new", "hindcast_main") if c in V]),
                 hide_index=True, width="stretch", column_config={
                     "papers_won": st.column_config.NumberColumn("Papers won", format="%d"),
                     "papers_rows": st.column_config.NumberColumn("of rows", format="%d")})
    st.caption("Source: the written pre-registration of each version (paper/preregistration_*.md).")

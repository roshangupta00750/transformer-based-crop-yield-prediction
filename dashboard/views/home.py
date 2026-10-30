"""Overview: the whole project on one screen, for any reader."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _results import (AVG5, BAD, CROPS, HELP, MAIN, MODEL, MODEL_NAME, crop_label, header, load_summary,  # noqa: E402
                      load_table, missing, plain_layout)

S = load_summary()
N = load_table("national.csv")
W = load_table("bad_year_warnings.csv")
if not S or N.empty:
    missing("summary.json")
    st.stop()

header(f"{MODEL_NAME}: transformer-based crop yield prediction for Germany",
       "We built an AI model that estimates the harvest of <b>10 crops in every German district</b> from daily "
       "weather, satellite images and soil data, right after harvest and about 3 months before the official "
       "statistics. Each year from 2010 to 2025 was estimated by a model trained only on earlier years. Our final "
       f"model, <b>{MODEL_NAME}</b> (version {MAIN}: the average of three transformers, iTransformer, TFT and "
       f"PatchTST, plus a correction for each district's recent errors), is typically <b>{S['mape_model']:.1f}% away</b> from the real district yield, against "
       f"{S['mape_avg5']:.1f}% for simply using the last 5 years' average: about "
       f"<b>{S['error_reduction_vs_avg5']:.0f}% smaller errors</b>. The gain is modest in normal years and large in "
       "drought years.")

k = st.columns(4)
k[0].metric("District accuracy", f"{S['accuracy_district']:.1f}%",
            f"error {S['mape_model']:.1f}% vs {S['mape_avg5']:.1f}% (5-yr avg)", delta_color="off",
            help=HELP["accuracy"] + " " + HELP["unseen"])
k[1].metric("National accuracy", f"{S['accuracy_national']:.1f}%",
            f"5-year average: {S['accuracy_national_avg5']:.1f}%", delta_color="off", help=HELP["national"])
w5 = W[W.warning_rule.str.contains("5%")].iloc[0] if not W.empty else None
k[2].metric("Bad years warned", f"{int(w5.bad_years_warned) if w5 is not None else S['bad_detected']} of {S['bad_years']}",
            (f"rule: estimate ≥5% below normal · {int(w5.warnings_in_other_years)} warnings in "
             f"{int(w5.warnings_in_other_years + w5.other_years_quiet)} other crop-years") if w5 is not None else "",
            delta_color="off", help=HELP["bad_year"] + " See the Bad years page for other warning rules.")
k[3].metric("Paper comparisons won", f"{S['papers_won']} of {S['papers_total']}",
            (f"by the final model ({MAIN}) on each paper's own split" if S.get("papers_model") == MAIN else
             f"iTransformer versions (v12 · v15: {S.get('papers_won_v15', '–')} of {S.get('papers_total_v15', '–')} "
             "where allowed); the final model's paper runs are in progress"),
            delta_color="off",
            help="Head-to-head comparisons with three published German studies, each repeated with that paper's own "
                 "training and test years. See 'Which model is which' below.")
st.caption(f"Within ±10% of the real yield: {S['within10']:.0f}% of district estimates "
           f"(5-year average: {S['within10_avg5']:.0f}%). So roughly every second district estimate is more than 10% off; "
           "national totals are much more reliable because district errors partly cancel out.")

c = st.columns(2)
with c[0]:
    st.markdown("##### Which model is which")
    st.table(pd.DataFrame([
        (f"{MODEL_NAME} ({MAIN}): the final model", "All result and analysis pages, all headline numbers above, "
         "the 2026 ensemble forecast", "average of iTransformer, TFT and PatchTST + a correction for each district's "
         "recent errors"),
        (f"{MAIN} without its correction", f"Paper comparisons ({S['papers_won']} of {S['papers_total']})"
         if S.get("papers_model") == MAIN else "Paper comparisons (in progress)",
         "the correction needs last year's real yields; the papers' test rules don't allow that"),
        ("YieldFormer-DE-i (v15): the iTransformer member", "2026 YieldFormer-DE-i forecast; earlier paper comparisons",
         "the main model until 11 Oct 2026; one transformer + the correction. Its siblings YieldFormer-DE-T (TFT) and "
         "YieldFormer-DE-P (PatchTST) are the other two members of the ensemble"),
        ("v1: the first model", "2026 forecast (v1)",
         "frozen on 27 Sep 2026, before the later versions existed"),
    ], columns=["Model", "Used on", "Why"]).set_index("Model"))
with c[1]:
    st.markdown("##### Why the counts differ")
    st.table(pd.DataFrame([
        ("~400", "all German districts (Landkreise and cities)"),
        (str(S["districts"]), "districts with official yields in the test 2010–2025 (most pages)"),
        (f"{S['rows']:,}", "district × crop × year estimates checked (all 10 crops)"),
        ("28,081 / 269", "estimates / districts that every version has in common (Model journey)"),
        ("209–260", "districts per crop in the 2026 forecast (the older data set used by v1)"),
    ], columns=["Number", "What it counts"]).set_index("Number"))

st.markdown("---")
st.markdown("#### How it works")
c = st.columns(3)
c[0].markdown("**1 · Inputs**  \nFor each district and season: daily temperature, rain, soil moisture and "
              "humidity (German Weather Service), plant greenness from satellites (NASA MODIS), "
              "soil and terrain, and the crop's growth stages.")
c[1].markdown("**2 · Model**  \nA transformer (*iTransformer*), the same family of AI as modern "
              "language models, reads these time series and learns how weather during each growth "
              "stage pushes the yield above or below the district's normal.")
c[2].markdown("**3 · Output**  \nA yield estimate in dt/ha for every district and crop, available right "
              "after harvest, about 3 months before the official statistics. It is not an early warning "
              "months ahead: it uses the weather up to harvest.")

st.markdown("---")
st.markdown("#### Germany's harvest: real vs estimate", help=HELP["national"] + " " + HELP["unseen"])
crop = st.selectbox("Crop", CROPS, format_func=crop_label, key="home_crop")
d = N[N.crop == crop].sort_values("year")
fig = go.Figure()
bad = d[d.kind == "bad"]
for _, r in bad.iterrows():
    fig.add_vrect(x0=r.year - 0.5, x1=r.year + 0.5, fillcolor=BAD, opacity=0.12, line_width=0)
fig.add_trace(go.Scatter(x=d.year, y=d.avg5, name="5-year average", mode="lines",
                         line=dict(color=AVG5, dash="dot", width=2)))
fig.add_trace(go.Scatter(x=d.year, y=d.actual, name="Real yield (Destatis)", mode="lines+markers",
                         line=dict(color="#222", width=2.5), marker=dict(size=7)))
fig.add_trace(go.Scatter(x=d.year, y=d.predicted, name=f"{MODEL_NAME} estimate", mode="lines+markers",
                         line=dict(color=MODEL, width=2.5), marker=dict(size=7)))
plain_layout(fig, 430, yaxis_title="dt/ha", hovermode="x unified")
st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
d = d.assign(err=100 * (d.predicted / d.actual - 1))
worst = d.reindex(d.err.abs().sort_values(ascending=False).index).head(2)
bands = (f"Red bands = bad harvest years for {crop_label(crop).lower()} (national yield ≥ 10% below normal): "
         + ", ".join(str(int(y)) for y in bad.year) + "." if len(bad) else
         f"{crop_label(crop)} had no bad harvest year (national yield ≥ 10% below normal) in 2010–2025, so no red bands.")
misses = "Largest misses: " + "; ".join(f"{int(r.year)} estimate {r.predicted:.1f} vs real {r.actual:.1f} ({r.err:+.0f}%)"
                                         for r in worst.itertuples()) + "."
note = (" 2025 wheat: YieldFormer-DE-i (the iTransformer alone) was 15% too low because it over-read the dry spring; TFT was "
        "only 2% too low (YieldFormer-DE-T), so the average of the three is in between." if crop == "winter_wheat" else "")
st.caption(bands + " " + misses + note + " " + HELP["dt_ha"])

st.markdown("---")
st.markdown("#### Top insights")
I = S.get("insights", {})
YR = load_table("accuracy_by_year.csv")
y18 = YR.loc[(YR.accuracy_model - YR.accuracy_avg5).idxmax()] if not YR.empty else None   # year with the largest gain
es = I.get("error_share", {})
st.markdown(
    f"1. **Beats the simple 5-year average for all 10 crops**, most in drought years"
    + (f" ({int(y18.year)}: {y18.accuracy_model:.1f}% vs {y18.accuracy_avg5:.1f}%).\n" if y18 is not None else ".\n")
    + 
    f"2. **Germany-wide totals are reliable** ({S['accuracy_national']:.1f}%); single districts much less so "
    f"({S['within10']:.0f}% within ±10%).\n"
    f"3. **{es.get('Single district', 0):.0f}% of the remaining error belongs to single districts**, which weather "
    "data cannot explain; better weather or satellite data (v13, v14) did not help.\n"
    f"4. **Where a crop is rare, errors are about {I.get('error_small_vs_large', [0, 1])[0] / I.get('error_small_vs_large', [0, 1])[1]:.1f}× "
    "larger**: the official statistics are noisy there.\n"
    "5. **Summer rain decides maize, potatoes and sugar beet** (r ≈ 0.8–0.9); cereals depend on other seasons and "
    "are harder.\n\n"
    "All ten insights, with charts: **Key insights** page. Analytics: **Error analysis**, **Weather and yield**, "
    "**Where crops are grown**, **Predictions vs actual**.")

st.markdown("#### Where to look next")
st.markdown(
    "- **Key insights**: the ten main findings, each with a chart.\n"
    "- **Predictions vs actual**: every estimate next to the official yield; filter and download.\n"
    "- **Accuracy**: how close the estimates are, by crop, year and district, with a map.\n"
    "- **Bad years**: droughts and failed harvests, which ones the model warned about, and its limits.\n"
    "- **Compared with papers**: the same tests as three published studies.\n"
    "- **Model journey**: every version (v1–v20), its values, and what we learned.\n"
    "- **Forecasts**: the frozen forecasts for 2026–2028, and how the models did in 2024 and 2025.\n"
    "- **Analytics**: where crops are grown, error analysis, weather and yield.\n"
    "- **How it works**: the method and a glossary, without jargon."
)

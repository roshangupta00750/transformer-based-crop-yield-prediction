"""Forecasts: frozen YieldFormer-DE forecasts (2026, 2027) and the earlier v1 forecast for 2026."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _results import (BAD, CROPS, DIVERGING, GOOD, MAIN_LABEL, NAME_I, NAME_T, NAME_P, crop_label, district_map, header,  # noqa: E402
                      load_table, missing, plain_layout)

YF = load_table("forecast_yieldformer.csv")
F = load_table("forecast_2026.csv")
if YF.empty and F.empty:
    missing("forecast_yieldformer.csv")
    st.stop()

header("Forecasts",
       "A forecast only counts as a test if it is written down <b>before</b> the official figures exist. Each "
       "forecast below was frozen on the day it was issued and sealed with a cryptographic fingerprint (SHA-256), "
       "so it cannot be changed afterwards. When Destatis publishes the district yields, each one is scored "
       "against the official figures, next to the 5-year average and the long-term average.")


def national(D):
    n = (D.groupby("crop").agg(pred=("y_pred", "mean"), normal=("y_hist", "mean"), lo=("lo80", "mean"),
                               hi=("hi80", "mean"), n=("y_pred", "size"), lead=("lead_days", "median"))
         .reindex([c for c in CROPS if c in set(D.crop)]).reset_index())
    n["vs"] = 100 * (n.pred / n.normal - 1)
    return n


def vs_normal_bars(nat, key):
    fig = go.Figure(go.Bar(y=nat.crop.map(crop_label), x=nat.vs, orientation="h",
                           marker_color=[GOOD if v > 0 else BAD for v in nat.vs],
                           text=nat.vs.map("{:+.1f}%".format), textposition="outside", cliponaxis=False,
                           customdata=np.c_[nat.pred, nat.normal],
                           hovertemplate="%{y}<br>forecast %{customdata[0]:.1f} dt/ha, normal %{customdata[1]:.1f}"
                                         "<extra></extra>"))
    fig.add_vline(x=0, line_color="#222", line_width=1)
    plain_layout(fig, 420, xaxis=dict(ticksuffix="%", title="forecast national yield vs long-term average"),
                 yaxis=dict(autorange="reversed"), showlegend=False)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key=key)


def per_district(D, key, model_note):
    crop = st.selectbox("Crop", [c for c in CROPS if c in set(D.crop)], format_func=crop_label, key=f"{key}_crop")
    g = D[D.crop == crop]
    if g.vs_normal.abs().max() > 0.05:
        lim = float(np.nanpercentile(g.vs_normal.abs(), 95))
        fig = district_map(g.assign(v=-g.vs_normal), "v", label="vs normal", colorscale=DIVERGING, zmid=0,
                           zrange=(-lim, lim), value_fmt="+.1f")
        if fig is not None:
            fig.data[-1].customdata = g.set_index("landkreis").reindex(fig.data[-1].locations).vs_normal.values
            fig.data[-1].hovertemplate = "%{text}<br>%{customdata:+.1f}% vs normal<extra></extra>"
            fig.data[-1].showscale = False
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key=f"{key}_map")
            st.caption("Blue = forecast above the district's normal yield, red = below.")
    t = g[["district", "y_pred", "lo80", "hi80", "y_hist", "vs_normal"]].sort_values("district")
    st.dataframe(t, hide_index=True, width="stretch", height=320, column_config={
        "district": "District", "y_pred": st.column_config.NumberColumn("Forecast (dt/ha)", format="%.1f"),
        "lo80": st.column_config.NumberColumn("Low (80%)", format="%.1f",
                                              help=f"80% range from {model_note}'s errors in 2016–2025: about 8 of 10 "
                                                   "real yields should fall inside. Whether that holds is part of the test."),
        "hi80": st.column_config.NumberColumn("High (80%)", format="%.1f"),
        "y_hist": st.column_config.NumberColumn("Normal", format="%.1f"),
        "vs_normal": st.column_config.NumberColumn("vs normal", format="%+.1f%%")}, key=f"{key}_tbl")
    st.caption(f"{len(g)} districts, A–Z. Normal = the district's long-term average with trend. Very small numbers "
               "(e.g. city districts) come from districts that grow very little of the crop. dt/ha = 100 kg per hectare.")


# --------------------------------------------------------------------------- #
# YieldFormer-DE
# --------------------------------------------------------------------------- #
st.markdown("### YieldFormer-DE forecasts")
if YF.empty:
    st.info("No YieldFormer-DE forecast frozen yet.")
else:
    if "model" not in YF:
        YF["model"] = ""
    files = (YF.drop_duplicates("file")[["year", "file", "issued", "sha256", "method", "data_end", "model"]]
               .sort_values(["year", "issued"], ascending=[True, False]))
    st.markdown("".join(f"- **{int(r.year)}**: issued {r.issued}, {r.method.split(':')[-1].strip()}"
                        + ("" if str(r.data_end).startswith("none") else f"; data up to {r.data_end}")
                        + f". Fingerprint `{r.sha256[:16]}…`\n" for r in files.itertuples()))
    label = {r.file: f"{int(r.year)} · {r.model}" for r in files.itertuples()}
    pick = st.radio("Forecast", list(label), index=0, horizontal=True, format_func=label.get, key="yf_file")
    D = YF[YF.file == pick]
    yr = int(D.year.iloc[0])
    nat = national(D)
    st.markdown(f"##### Germany: {yr} forecast vs normal")
    if nat.vs.abs().max() < 0.05:
        st.info(f"**{yr} = each district's normal yield.** No {yr} season data exist yet (winter crops are only being "
                "sown). Our tests show that this far ahead the network cannot beat the normal, and the error "
                "correction did not pass its test for the normal, so the registered rule makes the frozen forecast "
                "the district normal. An update with the network can be issued once the season has started; this "
                "file stays as it is.", icon="ℹ️")
        rel = lambda v: 100 * v / nat.normal
        fig = go.Figure(go.Scatter(x=rel(nat.pred), y=[f"{crop_label(c)} · {v:.1f} dt/ha" for c, v in zip(nat.crop, nat.pred)],
                                   mode="markers", marker=dict(color="#2a78d6", size=11),
                                   text=[f"{v:.1f} dt/ha" for v in nat.pred],
                                   error_x=dict(type="data", symmetric=False, array=rel(nat.hi) - rel(nat.pred),
                                                arrayminus=rel(nat.pred) - rel(nat.lo), color="#2a78d6", thickness=1.5),
                                   customdata=np.c_[nat.lo, nat.hi],
                                   hovertemplate="%{y}: %{text}<br>80% range %{customdata[0]:.1f}–%{customdata[1]:.1f}"
                                                 "<extra></extra>"))
        fig.add_vline(x=100, line_color="#222", line_width=1)
        plain_layout(fig, 420, xaxis=dict(ticksuffix="%", title="% of the normal yield: dot = national forecast, bar = 80% range of a single district"),
                     yaxis=dict(autorange="reversed"), showlegend=False)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key="yf_nat")
    else:
        vs_normal_bars(nat, "yf_bars")
        late = nat[nat.lead > 0]
        if len(late):
            st.caption("Crops still in the field at the data end: "
                       + ", ".join(f"{crop_label(r.crop)} ({int(r.lead)} days before typical harvest)"
                                   for r in late.itertuples()) + ".")
    st.markdown("##### Per district")
    per_district(D, f"yf{pick}", "YieldFormer-DE")

# --------------------------------------------------------------------------- #
# Backtests 2024 / 2025
# --------------------------------------------------------------------------- #
BT = load_table("backtest_2024_2025.csv")
if not BT.empty:
    st.markdown("---")
    st.markdown("### How the forecasts did in 2024 and 2025")
    st.caption("Backtests: for each year, every model was trained only on data up to two years earlier, so it never "
               "saw that year's yield, and is checked against the official figures. Unlike the frozen forecasts "
               "above, these were not written down in advance; they show what to expect, the 2026 scoring is the "
               "real test.")
    bt_years = sorted(int(y) for y in BT.year.unique())
    by = st.radio("Year", bt_years, index=len(bt_years) - 1, horizontal=True, key="bt_year")
    b = BT[BT.year == by]
    order = [MAIN_LABEL, NAME_I, NAME_T, NAME_P,
             "XGBoost + memory (reference)", "5-year average", "Long-term average"]
    acc = (b.assign(w=b.district_accuracy * b.rows).groupby("model")[["w", "rows"]].sum()
             .pipe(lambda d: d.w / d.rows).reindex([m for m in order if m in set(b.model)]))
    col = {m: ("#2a78d6" if m == MAIN_LABEL else "#9ec5f4" if m in (NAME_I, NAME_T, NAME_P)
               else "#c9c8c2") for m in acc.index}
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"##### Accuracy in {by}, all districts and crops")
        fig = go.Figure(go.Bar(y=acc.index, x=acc.values, orientation="h", marker_color=[col[m] for m in acc.index],
                               text=[f"{v:.2f}%" for v in acc.values], textposition="outside", cliponaxis=False))
        plain_layout(fig, 360, xaxis=dict(range=[float(acc.min()) - 2, float(acc.max()) + 1], ticksuffix="%"),
                     yaxis=dict(autorange="reversed"), showlegend=False)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key="bt_acc")
        st.caption(f"Blue = transformers (dark = {MAIN_LABEL}, the ensemble of the three light-blue ones); grey = simple "
               "methods and the XGBoost reference. "
                   "Axis zoomed.")
    with c2:
        st.markdown(f"##### Germany {by}: forecast vs real, by crop")
        fig = go.Figure()
        crops = [c for c in CROPS if c in set(b.crop)]
        for m, color, sym in ((MAIN_LABEL, "#2a78d6", "circle"), (NAME_I, "#9ec5f4", "circle-open"), ("5-year average", "#898781", "diamond")):
            d = b[b.model == m].set_index("crop").reindex(crops)
            err = 100 * (d.forecast / d.real - 1)
            fig.add_trace(go.Scatter(x=err, y=[crop_label(c) for c in crops], mode="markers", name=m.split(" (")[0],
                                     marker=dict(color=color, size=11, symbol=sym, line=dict(width=2, color=color)),
                                     customdata=np.c_[d.forecast, d.real],
                                     hovertemplate="%{y}: forecast %{customdata[0]:.1f}, real %{customdata[1]:.1f} dt/ha"
                                                   " (%{x:+.1f}%)<extra></extra>"))
        fig.add_vline(x=0, line_color="#222", line_width=1)
        plain_layout(fig, 360, xaxis=dict(ticksuffix="%", title="forecast vs real national yield"),
                     yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key="bt_crop")
        e = b[b.model == MAIN_LABEL].set_index("crop")
        low = int((e.forecast < e.real).sum())
        st.caption(f"Left of the line = forecast too low. In {by} the ensemble was too low for {low} of {len(e)} crops"
                   + (": a better harvest than any method expected." if low >= 0.7 * len(e) else "."))

# --------------------------------------------------------------------------- #
# 2026: first check against PRELIMINARY official national yields
# --------------------------------------------------------------------------- #
PR = load_table("prelim_2026.csv")
if not PR.empty:
    st.markdown("---")
    st.markdown("### 2026: first check against preliminary official yields")
    cols = [MAIN_LABEL, NAME_I, "Normal year"]
    mae = {c: float(PR[c].abs().mean()) for c in cols if c in PR}
    st.caption("Destatis has published preliminary national yields for 8 crops, but no district yields yet, so this "
               "is not the real test of the sealed forecast. District forecasts are turned into a national figure "
               "with the 2020 census crop areas, the same way as for the 2025 comparison with JRC MARS.")
    k = st.columns(len(mae))
    for i, (c, v) in enumerate(mae.items()):
        k[i].metric(f"Mean error, {c.split(' (')[0] if c != NAME_I else NAME_I}", f"{v:.1f}%")
    fig = go.Figure()
    crops = [c for c in CROPS if c in set(PR.crop)]
    P2 = PR.set_index("crop").reindex(crops)
    for c, color, sym in ((MAIN_LABEL, "#2a78d6", "circle"), (NAME_I, "#9ec5f4", "circle-open"),
                          ("Normal year", "#898781", "diamond")):
        if c in P2:
            fig.add_trace(go.Scatter(x=P2[c], y=[crop_label(x) for x in crops], mode="markers", name=c.split(" (")[0]
                                     if c != NAME_I else NAME_I,
                                     marker=dict(color=color, size=11, symbol=sym, line=dict(width=2, color=color)),
                                     hovertemplate="%{y}: %{x:+.1f}% vs preliminary official<extra></extra>"))
    fig.add_vline(x=0, line_color="#222", line_width=1)
    plain_layout(fig, 360, xaxis=dict(ticksuffix="%", title="forecast vs preliminary official national yield"),
                 yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key="prelim_2026")
    af = PR.set_index("crop").get("AgroForecast-DE July (wheat, ratio basis)")
    yr = PR.set_index("crop").get("YieldFormer-DE (v18, ratio basis)")
    wheat = (f" For winter wheat, the July forecast of AgroForecast-DE was closer ({af['winter_wheat']:+.1f}% against "
             f"{yr['winter_wheat']:+.1f}% for {MAIN_LABEL}, compared on the same basis)."
             if af is not None and yr is not None and af.get('winter_wheat') == af.get('winter_wheat') else "")
    st.caption(f"Right of the line = forecast too high. 2026 was hot and dry: {MAIN_LABEL} is close for spring barley "
               "and triticale and too high for rapeseed, wheat, potatoes and oats." + wheat + " The ranking depends on "
               "how district forecasts are turned into a national figure: without any conversion, the normal-year "
               "forecast is slightly ahead.")

# --------------------------------------------------------------------------- #
# v1
# --------------------------------------------------------------------------- #
if not F.empty:
    st.markdown("---")
    st.markdown("### Earlier forecast from v1")
    st.warning(f"**This 2026 forecast is from v1, not the final model {MAIN_LABEL}.** It was frozen on "
               f"{F.issued.iloc[0]}, before the later versions existed, with data up to {F.data_end.iloc[0]}; changing it afterwards "
               "would spoil the test. v1 also uses an older district data set (about 209–260 districts per crop). "
               "Both 2026 forecasts will be scored side by side.", icon="ℹ️")
    st.markdown("##### Germany: 2026 forecast vs normal")
    vs_normal_bars(national(F), "v1_bars")
    with st.expander("v1: per district"):
        per_district(F, "v1", "v1")

"""Dashboard: one interactive screen. Global filters in the sidebar drive every KPI, chart and table.

Each block answers one decision question with the chart form that fits it (see dashboard/SPEC.md):
stat tiles for headline numbers, dumbbells for model-vs-comparison per item, diverging bars for
gains and biases, lines for change over time, small multiples for many crops, a plain table for the
multi-measure scorecard. No heatmaps of numbers, no scrolling tables.
"""
from __future__ import annotations

import html
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _results import (AVG5, BAD, CROPS, MODEL, MODEL_NAME, ROOT, SEQ_BLUE, WEATHER_LABEL,  # noqa: E402
                      crop_label, district_map, load_table, missing, MAIN, MAIN_LABEL, NAME_I, NAME_T, NAME_P)

M = load_table("model_rows.parquet")
AREA = load_table("crop_area.csv")
if M.empty:
    missing("model_rows.parquet")
    st.stop()

REFERENCE = "XGBoost + memory (reference)"   # SPEC model family rule: comparison only, never the model
MODELS = [c for c in (MAIN_LABEL, NAME_I, NAME_T, NAME_P, "v12", "v15b", "v14", "v13",
                       "v10", "Trend + memory",
                       "5-year average", "Long-term average", REFERENCE)
          if c in M.columns]
SHORT = {MAIN_LABEL: MAIN, NAME_I: "YF-DE-i", NAME_T: "YF-DE-T", NAME_P: "YF-DE-P", "5-year average": "5-yr avg", "Long-term average": "LT avg",
         "Trend + memory": "trend+mem", REFERENCE: "XGB ref"}
INK, MUTED_INK = "#1f1f1d", "#6b6a65"
CFG = {"displayModeBar": False}
st.markdown("""<style>
.block-container {padding-top: 1.2rem; padding-bottom: 1rem;}
.tiles {display: grid; grid-template-columns: repeat(6, 1fr); gap: 12px; margin: 4px 0 10px 0;}
.tile {border: 1px solid #e1e0d9; border-radius: 10px; padding: 10px 14px; background: #fcfcfb;}
.tile .l {font-size: 0.8rem; color: #6b6a65;}
.tile .v {font-size: 1.6rem; font-weight: 600; color: #1f1f1d; line-height: 1.25;}
.tile .s {font-size: 0.78rem; color: #6b6a65;}
.tile .up {color: #1a7f4b; font-weight: 600;} .tile .down {color: #b3261e; font-weight: 600;}
.score {width: 100%; border-collapse: collapse; font-size: 0.86rem;}
.score th {text-align: right; color: #6b6a65; font-weight: 500; padding: 6px 8px; border-bottom: 1px solid #e1e0d9;}
.score th:first-child, .score td:first-child {text-align: left;}
.score td {text-align: right; padding: 6px 8px; border-bottom: 1px solid #f0efea; font-variant-numeric: tabular-nums;}
.score tr.main td {background: #eef4fd;} .score b {color: #184f95;}
.score td.ours {background: #eef4fd;} .score .up {color: #1a7f4b; font-weight: 600;}
.score .down {color: #b3261e; font-weight: 600;}
</style>""", unsafe_allow_html=True)


def short(m: str) -> str:
    return SHORT.get(m, m)


def acc(p, t) -> float:
    return float(100 - 100 * np.mean(np.abs(p - t) / t)) if len(t) else float("nan")


def tight(fig, h=300, **kw):
    fig.update_layout(height=h, margin=dict(l=6, r=6, t=8, b=6), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(color=INK, size=11),
                      legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, font=dict(size=10)), **kw)
    fig.update_xaxes(gridcolor="#ecebe6", zeroline=False, linecolor="#e1e0d9")
    fig.update_yaxes(gridcolor="#ecebe6", zeroline=False, linecolor="#e1e0d9")
    return fig


def tile(label: str, value: str, sub: str = "", tip: str = "") -> str:
    return (f"<div class='tile' title='{html.escape(tip, quote=True)}'><div class='l'>{label}</div>"
            f"<div class='v'>{value}</div><div class='s'>{sub}</div></div>")


def title(text: str, tip: str = "") -> None:
    st.markdown(f"**{text}**", help=tip or None)


def dumbbell(items, ref_vals, model_vals, h):
    """Model (dot) vs comparison (tick) per item: the right form for 'better or worse than the reference'."""
    fig = go.Figure()
    for y, a, b in zip(items, ref_vals, model_vals):
        fig.add_trace(go.Scatter(x=[a, b], y=[y, y], mode="lines", line=dict(color="#d4d3cd", width=2),
                                 showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=ref_vals, y=items, mode="markers", name=ref, marker=dict(
        color=AVG5, symbol="line-ns-open", size=12, line=dict(width=3, color=AVG5)),
        hovertemplate="%{y}: %{x:.1f}%<extra>" + ref + "</extra>"))
    fig.add_trace(go.Scatter(x=model_vals, y=items, mode="markers", name=model, marker=dict(color=MODEL, size=10),
                             hovertemplate="%{y}: %{x:.1f}%<extra>" + model + "</extra>"))
    return tight(fig, h, xaxis=dict(ticksuffix="%"))


def by(df: pd.DataFrame, key: str) -> pd.DataFrame:
    return df.groupby(key).apply(lambda d: pd.Series({"m": acc(d[model], d.actual), "r": acc(d[ref], d.actual),
                                                      "n": len(d)}), include_groups=False)


# --------------------------------------------------------------------------- #
# Global filters (one place, scope everything)
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("### Filters")
    model = st.selectbox("Model", [m for m in MODELS if m != REFERENCE], index=0, key="bi_model",
                         help=f"{MAIN_LABEL} is the final model: the average of three transformers (iTransformer, TFT, PatchTST) "
                              "plus the district error memory. All models are scored on the same rows.")
    others = [m for m in MODELS if m != model]
    ref = st.selectbox("Compare with", others, index=others.index("5-year average") if "5-year average" in others else 0,
                       key="bi_ref", help=f"{REFERENCE}: a strong tree model on the same inputs, shown for reference "
                                          "only; it is not one of our candidate models.")
    crops = st.multiselect("Crops", CROPS, default=[], format_func=crop_label, key="bi_crops", placeholder="All crops")
    years = st.slider("Years", 2010, 2025, (2010, 2025), key="bi_years")
    states = st.multiselect("States", sorted(M.state.dropna().unique()), key="bi_states", placeholder="All states")
    dlist = M[M.state.isin(states)] if states else M
    dnames = ["All districts"] + sorted(dlist.district.dropna().unique())
    # a map click or drill-down sets a pending district; apply it before the widget is created
    if "_bi_district_pending" in st.session_state:
        st.session_state["bi_district"] = st.session_state.pop("_bi_district_pending")
    if st.session_state.get("bi_district") not in dnames:
        st.session_state["bi_district"] = "All districts"
    district = st.selectbox("District", dnames, key="bi_district", help="Or click a district on the map.")
    if st.button("Reset filters", use_container_width=True):
        for k in ("bi_crops", "bi_states", "bi_years", "bi_district"):
            st.session_state.pop(k, None)
        st.rerun()

F = M[M.year.between(*years)]
if crops:
    F = F[F.crop.isin(crops)]
if states:
    F = F[F.state.isin(states)]
FD = F if district == "All districts" else F[F.district == district]
one_crop = crops[0] if len(crops) == 1 else None

scope = (", ".join(crop_label(c) for c in crops) if crops else "all crops") + f" · {years[0]}–{years[1]}" + \
        (f" · {', '.join(states)}" if states else "") + (f" · {district}" if district != "All districts" else "")
st.markdown(f"## {MODEL_NAME} dashboard")
st.caption(f"**{model}** compared with **{ref}** · {scope}")
if FD.empty:
    st.info("No estimates match the filters.")
    st.stop()

# --------------------------------------------------------------------------- #
# KPI tiles: value + plain reference line; colour only on a real gain or loss
# --------------------------------------------------------------------------- #
err = 100 * (FD[model] - FD.actual) / FD.actual
err_r = 100 * (FD[ref] - FD.actual) / FD.actual
nat = FD.groupby(["crop", "year"])[[model, ref, "actual", "Long-term average"]].mean()
nat_acc, nat_ref = acc(nat[model], nat.actual), acc(nat[ref], nat.actual)
bad = nat.actual / nat["Long-term average"] - 1 <= -0.10
warned = nat[model] / nat["Long-term average"] - 1 <= -0.05
gain = err_r.abs().mean() - err.abs().mean()
cls = "up" if gain > 0 else "down"
st.markdown("<div class='tiles'>" + "".join([
    tile("Accuracy", f"{100 - err.abs().mean():.1f}%", f"<span class='{cls}'>{gain:+.1f} pts</span> vs {short(ref)}",
         "100% minus the typical (mean absolute) error in percent, district level."),
    tile("Typical error", f"{err.abs().mean():.1f}%", f"{short(ref)}: {err_r.abs().mean():.1f}%",
         "Mean absolute difference between estimate and official yield, in percent."),
    tile("Within ±10%", f"{100 * (err.abs() <= 10).mean():.0f}%", f"{short(ref)}: {100 * (err_r.abs() <= 10).mean():.0f}%",
         "Share of district estimates within ±10% of the official yield."),
    tile("National accuracy", f"{nat_acc:.1f}%", f"{short(ref)}: {nat_ref:.1f}%",
         "Plain average over the districts in the selection, per crop and year."),
    tile("Bad years warned", f"{int((bad & warned).sum())} of {int(bad.sum())}", f"{int((~bad & warned).sum())} false warnings",
         "Bad year: the selection's yield ≥10% below normal. Warned: estimate ≥5% below normal."),
    tile("Estimates", f"{len(FD):,}", f"{FD.landkreis.nunique()} districts · {FD.crop.nunique()} crops",
         "District × crop × year estimates in the selection."),
]) + "</div>", unsafe_allow_html=True)

tabs = st.tabs(["Overview", "Drill-down table", "Accuracy", "Errors", "Model comparison", "Weather", "Crop areas",
                "Trends"])
T_OVER, T_DRILL, T_ACC, T_ERR, T_MOD, T_WEA, T_AREA, T_TREND = tabs

# --------------------------------------------------------------------------- #
# Overview — Where is the model better? Over time? Per crop? How close?
# --------------------------------------------------------------------------- #
with T_OVER:
    c1, c2 = st.columns([1, 1.25])
    with c1, st.container(border=True):
        metric = st.radio("Map", ["Gain vs comparison", "Accuracy", "Mean yield"], horizontal=True,
                          key="bi_mapm", label_visibility="collapsed")
        g = F.groupby("landkreis").apply(lambda d: pd.Series({
            "acc": acc(d[model], d.actual), "acc_r": acc(d[ref], d.actual),
            "yield": d.actual.mean() if one_crop else np.nan}), include_groups=False).reset_index()
        g["gain"] = g.acc - g.acc_r
        fig = None
        if metric == "Gain vs comparison":
            lim = float(np.nanpercentile(g.gain.abs(), 95)) if len(g) else 5
            fig = district_map(g.assign(v=g.gain), "v", label="pts", zmid=0, zrange=(-lim, lim), height=430,
                               colorscale=[[0, BAD], [0.5, "#f0efec"], [1, MODEL]], value_fmt="+.1f")
            note = "Blue = model more accurate than the comparison, red = less."
        elif metric == "Accuracy":
            fig = district_map(g, "acc", label="%", colorscale=SEQ_BLUE, height=430,
                               zrange=(float(np.nanpercentile(g.acc, 5)), float(np.nanpercentile(g.acc, 95))))
            note = "Darker = more accurate."
        else:
            note = "Darker = higher yield." if one_crop else "Select one crop to map yields."
            if one_crop:
                fig = district_map(g.dropna(subset=["yield"]), "yield", label="dt/ha", colorscale=SEQ_BLUE, height=430)
        if fig is not None:
            ev = st.plotly_chart(fig, use_container_width=True, config=CFG, on_select="rerun",
                                 selection_mode="points", key="bi_map")
            pts = getattr(getattr(ev, "selection", None), "points", None) if ev is not None else None
            if pts is None and isinstance(ev, dict):
                pts = ev.get("selection", {}).get("points", [])
            if pts:
                loc = pts[0].get("location")
                name = M.loc[M.landkreis == loc, "district"].iloc[0] if loc in set(M.landkreis) else None
                if name and name != st.session_state.get("bi_district"):
                    st.session_state["_bi_district_pending"] = name
                    st.rerun()
        st.caption(note + " Click a district to filter.")
    with c2:
        with st.container(border=True):
            if one_crop:
                title("Real yield vs estimate", "National (or selection) average per year, dt/ha.")
                n = FD.groupby("year")[[model, ref, "actual"]].mean()
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=n.index, y=n[ref], name=ref, line=dict(color=AVG5, width=2)))
                fig.add_trace(go.Scatter(x=n.index, y=n.actual, name="Real yield", mode="lines+markers",
                                         line=dict(color=INK, width=2.5), marker=dict(size=7)))
                fig.add_trace(go.Scatter(x=n.index, y=n[model], name=model, mode="lines+markers",
                                         line=dict(color=MODEL, width=2.5), marker=dict(size=7)))
                tight(fig, 250, yaxis_title="dt/ha", hovermode="x unified")
            else:
                title("Accuracy by year", "District-level accuracy per year. Select one crop to see yields in dt/ha.")
                y = by(FD, "year")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=y.index, y=y.r, name=ref, line=dict(color=AVG5, width=2)))
                fig.add_trace(go.Scatter(x=y.index, y=y.m, name=model, mode="lines+markers", line=dict(color=MODEL, width=2.5)))
                tight(fig, 250, yaxis=dict(title="accuracy", ticksuffix="%"), hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True, config=CFG)
        a1, a2 = st.columns(2)
        with a1, st.container(border=True):
            title("Accuracy by crop", "Dot = model, tick = comparison. The line shows the gap.")
            c = by(FD, "crop")
            c = c.reindex([x for x in CROPS if x in c.index][::-1])     # fixed crop order, no ranking
            st.plotly_chart(dumbbell([crop_label(x) for x in c.index], c.r, c.m, 250), use_container_width=True, config=CFG)
        with a2, st.container(border=True):
            title("How close the estimates are", "Share of district estimates within ±5%, ±10% and ±20% of the "
                  "official yield.")
            th = [5, 10, 20]
            vm = [100 * (err.abs() <= t).mean() for t in th]
            vr = [100 * (err_r.abs() <= t).mean() for t in th]
            fig = go.Figure()
            fig.add_trace(go.Bar(x=[f"within ±{t}%" for t in th], y=vr, name=ref, marker_color=AVG5,
                                 text=[f"{v:.0f}%" for v in vr], textposition="outside", cliponaxis=False))
            fig.add_trace(go.Bar(x=[f"within ±{t}%" for t in th], y=vm, name=model, marker_color=MODEL,
                                 text=[f"{v:.0f}%" for v in vm], textposition="outside", cliponaxis=False))
            st.plotly_chart(tight(fig, 250, barmode="group", bargap=0.35, yaxis=dict(range=[0, 105], ticksuffix="%")),
                            use_container_width=True, config=CFG)

# --------------------------------------------------------------------------- #
# Drill-down table — Crop → State → District → Year, click a row to go deeper
# --------------------------------------------------------------------------- #
PAGE = 15
with T_DRILL:
    ss = st.session_state
    for k in ("dd_crop", "dd_state", "dd_district"):
        ss.setdefault(k, None)
    if one_crop:
        ss.dd_crop = one_crop
    if len(states) == 1:
        ss.dd_state = states[0]
    elif ss.dd_state and states and ss.dd_state not in states:
        ss.dd_state = ss.dd_district = None
    if ss.dd_crop and crops and ss.dd_crop not in crops:
        ss.dd_crop = ss.dd_state = ss.dd_district = None
    if district != "All districts":
        row0 = F[F.district == district]
        if len(row0):
            ss.dd_district = district
            ss.dd_state = row0.state.iloc[0]
    D = F
    if ss.dd_crop:
        D = D[D.crop == ss.dd_crop]
    if ss.dd_state:
        D = D[D.state == ss.dd_state]
    if ss.dd_district:
        D = D[D.district == ss.dd_district]
    if not ss.dd_crop and not one_crop:
        level, key, label = "crop", "crop", "Crop"
    elif not ss.dd_state:
        level, key, label = "state", "state", "State"
    elif not ss.dd_district:
        level, key, label = "district", "district", "District"
    else:
        level, key, label = "year", "year", "Year"

    # breadcrumb: click to go back up
    crumbs = [("All crops", dict(dd_crop=None, dd_state=None, dd_district=None))]
    if ss.dd_crop:
        crumbs.append((crop_label(ss.dd_crop), dict(dd_state=None, dd_district=None)))
    if ss.dd_state:
        crumbs.append((ss.dd_state, dict(dd_district=None)))
    if ss.dd_district:
        crumbs.append((ss.dd_district, {}))
    bc = st.columns([1.2] * len(crumbs) + [6 - 1.2 * len(crumbs) if len(crumbs) < 5 else 0.1])
    for i, (name, reset) in enumerate(crumbs):
        last = i == len(crumbs) - 1
        if bc[i].button(("› " if i else "") + name, key=f"dd_bc_{i}", disabled=last, use_container_width=True,
                        type="secondary"):
            if one_crop and "dd_crop" in reset:
                reset.pop("dd_crop")
            for k, v in reset.items():
                ss[k] = v
            if "dd_district" in reset and district != "All districts":
                ss["_bi_district_pending"] = "All districts"
            ss["dd_page"] = 0
            st.rerun()

    if D.empty:
        st.info("No estimates for this selection.")
    else:
        e_m = 100 * (D[model] - D.actual) / D.actual
        e_r = 100 * (D[ref] - D.actual) / D.actual
        D = D.assign(e_m=e_m, e_r=e_r)
        if level != "year":
            agg = D.groupby(key).apply(lambda d: pd.Series({
                "Estimates": len(d), "Real yield": d.actual.mean(), "Estimate": d[model].mean(),
                "Accuracy": 100 - d.e_m.abs().mean(), "Comparison accuracy": 100 - d.e_r.abs().mean(),
                "Within ±10%": 100 * (d.e_m.abs() <= 10).mean(),
                "Years improved": 100 * np.mean([(g_.e_m.abs().mean() < g_.e_r.abs().mean())
                                                  for _, g_ in d.groupby("year")])}), include_groups=False).reset_index()
            agg["Gain"] = agg["Accuracy"] - agg["Comparison accuracy"]
            agg[label] = agg[key].map(crop_label) if key == "crop" else agg[key]
            show_yield = bool(ss.dd_crop)          # averages of dt/ha only make sense within one crop
        else:
            agg = D.sort_values("year").assign(Year=D.year.astype(str), **{
                "Real yield": D.actual, "Estimate": D[model], "Comparison": D[ref], "Error": D.e_m,
                "Comparison error": D.e_r})
            if not ss.dd_crop:
                agg["Crop"] = agg.crop.map(crop_label)

        c1, c2, c3 = st.columns([2, 2, 3])
        q = c1.text_input("Search", key=f"dd_q_{level}", placeholder=f"Find a {label.lower()}…",
                          label_visibility="collapsed") if level in ("state", "district") else ""
        sorts = {"Standard order": ("_order", True), "Most estimates": ("Estimates", False)} \
            if level != "year" else {"Year": ("year", True)}
        sort = c2.selectbox("Sort", list(sorts), key=f"dd_sort_{level}", label_visibility="collapsed")
        if q:
            agg = agg[agg[label].astype(str).str.contains(q, case=False)]
        col_, asc = sorts[sort]
        if col_ == "_order":                     # crops in the fixed crop order, states/districts A-Z
            agg = agg.assign(_order=agg[key].map({c: i for i, c in enumerate(CROPS)}) if key == "crop" else agg[label])
        agg = agg.sort_values(col_, ascending=asc)
        pages = max(1, int(np.ceil(len(agg) / PAGE)))
        ss.setdefault("dd_page", 0)
        if ss.get("dd_last") != (level, ss.dd_crop, ss.dd_state, ss.dd_district, sort, q):
            ss.dd_page, ss.dd_last = 0, (level, ss.dd_crop, ss.dd_state, ss.dd_district, sort, q)
        page = min(ss.dd_page, pages - 1)
        with c3:
            p1, p2, p3 = st.columns([1, 2, 1])
            if p1.button("◀", key="dd_prev", disabled=page == 0, use_container_width=True):
                ss.dd_page = page - 1
                st.rerun()
            p2.markdown(f"<div style='text-align:center;padding-top:6px;color:#6b6a65'>{len(agg)} row{'s' if len(agg) != 1 else ''} · page "
                        f"{page + 1} of {pages}</div>", unsafe_allow_html=True)
            if p3.button("▶", key="dd_next", disabled=page >= pages - 1, use_container_width=True):
                ss.dd_page = page + 1
                st.rerun()
        view = agg.iloc[page * PAGE:(page + 1) * PAGE]

        if level != "year":
            cols = [label, "Estimates"] + (["Real yield", "Estimate"] if show_yield else []) + \
                   ["Accuracy", "Comparison accuracy", "Gain", "Years improved", "Within ±10%"]
            cfg = {label: st.column_config.TextColumn(label, width="medium"),
                   "Estimates": st.column_config.NumberColumn("Estimates", format="%d"),
                   "Real yield": st.column_config.NumberColumn("Real yield (dt/ha)", format="%.1f"),
                   "Estimate": st.column_config.NumberColumn(f"{short(model)} (dt/ha)", format="%.1f"),
                   "Accuracy": st.column_config.NumberColumn(f"Accuracy {short(model)}", format="%.1f%%"),
                   "Comparison accuracy": st.column_config.NumberColumn(f"Accuracy {short(ref)}", format="%.1f%%"),
                   "Gain": st.column_config.NumberColumn("Gain (pts)", format="%+.1f"),
                   "Years improved": st.column_config.NumberColumn(
                       "Years improved", format="%.0f%%",
                       help="Share of years in which the model was more accurate than the comparison."),
                   "Within ±10%": st.column_config.ProgressColumn("Within ±10%", format="%.0f%%", min_value=0,
                                                                   max_value=100)}
            nxt = {"crop": "its states", "state": "its districts", "district": "the years of that district"}[level]
            hint = f"Click a row to see {nxt}."
        else:
            cols = ["Year"] + (["Crop"] if not ss.dd_crop else []) + ["Real yield", "Estimate", "Comparison", "Error",
                                                                       "Comparison error"]
            cfg = {"Year": st.column_config.TextColumn("Year", width="small"),
                   "Real yield": st.column_config.NumberColumn("Real (dt/ha)", format="%.1f"),
                   "Estimate": st.column_config.NumberColumn(f"{short(model)} (dt/ha)", format="%.1f"),
                   "Comparison": st.column_config.NumberColumn(f"{short(ref)} (dt/ha)", format="%.1f"),
                   "Error": st.column_config.NumberColumn(f"Error {short(model)}", format="%+.1f%%"),
                   "Comparison error": st.column_config.NumberColumn(f"Error {short(ref)}", format="%+.1f%%")}
            hint = "Bottom level: each row is one official yield and its estimates."
        ev = st.dataframe(view[cols], hide_index=True, width="stretch", height=35 * (len(view) + 1) + 3,
                          column_config=cfg, on_select="rerun" if level != "year" else "ignore",
                          selection_mode="single-row", key=f"dd_tbl_{level}_{page}_{sort}")
        st.caption(hint + " Global filters (sidebar) apply.")
        rows = getattr(getattr(ev, "selection", None), "rows", []) if level != "year" else []
        if rows:
            picked = view.iloc[rows[0]][key]
            ss[{"crop": "dd_crop", "state": "dd_state", "district": "dd_district"}[level]] = picked
            st.rerun()
        st.download_button("⬇ Download this table (CSV)", agg[cols].to_csv(index=False).encode("utf-8"),
                           file_name=f"yieldformer_de_{level}.csv", mime="text/csv")

# --------------------------------------------------------------------------- #
# Accuracy — When does the model help? Where (state)? Is it calibrated?
# --------------------------------------------------------------------------- #
with T_ACC:
    c1, c2 = st.columns([1.3, 1])
    with c1, st.container(border=True):
        title("Gain over the comparison by year", "Model accuracy minus comparison accuracy, percentage points. "
              "Blue = model better that year, red = worse.")
        y = by(FD, "year")
        gy = y.m - y.r
        fig = go.Figure(go.Bar(x=gy.index, y=gy.values, marker_color=[MODEL if v >= 0 else BAD for v in gy.values],
                               text=[f"{v:+.1f}" for v in gy.values], textposition="outside", cliponaxis=False,
                               hovertemplate="%{x}: %{y:+.2f} pts<extra></extra>"))
        fig.add_hline(y=0, line_color="#b9b8b2", line_width=1)
        st.plotly_chart(tight(fig, 470, yaxis=dict(title="points")), use_container_width=True, config=CFG)
    with c2, st.container(border=True):
        title("Accuracy by state", "Dot = model, tick = comparison.")
        s = by(FD, "state").sort_index(ascending=False)                  # alphabetical, top to bottom
        st.plotly_chart(dumbbell(list(s.index), s.r, s.m, 470), use_container_width=True, config=CFG)

# --------------------------------------------------------------------------- #
# Errors — Systematic bias? Data quality? Worst cases?
# --------------------------------------------------------------------------- #
with T_ERR:
    c1, c2 = st.columns(2)
    with c1, st.container(border=True):
        title("Over- / under-estimate by year", "Mean signed error. Red = estimates too high (the year was worse "
              "than estimated), blue = too low.")
        b = err.groupby(FD.year).mean()
        fig = go.Figure(go.Bar(x=b.index, y=b.values, marker_color=[BAD if v > 0 else MODEL for v in b.values],
                               hovertemplate="%{x}: %{y:+.1f}%<extra></extra>"))
        fig.add_hline(y=0, line_color="#b9b8b2", line_width=1)
        st.plotly_chart(tight(fig, 240, yaxis=dict(ticksuffix="%")), use_container_width=True, config=CFG)
    with c2, st.container(border=True):
        title("Error vs crop's share of farmland", "Where a crop is rare, the official yields rest on few farms "
              "and are noisy.")
        if not AREA.empty:
            x = FD.merge(AREA[["landkreis", "crop", "share_of_farmland"]], on=["landkreis", "crop"], how="left") \
                .dropna(subset=["share_of_farmland"])
            if len(x) > 50:
                x["bin"] = pd.qcut(x.share_of_farmland.rank(method="first"), 5,
                                   labels=["smallest 20%", "2nd 20%", "middle 20%", "4th 20%", "largest 20%"])
                e2 = x.groupby("bin", observed=True).apply(lambda d: pd.Series({
                    "m": (100 * (d[model] - d.actual) / d.actual).abs().mean(),
                    "r": (100 * (d[ref] - d.actual) / d.actual).abs().mean()}), include_groups=False)
                fig = go.Figure()
                fig.add_trace(go.Bar(x=e2.index.astype(str), y=e2.r, name=ref, marker_color=AVG5))
                fig.add_trace(go.Bar(x=e2.index.astype(str), y=e2.m, name=model, marker_color=MODEL))
                st.plotly_chart(tight(fig, 240, barmode="group", bargap=0.3,
                                      yaxis=dict(title="typical error", ticksuffix="%")),
                                use_container_width=True, config=CFG)
            else:
                st.caption("Too few estimates in the selection.")
    with st.container(border=True):
        title("Data-quality check: estimates far from the official figure",
              "The 10 records where estimate and official yield differ most. Most are implausible official figures "
              "for crops grown on tiny areas (e.g. silage maize 45 dt/ha where 400–500 is normal), not model failures. "
              "Hover for the values.")
        w = FD.assign(error=err).reindex(err.abs().sort_values(ascending=False).index).head(10).iloc[::-1]
        lab = w.district.str.replace(", Kreisfreie Stadt", " (city)").str.replace(", Landkreis", "") + " · " + \
            w.crop.map(crop_label) + " " + w.year.astype(str)
        fig = go.Figure(go.Bar(y=lab, x=w.error, orientation="h", marker_color=[BAD if v > 0 else MODEL for v in w.error],
                               text=[f"{v:+.0f}%" for v in w.error], textposition="outside", cliponaxis=False,
                               customdata=np.c_[w.actual, w[model]],
                               hovertemplate="%{y}<br>real %{customdata[0]:.1f}, estimate %{customdata[1]:.1f} dt/ha"
                                             "<extra></extra>"))
        st.plotly_chart(tight(fig, 270, xaxis=dict(title="error (%)")), use_container_width=True, config=CFG)

# --------------------------------------------------------------------------- #
# Model comparison — Which model, by how much, on which measure?
# --------------------------------------------------------------------------- #
with T_MOD:
    view_m = st.radio("Compare against", ["Published models", "Our versions & simple averages"], horizontal=True,
                      key="bi_mview", label_visibility="collapsed")
    if view_m == "Published models":
        PP = load_table("papers.csv")
        PAPERS = {
            "AgroForecast-DE": ("AgroForecast-DE (same districts)", "AgroForecast-DE",
                                "The CropFusionNet team's public benchmark for German districts. Test years 2023–24, "
                                "scored on exactly their published district-years."),
            "CropFusionNet": ("CropFusionNet paper", "CropFusionNet",
                              "Transformer-style model (Srivastava et al., ZALF, 2026). Test years 2022–24; their "
                              "headline score most likely also counts the tuning years 2019–21, so both are shown."),
            "Srivastava 2022": ("Srivastava 2022 CNN", "Srivastava 2022 (CNN)",
                                "Convolutional network for winter wheat in 271 counties, one test year at a time. "
                                "Their yields appear to be dry matter (~14% lower), so their errors look ~14% smaller."),
        }
        METRIC = {"r2": ("R²", True), "rmse": ("RMSE (dt/ha)", False), "mae": ("MAE (dt/ha)", False),
                  "mape": ("MAPE (%)", False)}

        # the final model (v18 ensemble) once its runs on the papers' splits exist; until then the
        # iTransformer versions (v12, and v15 where its error memory is allowed)
        ENS_ON_PAPERS = bool((PP.version == MAIN).any())

        def ours(paper_key: str) -> pd.DataFrame:
            if ENS_ON_PAPERS:
                base = PP[(PP.version == MAIN) & (PP.paper == paper_key)].copy()
                base["ours"], base["ours_is_v15"] = base.ours_equal_protocol, False
            else:
                base = PP[(PP.version == "v12") & (PP.paper == paper_key)].copy()
                v15 = PP[(PP.version == "v15") & (PP.paper == paper_key)].set_index(["crop", "years", "metric"])
                k = list(zip(base.crop, base.years, base.metric))
                v = [v15.ours_equal_protocol.get(t, np.nan) if len(v15) else np.nan for t in k]
                base["ours"] = np.where(pd.notna(v), v, base.ours_equal_protocol)
                base["ours_is_v15"] = pd.notna(v)
            hb = base.metric.map(lambda m: METRIC[m][1])
            base["win"] = np.where(hb, base.ours > base.theirs, base.ours < base.theirs)
            base["label"] = base.crop.map(crop_label) + " · " + base.years.astype(str).str.replace(
                r" \(.*\)", "", regex=True).str.replace(
                "2019-24 pooled incl. our validation years 2019-21, as their code", "2019-24 pooled", regex=False)
            return base[base.crop.isin(crops)] if crops else base

        allp = {k: ours(v[0]) for k, v in PAPERS.items()}
        tot_w = sum(int(d.win.sum()) for d in allp.values())
        tot_n = sum(len(d) for d in allp.values())
        st.markdown("<div class='tiles' style='grid-template-columns: repeat(4, 1fr)'>" + "".join(
            [tile(f"vs {k}", f"{int(d.win.sum())} of {len(d)}" if len(d) else "–", "comparisons won by YieldFormer-DE",
                  PAPERS[k][2]) for k, d in allp.items()] +
            [tile("All published models", f"{tot_w} of {tot_n}", "comparisons won",
                  "Each comparison = one crop × test period × measure, on the paper's own split.")]) + "</div>",
            unsafe_allow_html=True)
        pick = st.radio("Paper", list(PAPERS), horizontal=True, key="bi_paper", label_visibility="collapsed")
        d = allp[pick]
        pname = PAPERS[pick][1]
        if d.empty:
            st.info("No comparison for the selected crops in this paper.")
        else:
            st.caption(PAPERS[pick][2] + (
                f" {MAIN_LABEL}* = the three-transformer average without the error memory: this paper's test setup "
                "freezes information at the end of training, so last year's yields cannot be used." if ENS_ON_PAPERS else
                " Shown: the iTransformer version (v15) where its error memory is allowed, else v12 (*, without the "
                "memory). The final model's runs on this paper's split are in progress."))
            mets = [m for m in METRIC if m in set(d.metric)]
            fig = make_subplots(rows=1, cols=len(mets), horizontal_spacing=0.08,
                                subplot_titles=[f"{METRIC[m][0]} · {'higher' if METRIC[m][1] else 'lower'} is better"
                                                for m in mets])
            ours_name = (f"{MAIN_LABEL}*" if ENS_ON_PAPERS else
                         NAME_I if d.ours_is_v15.all() else "iTransformer without memory (v12)*")
            for i, m in enumerate(mets, 1):
                dm = d[d.metric == m]
                fmt = "{:.2f}" if m == "r2" else "{:.1f}"
                for colname, name, color, show in (("theirs", pname, "#eda100", i == 1),
                                                   ("ours", ours_name, MODEL, i == 1),
                                                   ("hist_avg", "Historical average", "#c9c8c2", i == 1)):
                    fig.add_trace(go.Bar(x=dm.label, y=dm[colname], name=name, marker_color=color, showlegend=show,
                                         marker_line=dict(color="#184f95" if colname == "ours" else color,
                                                          width=2 if colname == "ours" else 0),
                                         text=[fmt.format(v) for v in dm[colname]], textposition="outside",
                                         cliponaxis=False, textfont=dict(size=10)), row=1, col=i)
                top = float(dm[["theirs", "ours", "hist_avg"]].max().max())
                fig.update_yaxes(range=[0, top * 1.2], row=1, col=i)
            fig.update_annotations(font=dict(size=11, color=INK))
            tight(fig, 330, barmode="group", bargap=0.25)
            fig.update_layout(margin=dict(l=6, r=6, t=40, b=6), legend=dict(y=1.18))
            with st.container(border=True):
                title(f"YieldFormer-DE vs {pname}", "Same training and test years as the paper. Blue (outlined) = our "
                      "model; orange = the published model; grey = a simple historical average.")
                st.plotly_chart(fig, use_container_width=True, config=CFG)
                rows_html = "".join(
                    f"<tr><td>{html.escape(r.label)}</td><td>{METRIC[r.metric][0]}</td>"
                    f"<td>{r.theirs:.3f}</td><td class='ours'><b>{r.ours:.3f}</b>{'' if r.ours_is_v15 else '*'}</td>"
                    f"<td>{r.hist_avg:.3f}</td>"
                    f"<td>{'<span class=up>✔ ours better</span>' if r.win else '<span class=down>✖ paper better</span>'}</td></tr>"
                    for r in d.itertuples())
                st.markdown("<table class='score'><tr><th>Crop · test years</th><th>Measure</th>"
                            f"<th>{html.escape(pname)}</th><th>{ours_name}</th><th>Historical avg</th><th>Result</th></tr>"
                            f"{rows_html}</table>", unsafe_allow_html=True)
            st.caption("Published models report only these summary scores, so this comparison uses each paper's own crops "
                       "and test years; the sidebar year/state filters do not apply here (crop filter does).")
    else:
        rows = []
        for m in MODELS:
            e_ = 100 * (FD[m] - FD.actual) / FD.actual
            nn = FD.groupby(["crop", "year"])[[m, "actual"]].mean()
            rows.append({"Model": m, "Accuracy": 100 - e_.abs().mean(), "Within ±10%": 100 * (e_.abs() <= 10).mean(),
                         "National": acc(nn[m], nn.actual), "Bias": e_.mean()})
        T = pd.DataFrame(rows)
        c1, c2 = st.columns([1.15, 1])
        with c1, st.container(border=True):
            title("All models on the selected estimates", "Same rows, same official yields. Axis zoomed: versions differ "
                  "by tenths of a point. Light blue = our transformer versions; grey = no neural network "
                  "(averages, trend + memory) and the XGBoost reference, which uses the same inputs and is shown for "
                  "comparison only, not as a candidate.")
            Ts = T.sort_values("Accuracy")
            lab = [("★ " + m) if m == MAIN_LABEL else m for m in Ts.Model]
            fig = go.Figure(go.Bar(y=lab, x=Ts.Accuracy, orientation="h",
                                   marker_color=[MODEL if m == MAIN_LABEL else
                                                 ("#c9c8c2" if "average" in m or "memory" in m
                                                  else "#b8d2f3") for m in Ts.Model],
                                   text=[f"{v:.2f}%" for v in Ts.Accuracy], textposition="outside", cliponaxis=False))
            st.plotly_chart(tight(fig, 400, xaxis=dict(range=[float(Ts.Accuracy.min()) - 1.5,
                                                              float(Ts.Accuracy.max()) + 0.8], ticksuffix="%")),
                            use_container_width=True, config=CFG)
        with c2, st.container(border=True):
            title("Scorecard", "Best value per column in bold. Mean error: + = estimates too high on average.")
            Tt = T.sort_values("Accuracy", ascending=False)
            best = {"Accuracy": Tt.Accuracy.max(), "Within ±10%": Tt["Within ±10%"].max(), "National": Tt.National.max(),
                    "Bias": Tt.Bias.abs().min()}

            def cell(col, v):
                hit = abs(v) == best[col] if col == "Bias" else v == best[col]
                txt = f"{v:+.2f}%" if col == "Bias" else f"{v:.2f}%"
                return f"<td>{'<b>' + txt + '</b>' if hit else txt}</td>"
            body = "".join(f"<tr class='{'main' if r.Model == MAIN_LABEL else ''}'><td>"
                           f"{'★ ' if r.Model == MAIN_LABEL else ''}{html.escape(r.Model)}</td>"
                           + "".join(cell(c, r[c]) for c in ("Accuracy", "Within ±10%", "National", "Bias")) + "</tr>"
                           for _, r in Tt.iterrows())
            st.markdown("<table class='score'><tr><th>Model</th><th>Accuracy</th><th>Within ±10%</th><th>National</th>"
                        f"<th>Mean error</th></tr>{body}</table>", unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# Weather — Which weather drives this crop?
# --------------------------------------------------------------------------- #
with T_WEA:
    MET = ROOT / "results" / "metrics"
    if (MET / "rca_national_weather.csv").exists():
        W = pd.read_csv(MET / "rca_national_weather.csv").set_index("year")
        A = pd.read_csv(MET / "rca_national_anomaly.csv").set_index("year")
        cs = [c for c in (crops or CROPS) if c in A.columns]
        s1, s2 = st.columns(2)
        wc = s1.selectbox("Crop", cs, format_func=crop_label, key="bi_wcrop")
        r = pd.Series({w: W[w].corr(A[wc]) for w in W.columns}).sort_values()
        ws = s2.selectbox("Weather signal", list(W.columns), index=list(W.columns).index(r.abs().idxmax()),
                          format_func=lambda c: WEATHER_LABEL.get(c, c), key="bi_wsig")
        c1, c2 = st.columns(2)
        with c1, st.container(border=True):
            title(f"Which weather matters for {crop_label(wc).lower()}",
                  "Correlation between the national yield swing (vs trend) and seasonal weather, 2003–2025. Blue = more "
                  "of this weather goes with better yields, red = with worse. |r| above about 0.4 is unlikely to be chance.")
            fig = go.Figure(go.Bar(y=[WEATHER_LABEL.get(w, w) for w in r.index], x=r.values, orientation="h",
                                   marker_color=[MODEL if v > 0 else BAD for v in r.values],
                                   text=[f"{v:+.2f}" for v in r.values], textposition="outside", cliponaxis=False))
            fig.add_vline(x=0, line_color="#b9b8b2", line_width=1)
            st.plotly_chart(tight(fig, 360, xaxis=dict(range=[-1, 1], title="correlation r")), use_container_width=True,
                            config=CFG)
        with c2, st.container(border=True):
            title("Each year: weather vs yield", "Each dot is a year. Red = yield ≥10% below trend.")
            d = pd.DataFrame({"x": W[ws], "y": A[wc]}).dropna()
            fig = go.Figure(go.Scatter(x=d.x, y=d.y, mode="markers+text", text=d.index.astype(str),
                                       textposition="top center", textfont=dict(size=9, color=MUTED_INK),
                                       marker=dict(size=9, color=[BAD if v <= -10 else MODEL for v in d.y])))
            if len(d) > 2:
                bfit = np.polyfit(d.x, d.y, 1)
                xs = np.linspace(d.x.min(), d.x.max(), 20)
                fig.add_trace(go.Scatter(x=xs, y=bfit[0] * xs + bfit[1], mode="lines", hoverinfo="skip",
                                         line=dict(color="#9e9d97", width=1.5)))
            fig.add_hline(y=0, line_color="#d4d3cd", line_width=1)
            st.plotly_chart(tight(fig, 360, xaxis_title=WEATHER_LABEL.get(ws, ws), yaxis_title="yield vs trend (%)",
                                  showlegend=False), use_container_width=True, config=CFG)

# --------------------------------------------------------------------------- #
# Crop areas — Where is the crop grown?
# --------------------------------------------------------------------------- #
with T_AREA:
    if AREA.empty:
        st.info("Crop areas not built.")
    else:
        ac = st.selectbox("Crop", [c for c in (crops or CROPS) if c in set(AREA.crop)], format_func=crop_label,
                          key="bi_acrop")
        a = AREA[AREA.crop == ac]
        if states:
            a = a[a.state.isin(states)]
        st.markdown("<div class='tiles' style='grid-template-columns: repeat(4, 1fr)'>" + "".join([
            tile("Crop area", f"{a.area_ha.sum() / 1e3:,.0f}k ha", "latest agricultural census"),
            tile("Districts growing it", f"{(a.area_ha > 0).sum()}", ""),
            tile("Median share of farmland", f"{a.share_of_farmland.median():.1f}%", ""),
            tile("Average yield 2016–25", f"{a.yield_2016_25.mean():.1f} dt/ha", ""),
        ]) + "</div>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1, st.container(border=True):
            title("Crop area (1,000 ha)")
            fig = district_map(a.assign(v=a.area_ha / 1e3), "v", label="k ha", colorscale=SEQ_BLUE, height=380,
                               zrange=(0, float(np.nanpercentile(a.area_ha / 1e3, 97))))
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True, config=CFG)
        with c2, st.container(border=True):
            title("Share of farmland (%)")
            fig = district_map(a, "share_of_farmland", label="%", colorscale=SEQ_BLUE, height=380,
                               zrange=(0, float(np.nanpercentile(a.share_of_farmland, 97))))
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True, config=CFG)
        with c3, st.container(border=True):
            title("Top 12 districts", "By crop area; the label shows the share of farmland.")
            t = a.sort_values("area_ha", ascending=False).head(12).iloc[::-1]
            fig = go.Figure(go.Bar(y=t.district.str.replace(", Landkreis", "").str.replace(", Kreisfreie Stadt", " (city)"),
                                   x=t.area_ha / 1e3, orientation="h", marker_color=MODEL,
                                   text=[f"{v:.0f}%" for v in t.share_of_farmland], textposition="outside",
                                   cliponaxis=False))
            st.plotly_chart(tight(fig, 380, xaxis=dict(title="1,000 ha")), use_container_width=True, config=CFG)
        st.caption("Source: German agricultural census via Duden, Nacke & Offermann (2024), Scientific Data 11:95.")

# --------------------------------------------------------------------------- #
# Trends — Real vs estimate per crop over time; downloads
# --------------------------------------------------------------------------- #
with T_TREND:
    where = "Germany" if not states and district == "All districts" else ("the selection" if district == "All districts"
                                                                          else district)
    with st.container(border=True):
        title(f"Real vs estimate by crop — {where}", "Average yield per year (dt/ha): black = official, blue = model. "
              "Each crop on its own scale.")
        cl = [c for c in CROPS if c in set(FD.crop)]
        ncol = 5
        nrow = int(np.ceil(len(cl) / ncol))
        fig = make_subplots(rows=nrow, cols=ncol, subplot_titles=[crop_label(c) for c in cl],
                            horizontal_spacing=0.04, vertical_spacing=0.16)
        nn = FD.groupby(["crop", "year"])[[model, "actual"]].mean().reset_index()
        for i, c in enumerate(cl):
            d = nn[nn.crop == c]
            rr, kk = i // ncol + 1, i % ncol + 1
            fig.add_trace(go.Scatter(x=d.year, y=d.actual, mode="lines", line=dict(color=INK, width=2), name="Real",
                                     showlegend=i == 0, hovertemplate="%{x}: %{y:.1f}<extra>real</extra>"), row=rr, col=kk)
            fig.add_trace(go.Scatter(x=d.year, y=d[model], mode="lines", line=dict(color=MODEL, width=2), name=model,
                                     showlegend=i == 0, hovertemplate="%{x}: %{y:.1f}<extra>estimate</extra>"),
                          row=rr, col=kk)
        fig.update_annotations(font=dict(size=11, color=INK))
        fig.update_xaxes(nticks=3, tickfont=dict(size=9))
        fig.update_yaxes(nticks=4, tickfont=dict(size=9))
        tight(fig, 165 * nrow + 60)
        fig.update_layout(margin=dict(l=6, r=6, t=56, b=6),
                          legend=dict(orientation="h", yanchor="bottom", y=1.10, x=1, xanchor="right"))
        st.plotly_chart(fig, use_container_width=True, config=CFG)
    g = F.groupby("district").apply(lambda d: acc(d[model], d.actual), include_groups=False).dropna().sort_values()
    d1, d2, d3 = st.columns(3)
    d1.download_button("⬇ Every estimate, all models (CSV)", FD.to_csv(index=False).encode("utf-8"),
                       file_name="yieldformer_de_selection.csv", mime="text/csv", use_container_width=True)
    d2.download_button("⬇ Year × crop averages (CSV)", nn.to_csv(index=False).encode("utf-8"),
                       file_name="yieldformer_de_year_crop.csv", mime="text/csv", use_container_width=True)
    d3.download_button("⬇ Accuracy by district (CSV)", g.rename("accuracy").reset_index().to_csv(index=False).encode("utf-8"),
                       file_name="yieldformer_de_by_district.csv", mime="text/csv", use_container_width=True)

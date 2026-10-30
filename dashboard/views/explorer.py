"""Predictions vs actual: every estimate of the final model next to the official yield, year by year."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _results import CROPS, HELP, MODEL_NAME, crop_label, header, load_summary, load_table, missing  # noqa: E402

S = load_summary()
R = load_table("rows.parquet")
N = load_table("national.csv")
if R.empty or N.empty:
    missing("rows.parquet")
    st.stop()

header("Predictions vs actual",
       f"Every estimate of <b>{MODEL_NAME}</b> next to the official Destatis yield: {S['rows']:,} district estimates "
       f"for 10 crops, 2010–2025. Each year was estimated by a model trained only on earlier years. Filter, sort, "
       "and download the tables.")

# --------------------------------------------------------------------------- #
st.markdown("##### Year by year: Germany", help=HELP["national"])
crop = st.selectbox("Crop", CROPS, format_func=crop_label, key="ex_nat_crop")
n = N[N.crop == crop].sort_values("year").copy()
n["error"] = 100 * (n.predicted / n.actual - 1)
n["avg5_error"] = 100 * (n.avg5 / n.actual - 1)
n["kind"] = n["kind"].map({"bad": "bad year", "good": "good year", "normal": "normal"})
st.dataframe(n[["year", "actual", "predicted", "error", "avg5", "avg5_error", "actual_vs_normal", "kind", "n_districts"]],
             hide_index=True, width="stretch", height=38 * (len(n) + 1) + 3, column_config={
                 "year": st.column_config.NumberColumn("Year", format="%d"),
                 "actual": st.column_config.NumberColumn("Real yield (dt/ha)", format="%.1f"),
                 "predicted": st.column_config.NumberColumn(f"{MODEL_NAME} (dt/ha)", format="%.1f"),
                 "error": st.column_config.NumberColumn("Error", format="%+.1f%%"),
                 "avg5": st.column_config.NumberColumn("5-year average (dt/ha)", format="%.1f"),
                 "avg5_error": st.column_config.NumberColumn("5-yr avg error", format="%+.1f%%"),
                 "actual_vs_normal": st.column_config.NumberColumn("Real vs normal", format="%+.1f%%"),
                 "kind": "Year type", "n_districts": st.column_config.NumberColumn("Districts", format="%d")})
st.caption("National values are plain averages over the districts that reported the crop that year. "
           "Error = estimate vs real yield. " + HELP["dt_ha"])

# --------------------------------------------------------------------------- #
st.markdown("---")
st.markdown("##### Year × crop table")
view = st.radio("Show", ["Real yield (dt/ha)", f"{MODEL_NAME} estimate (dt/ha)", "Error (%)"], horizontal=True,
                key="ex_pivot")
col = {"Real yield (dt/ha)": "actual", f"{MODEL_NAME} estimate (dt/ha)": "predicted"}.get(view)
NN = N.assign(error=100 * (N.predicted / N.actual - 1))
pv = NN.pivot(index="year", columns="crop", values=col or "error").reindex(columns=[c for c in CROPS if c in set(N.crop)])
pv.columns = [crop_label(c) for c in pv.columns]
st.dataframe(pv.style.format("{:+.1f}%" if col is None else "{:.1f}"), width="stretch", height=38 * (len(pv) + 1) + 3)
st.caption("Germany-wide values per year and crop (plain averages over the reporting districts). Error: + = estimate "
           "too high.")

# --------------------------------------------------------------------------- #
st.markdown("---")
st.markdown("##### Every estimate")
c = st.columns(4)
f_crop = c[0].multiselect("Crop", CROPS, format_func=crop_label, key="ex_crop")
f_year = c[1].multiselect("Year", sorted(R.year.unique()), key="ex_year")
f_state = c[2].multiselect("State", sorted(R.state.dropna().unique()), key="ex_state")
f_text = c[3].text_input("District name contains", key="ex_text")
d = R
if f_crop:
    d = d[d.crop.isin(f_crop)]
if f_year:
    d = d[d.year.isin(f_year)]
if f_state:
    d = d[d.state.isin(f_state)]
if f_text:
    d = d[d.district.fillna("").str.contains(f_text, case=False)]
d = d.assign(crop=d.crop.map(crop_label))[["year", "crop", "district", "state", "actual", "predicted", "err_predicted",
                                           "avg5", "err_avg5", "landkreis"]].sort_values(["year", "crop", "district"])
k = st.columns(4)
k[0].metric("Estimates shown", f"{len(d):,}")
k[1].metric("Accuracy", f"{100 - d.err_predicted.abs().mean():.1f}%" if len(d) else "–")
k[2].metric("5-year average accuracy", f"{100 - d.err_avg5.abs().mean():.1f}%" if len(d) else "–")
k[3].metric("Within ±10%", f"{100 * (d.err_predicted.abs() <= 10).mean():.0f}%" if len(d) else "–")
st.dataframe(d, hide_index=True, width="stretch", height=520, column_config={
    "year": st.column_config.NumberColumn("Year", format="%d"), "crop": "Crop", "district": "District",
    "state": "State", "landkreis": "NUTS-3 code",
    "actual": st.column_config.NumberColumn("Real (dt/ha)", format="%.1f"),
    "predicted": st.column_config.NumberColumn(f"{MODEL_NAME} (dt/ha)", format="%.1f"),
    "err_predicted": st.column_config.NumberColumn("Error", format="%+.1f%%"),
    "avg5": st.column_config.NumberColumn("5-yr avg (dt/ha)", format="%.1f"),
    "err_avg5": st.column_config.NumberColumn("5-yr avg error", format="%+.1f%%")})
st.download_button("Download these estimates (CSV)", d.to_csv(index=False).encode("utf-8"),
                   file_name="yieldformer_de_estimates.csv", mime="text/csv")

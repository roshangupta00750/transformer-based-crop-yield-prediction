"""Supplementary multi-criteria recommendation (not paper headline).

Loads ``results/metrics/composite_ranking.csv`` (produced by
``scripts/15_composite_ranking.py``) and recomputes the weighted
composite in-page as the user drags the four weight sliders, so it is
visible how the recommendation shifts when climate robustness,
interpretability, or compute cost is upweighted against pure R².

This page is Chapter 7 supplementary material — the paper's headline
claim remains per-crop accuracy validated against the a priori
taxonomy (see page 13 Taxonomy). Read this as a "we also explored"
exhibit, not a replacement ranking.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import METRICS_DIR, divider, page_header

st.set_page_config(page_title="Recommendation — Crop Yield Benchmark",
                   layout="wide")
page_header("Supplementary — multi-criteria model recommendation",
            "Composite = w_acc * accuracy + w_climate * climate stability "
            "+ w_interp * interpretability + w_compute * compute cost. "
            "Drag the sliders to see how the winner shifts.")

st.markdown(
    "### TL;DR\n\n"
    "If accuracy alone doesn't drive the call, the per-crop winner shifts "
    "once climate robustness, interpretability, or compute cost carry real "
    "weight — drag the sliders to see it."
)

with st.expander("More details — supplementary caveat", expanded=False):
    st.caption(
        "Chapter 7 supplementary — we also explored what happens when "
        "criteria other than raw R² carry weight; the paper's headline "
        "remains accuracy-only taxonomy validation."
    )
    st.info(
        "**Not the paper's headline claim.** The composite ranking is a "
        "supplementary exploration of how non-accuracy criteria (climate "
        "robustness under CMIP6 SSP2-4.5/5-8.5, architecture "
        "interpretability, training compute cost) would shift the "
        "per-crop model recommendation if they carried real weight. The "
        "paper's headline remains per-crop accuracy validated against "
        "the a priori taxonomy (see page 13). Use this exhibit as a "
        "sensitivity check, not as a competing verdict."
    )


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
COMPOSITE_CSV = METRICS_DIR / "composite_ranking.csv"


@st.cache_data(ttl=60)
def load_composite() -> pd.DataFrame:
    if not COMPOSITE_CSV.exists():
        return pd.DataFrame()
    return pd.read_csv(COMPOSITE_CSV)


df = load_composite()
if df.empty:
    st.error(
        f"Missing {COMPOSITE_CSV.name}. Generate it first:\n\n"
        "    python scripts/15_composite_ranking.py"
    )
    st.stop()

# --------------------------------------------------------------------------- #
# Weight sliders — recompute composite live
# --------------------------------------------------------------------------- #
st.subheader("Weights")
st.caption("Defaults: w_acc=0.50, w_climate=0.20, w_interp=0.15, w_compute=0.15. "
           "Weights are renormalized to sum to 1 when you drag; the effective "
           "weights are shown below.")

c1, c2, c3, c4 = st.columns(4)
with c1:
    w_acc = st.slider("w_acc (accuracy)", 0.0, 1.0, 0.50, 0.05)
with c2:
    w_climate = st.slider("w_climate (climate)", 0.0, 1.0, 0.20, 0.05)
with c3:
    w_interp = st.slider("w_interp (interpret.)", 0.0, 1.0, 0.15, 0.05)
with c4:
    w_compute = st.slider("w_compute (compute)", 0.0, 1.0, 0.15, 0.05)

total = w_acc + w_climate + w_interp + w_compute
if total <= 0:
    st.warning("All weights are zero — set at least one weight above 0.")
    st.stop()

# Renormalize so the composite stays comparable across slider positions.
w_acc_n, w_climate_n, w_interp_n, w_compute_n = (
    w_acc / total, w_climate / total, w_interp / total, w_compute / total
)

wc1, wc2, wc3, wc4 = st.columns(4)
wc1.metric("acc (renorm.)", f"{w_acc_n:.2f}")
wc2.metric("climate (renorm.)", f"{w_climate_n:.2f}")
wc3.metric("interp (renorm.)", f"{w_interp_n:.2f}")
wc4.metric("compute (renorm.)", f"{w_compute_n:.2f}")

df = df.copy()
df["composite_score"] = (
    w_acc_n * df["acc_score"]
    + w_climate_n * df["climate_score"]
    + w_interp_n * df["interp_score"]
    + w_compute_n * df["compute_score"]
)
df["rank"] = (df.groupby("crop")["composite_score"]
                .rank(ascending=False, method="min").astype(int))

divider("Per-crop composite winner")

# --------------------------------------------------------------------------- #
# Per-crop winner table with component breakdown
# --------------------------------------------------------------------------- #
winner_rows = []
for crop, g in df.groupby("crop"):
    top = g.sort_values("composite_score", ascending=False).iloc[0]
    winner_rows.append({
        "crop": crop,
        "winner": top["model"],
        "composite": round(float(top["composite_score"]), 3),
        "acc": round(float(top["acc_score"]), 3),
        "climate": round(float(top["climate_score"]), 3),
        "interp": round(float(top["interp_score"]), 3),
        "compute": round(float(top["compute_score"]), 3),
    })
winners = pd.DataFrame(winner_rows).sort_values("crop").reset_index(drop=True)
st.dataframe(winners, use_container_width=True, hide_index=True)

divider("Composite heatmap (crop × model)")

# --------------------------------------------------------------------------- #
# Heatmap
# --------------------------------------------------------------------------- #
pivot = df.pivot(index="crop", columns="model", values="composite_score")
# Order columns by mean composite so the fastest-plus-accurate models sit left.
col_order = pivot.mean(axis=0).sort_values(ascending=False).index.tolist()
pivot = pivot.loc[:, col_order]

fig = px.imshow(
    pivot.values,
    x=pivot.columns, y=pivot.index,
    color_continuous_scale="Viridis",
    zmin=0.0, zmax=1.0,
    aspect="auto",
    text_auto=".2f",
    labels={"x": "model", "y": "crop", "color": "composite"},
)
fig.update_layout(height=520, margin=dict(l=10, r=10, t=30, b=10))
st.plotly_chart(fig, use_container_width=True)

st.caption("Cell = weighted composite in [0, 1] for that (crop, model). "
           "Bright = high composite under the current weights. The bright "
           "cell per row is the winner shown in the table above.")

# --------------------------------------------------------------------------- #
# Full per-(crop, model) breakdown for inspection
# --------------------------------------------------------------------------- #
divider("Full per-(crop, model) breakdown")
show_cols = ["crop", "model", "rank", "composite_score",
             "acc_score", "climate_score", "interp_score", "compute_score"]
display = (df[show_cols]
             .sort_values(["crop", "rank"])
             .reset_index(drop=True))
# Round for display without mutating the underlying frame.
for c in ["composite_score", "acc_score", "climate_score",
          "interp_score", "compute_score"]:
    display[c] = display[c].round(3)
st.dataframe(display, use_container_width=True, hide_index=True)

st.caption("See ``scripts/15_composite_ranking.py`` for the component-score "
           "definitions, calibration constants, and caveats. The four "
           "component scores are pre-computed and cached; only the composite "
           "and ranks are recomputed live as you drag the weight sliders.")

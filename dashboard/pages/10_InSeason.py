"""In-season horizon analysis — the headline view from the PPT.

For each crop, shows R² vs days-pre-harvest with one line per model
(Depth 1), plus an interactive Horizon × Crop × Model matrix (Depth 2).

Data source: ``results/metrics/multicrop_horizons.csv`` produced by
``scripts/05c_eval_horizons.py``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import METRICS_DIR, MODEL_INFO, divider, page_header

st.set_page_config(page_title="In-Season Horizons — Crop Yield",
                   layout="wide")
page_header("In-season horizon analysis",
            "How model accuracy degrades the further before harvest you ask "
            "for a prediction. The PPT's headline research question.")
st.caption("_How each model's R² degrades the further before harvest the prediction is made — the horizon dimension the leaderboard collapses into a single number._")

# --------------------------------------------------------------------------- #
# Load
# --------------------------------------------------------------------------- #
csv_path = METRICS_DIR / "multicrop_horizons.csv"
if not csv_path.exists():
    st.warning(
        "`results/metrics/multicrop_horizons.csv` not found. "
        "Run `python scripts/05c_eval_horizons.py` to generate it."
    )
    st.stop()

df = pd.read_csv(csv_path)
if df.empty:
    st.warning("multicrop_horizons.csv is empty.")
    st.stop()

# Keep only the in-season horizons (drop 'full' from the line plot, keep it
# in the matrix view as an explicit choice).
horizon_options = (sorted(df[df["horizon"] != "full"]["horizon_days"].dropna()
                          .astype(int).unique().tolist()))
crops = sorted(df["crop"].unique().tolist())
models = (df.groupby("model")["r2"].mean()
            .sort_values(ascending=False).index.tolist())

# --------------------------------------------------------------------------- #
# Depth 1 — per-crop horizon curves
# --------------------------------------------------------------------------- #
divider("R² vs days-pre-harvest")
left, right = st.columns([1, 3])

with left:
    sel_crop = st.selectbox(
        "Crop",
        crops,
        index=crops.index("winter_wheat") if "winter_wheat" in crops else 0,
        key="curve_crop",
    )
    show_full = st.checkbox(
        "Include 'full' (no cutoff) endpoint",
        value=False,
        help="When on, the plot also shows R² when the model gets the entire "
             "growing-season time series — no in-season truncation. This is "
             "the easiest case; useful as an upper bound.",
    )

with right:
    sub = df[df["crop"] == sel_crop].copy()
    if not show_full:
        sub = sub[sub["horizon"] != "full"]
    if sub.empty:
        st.info("No data for this crop yet.")
    else:
        # Plot horizon_days on x-axis (larger = further pre-harvest).
        # "full" uses 0 days for visual placement on the left, labelled.
        sub["x"] = sub["horizon_days"].fillna(0).astype(int)
        sub = sub.sort_values(["model", "x"], ascending=[True, False])
        fig = px.line(
            sub, x="x", y="r2", color="model", markers=True,
            labels={"x": "Days pre-harvest", "r2": "R²", "model": "Model"},
            title=f"{sel_crop} — R² across horizons",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_xaxes(autorange="reversed")  # closer to harvest on the right
        fig.add_hline(y=0, line_dash="dot", line_color="grey",
                      annotation_text="R²=0 (constant-prediction baseline)",
                      annotation_position="bottom right")
        fig.update_layout(height=460, hovermode="x unified")
        st.plotly_chart(fig, width="stretch")

# Table beneath: numbers behind the chart for the chosen crop.
piv = (sub.pivot(index="model", columns="horizon", values="r2")
          .reindex(index=models))
if not piv.empty:
    # Order columns: h-30d, h-60d, ..., then 'full'
    cols = sorted([c for c in piv.columns if c.startswith("h-")],
                  key=lambda c: int(c[2:-1]))
    if "full" in piv.columns:
        cols.append("full")
    piv = piv[cols]
    st.dataframe(
        piv.round(3).style.format("{:.3f}")
            .background_gradient(cmap="RdYlGn", vmin=-0.2, vmax=0.7),
        width="stretch",
    )

# --------------------------------------------------------------------------- #
# Depth 2 — interactive Horizon × Crop × Model matrix
# --------------------------------------------------------------------------- #
divider("Crop × Model matrix at chosen horizon")
fc1, fc2 = st.columns([1, 1])
with fc1:
    sel_horizon = st.selectbox(
        "Horizon",
        ["h-30d", "h-60d", "h-90d", "h-120d", "full"],
        index=0,
        key="matrix_horizon",
    )
with fc2:
    metric = st.selectbox("Metric", ["r2", "rmse", "mae"], index=0,
                          key="matrix_metric")

mh = df[df["horizon"] == sel_horizon]
if mh.empty:
    st.info("No data for this horizon yet (likely still computing).")
else:
    mat = (mh.pivot(index="crop", columns="model", values=metric)
              .reindex(columns=models))
    # Sort crops by mean across models
    mat["__mean__"] = mat.mean(axis=1)
    mat = mat.sort_values("__mean__",
                          ascending=(metric != "r2"))
    mat = mat.drop(columns="__mean__")
    fmt = "{:.3f}" if metric == "r2" else "{:.2f}"
    if metric == "r2":
        styled = (mat.style.format(fmt)
                     .background_gradient(cmap="RdYlGn", vmin=-0.2, vmax=0.7))
    else:
        # lower is better for rmse/mae → reverse the colormap
        styled = (mat.style.format(fmt)
                     .background_gradient(cmap="RdYlGn_r"))
    st.dataframe(styled, width="stretch")

    # Best model per crop at this horizon
    st.markdown("##### Best model per crop at this horizon")
    best = (mh.sort_values(metric, ascending=(metric != "r2"))
              .groupby("crop", as_index=False).first()
              [["crop", "model", "r2", "rmse", "mae", "n"]])
    st.dataframe(
        best.style.format({"r2": "{:.3f}", "rmse": "{:.2f}",
                           "mae": "{:.2f}"})
            .background_gradient(subset=["r2"], cmap="RdYlGn"),
        width="stretch", hide_index=True,
    )

st.markdown("---")
st.caption(
    "Data file: `results/metrics/multicrop_horizons.csv` "
    "(regenerate with `python scripts/05c_eval_horizons.py`)."
)

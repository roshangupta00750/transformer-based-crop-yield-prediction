"""Predictions: how close each model gets, crop by crop, and which to use.

Three views: all six models against the historical average for one crop,
predicted vs actual for one model, and the best model per crop with weight
sliders (the former Recommendation page). Accuracy is error in dt/ha from the
official results; the sliders reuse the fixed interpretability and compute
scores per model.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import (ALL_MODELS, CROP_LABEL, CROPS, MODEL_COLOR,  # noqa: E402
                     MODEL_LABEL, MUTED, error_vs_history, load_polygons,
                     load_test_predictions, reproducible_pairs)
from utils import (HELP, METRICS_DIR, load_climatology_baseline,  # noqa: E402
                   load_multicrop_summary, page_header)

st.set_page_config(page_title="Predictions — Crop Yield Germany", layout="wide")
page_header("Predictions",
            "How close each model gets to the actual yield, crop by crop.")

mc = load_multicrop_summary()
ev = error_vs_history(mc, load_climatology_baseline())
preds = load_test_predictions()
ok_pairs = reproducible_pairs(preds, mc)
_, names = load_polygons()

if ev.empty:
    st.info("Model results not found.")
    st.stop()

crop = st.selectbox("Crop", CROPS, format_func=CROP_LABEL.get, key="pr_crop")
left, right = st.columns(2, gap="large")

# --------------------------------------------------------------------------- #
# All models for this crop vs the historical average
# --------------------------------------------------------------------------- #
with left:
    st.markdown(f"##### All models — {CROP_LABEL[crop]}",
                help=HELP["error"] + "\n\n" + HELP["historical"])
    d = ev[ev["crop"] == crop].sort_values("rmse")
    hist = float(d["hist_rmse"].iloc[0])
    fig = go.Figure(go.Bar(
        x=d["rmse"], y=[MODEL_LABEL[m] for m in d["model"]], orientation="h",
        marker=dict(color=[MODEL_COLOR[m] for m in d["model"]],
                    line=dict(width=2, color="#fcfcfb")),
        customdata=d["pct_vs_hist"],
        hovertemplate="%{y}: %{x:.1f} dt/ha<br>%{customdata:+.1f}% vs "
                      "historical average<extra></extra>",
    ))
    fig.add_vline(x=hist, line_dash="dot", line_width=2, line_color=MUTED)
    fig.add_annotation(x=hist, y=1.04, yref="paper", showarrow=False,
                       text=f"historical average {hist:.1f}",
                       font=dict(size=11, color=MUTED))
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10),
                      xaxis=dict(title="error (dt/ha)", gridcolor="#e1e0d9"),
                      yaxis=dict(autorange="reversed"), showlegend=False,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.caption("Error on 2024–25, lower is better. Bars left of the dotted "
               "line beat the historical average.")

# --------------------------------------------------------------------------- #
# Predicted vs actual for one model
# --------------------------------------------------------------------------- #
with right:
    st.markdown("##### Predicted vs actual", help=HELP["pred_vs_actual"])
    models = [m for m in ALL_MODELS if (m, crop) in ok_pairs]
    missing = [MODEL_LABEL[m] for m in ALL_MODELS if (m, crop) not in ok_pairs]
    if not models:
        st.info("No per-district predictions for this crop yet. Run "
                "`scripts/23_export_test_predictions.py`.")
    else:
        model = st.selectbox("Model", models, format_func=MODEL_LABEL.get,
                             key="pr_model", help=HELP["models_listed"])
        p = preds[(preds["model"] == model) & (preds["crop"] == crop)]
        p = p.merge(names, on="landkreis", how="left")
        p["name"] = p["name"].fillna(p["landkreis"])
        lo = float(min(p["y_true"].min(), p["y_pred"].min()))
        hi = float(max(p["y_true"].max(), p["y_pred"].max()))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                                 line=dict(color=MUTED, width=2, dash="dot"),
                                 hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(
            x=p["y_true"], y=p["y_pred"], mode="markers", showlegend=False,
            marker=dict(size=8, color=MODEL_COLOR[model], opacity=0.75,
                        line=dict(width=1, color="#fcfcfb")),
            customdata=np.stack([p["name"], p["year"]], axis=-1),
            hovertemplate="%{customdata[0]}, %{customdata[1]}<br>actual "
                          "%{x:.1f} · predicted %{y:.1f} dt/ha<extra></extra>",
        ))
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10),
                          xaxis=dict(title="actual (dt/ha)", gridcolor="#e1e0d9"),
                          yaxis=dict(title="predicted (dt/ha)", gridcolor="#e1e0d9"),
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.caption("Each dot is one district and year. On the dotted line = "
                   "exact prediction.")
    if missing:
        st.caption("Not shown per district yet: " + ", ".join(missing) +
                   " (predictions on this machine don't match the official results).")

# --------------------------------------------------------------------------- #
# Best model per crop, with weights
# --------------------------------------------------------------------------- #
st.markdown("---")
st.markdown("##### Best model per crop", help=HELP["best_model"])
s1, s2, s3 = st.columns(3)
w_acc = s1.slider("Accuracy", 0.0, 1.0, 1.0, 0.1, help=HELP["accuracy_w"])
w_int = s2.slider("Interpretability", 0.0, 1.0, 0.0, 0.1, help=HELP["interp_w"])
w_cmp = s3.slider("Low compute cost", 0.0, 1.0, 0.0, 0.1, help=HELP["compute_w"])

props = pd.DataFrame(columns=["interp_score", "compute_score"])
comp_path = METRICS_DIR / "composite_ranking.csv"
if comp_path.exists():
    props = (pd.read_csv(comp_path).groupby("model")[["interp_score",
                                                      "compute_score"]].first())

scored = ev.copy()
rng = scored.groupby("crop")["rmse"].agg(["min", "max"])
scored = scored.join(rng, on="crop")
span = (scored["max"] - scored["min"]).replace(0, np.nan)
scored["acc"] = (1 - (scored["rmse"] - scored["min"]) / span).fillna(1.0)
scored = scored.join(props, on="model")
w_sum = max(w_acc + w_int + w_cmp, 1e-9)
scored["score"] = (w_acc * scored["acc"]
                   + w_int * scored["interp_score"].fillna(0)
                   + w_cmp * scored["compute_score"].fillna(0)) / w_sum

best = scored.loc[scored.groupby("crop")["score"].idxmax()].set_index("crop")
best = best.reindex([c for c in CROPS if c in best.index])
table = pd.DataFrame({
    "Crop": [CROP_LABEL[c] for c in best.index],
    "Best model": [MODEL_LABEL[m] for m in best["model"]],
    "Error (dt/ha)": best["rmse"].round(1).to_numpy(),
    "vs historical average": [f"{v:+.1f}%" for v in best["pct_vs_hist"]],
    "Beats historical average": np.where(best["pct_vs_hist"] < 0, "yes", "no"),
})
st.dataframe(table, width="stretch", hide_index=True, column_config={
    "Error (dt/ha)": st.column_config.NumberColumn(help=HELP["error"], format="%.1f"),
    "vs historical average": st.column_config.TextColumn(help=HELP["pct"]),
    "Beats historical average": st.column_config.TextColumn(help=HELP["historical"]),
})
st.caption("Move the sliders to trade accuracy against interpretability or "
           "compute cost. Climate robustness returns once the climate "
           "scenarios are re-run on the current models.")

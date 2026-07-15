"""Interpretability comparison — TFT variable-selection vs. XGBoost gain.

The PPT's research question is the "accuracy–interpretability trade-off";
this page surfaces the interpretability half. TFT's variable-selection
network is one of its signature features, so we render it directly.
XGBoost's gain-based importance is shown side by side as a sanity anchor
("does the simple model attend to the same things?").

PatchTST and iTransformer attention extraction is on the roadmap — they
require modifying their forward passes to return the attention tensors,
which is more invasive and best done as a follow-up.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _interpret import (itransformer_attention, patchtst_attention,
                        tft_interpretation, xgboost_importance)
from utils import divider, load_unified, page_header

st.set_page_config(page_title="Interpretability — Crop Yield",
                   layout="wide")
page_header("Interpretability",
            "Which features did the models actually use? TFT's variable-"
            "selection network vs. XGBoost's gain-based importance.")
st.caption("_What each model actually attended to — TFT variable selection, PatchTST patch attention, iTransformer variate attention, and XGBoost gain — the interpretability half of the accuracy-vs-interpretability trade-off._")

crops = sorted(load_unified()["crop"].unique().tolist()) \
    if not load_unified().empty else ["winter_wheat"]
crop = st.selectbox(
    "Crop", crops,
    index=crops.index("winter_wheat") if "winter_wheat" in crops else 0,
)

# --------------------------------------------------------------------------- #
# TFT — variable-selection + attention
# --------------------------------------------------------------------------- #
divider("TFT — Variable-Selection Networks")
st.caption(
    "TFT's variable-selection networks output a soft-attention score per "
    "feature, indicating how much each input drove the prediction. Scores "
    "averaged across the test set's prediction points."
)

with st.spinner("Loading TFT and extracting interpretation…"):
    tft = tft_interpretation(crop)

if tft is None:
    st.warning("No TFT interpretation available. Train TFT (`scripts/03_train.py "
               "--model tft`) and ensure `data/processed/dataset.parquet` exists.")
else:
    st.caption(f"Averaged over {tft['n_samples']} prediction points for crop "
               f"**{crop}**.")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### Static features")
        s = tft["static_importance"]
        if not s.empty:
            df_s = s.reset_index()
            df_s.columns = ["feature", "importance"]
            fig = px.bar(df_s, x="importance", y="feature", orientation="h",
                         color="importance", color_continuous_scale="Viridis")
            fig.update_layout(height=320, yaxis_title="", coloraxis_showscale=False,
                              yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("No static-variable scores in this checkpoint.")

    with col2:
        st.markdown("##### Past-observed features (encoder)")
        e = tft["encoder_importance"]
        if not e.empty:
            df_e = e.reset_index()
            df_e.columns = ["feature", "importance"]
            fig = px.bar(df_e, x="importance", y="feature", orientation="h",
                         color="importance", color_continuous_scale="Plasma")
            fig.update_layout(height=320, yaxis_title="", coloraxis_showscale=False,
                              yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("No encoder-variable scores in this checkpoint.")

    # Decoder + attention
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("##### Known-future features (decoder)")
        d = tft["decoder_importance"]
        if not d.empty:
            df_d = d.reset_index()
            df_d.columns = ["feature", "importance"]
            fig = px.bar(df_d, x="importance", y="feature", orientation="h",
                         color="importance", color_continuous_scale="Inferno")
            fig.update_layout(height=260, yaxis_title="", coloraxis_showscale=False,
                              yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("No decoder-variable scores available.")

    with col4:
        st.markdown("##### Encoder attention over time")
        a = tft["encoder_attention_mean"]
        if not a.empty:
            df_a = a.reset_index()
            df_a.columns = ["timestep", "weight"]
            fig = px.bar(df_a, x="timestep", y="weight",
                         color="weight", color_continuous_scale="Cividis")
            fig.update_layout(height=260, xaxis_title="Past time-step "
                              "(t-N = N days before prediction)",
                              coloraxis_showscale=False)
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("No attention weights available.")

# --------------------------------------------------------------------------- #
# PatchTST — patch-level attention (which time-chunks the model attended to)
# --------------------------------------------------------------------------- #
divider("PatchTST — Patch-level attention")
st.caption(
    "PatchTST tokenises each variable's time series into overlapping patches "
    "and runs a transformer encoder over them. The bar chart below shows the "
    "**mean attention weight received by each patch** — high values mean the "
    "model weighted that time-chunk heavily when forming its prediction. "
    "Patch_1 is the earliest window of the season, the rightmost patch is "
    "closest to harvest."
)
with st.spinner("Loading PatchTST and running attention forward pass…"):
    p = patchtst_attention(crop)
if p is None:
    st.warning("No PatchTST attention available — model checkpoint missing.")
else:
    st.caption(f"Averaged across {p['n_samples']} test samples and all encoder layers.")
    df_p = p["attention"].reset_index()
    df_p.columns = ["patch", "attention"]
    df_p["order"] = df_p["patch"].str.extract(r"(\d+)").astype(int)
    df_p = df_p.sort_values("order")
    fig = px.bar(df_p, x="patch", y="attention",
                 color="attention", color_continuous_scale="Cividis")
    fig.update_layout(height=340, coloraxis_showscale=False,
                      xaxis_title="Patch (early → late season)")
    st.plotly_chart(fig, width="stretch")

# --------------------------------------------------------------------------- #
# iTransformer — variate-as-token attention (which variables matter together)
# --------------------------------------------------------------------------- #
divider("iTransformer — Variable-level attention")
st.caption(
    "iTransformer inverts the usual transformer: each *variable* (not each "
    "timestep) becomes a token. The attention map shows which variables the "
    "model co-attended to. High weight on a variable means the model used it "
    "heavily when building its representation."
)
with st.spinner("Loading iTransformer and running attention forward pass…"):
    it = itransformer_attention(crop)
if it is None:
    st.warning("No iTransformer attention available — model checkpoint missing.")
else:
    st.caption(f"Averaged across {it['n_samples']} test samples and all encoder layers.")
    df_it = it["attention"].reset_index()
    df_it.columns = ["variable", "attention"]
    fig = px.bar(df_it, x="attention", y="variable", orientation="h",
                 color="attention", color_continuous_scale="Plasma")
    fig.update_layout(height=360, coloraxis_showscale=False,
                      yaxis_title="", yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, width="stretch")

# --------------------------------------------------------------------------- #
# XGBoost — gain-based importance
# --------------------------------------------------------------------------- #
divider("XGBoost — Gain-based feature importance")
st.caption(
    "XGBoost trains on a 12-month aggregation of each past feature "
    "(mean / std / min / max per month). We collapse the booster's leaf-level "
    "gain back to two views: total importance per raw input, and which time-"
    "of-season chunks the boosted trees actually split on."
)

xgb_df = xgboost_importance()
if xgb_df is None or xgb_df.empty:
    st.warning("No XGBoost importance available. Train it first "
               "(`python scripts/03_train.py --model xgboost`).")
else:
    by_feature = xgb_df.groupby("feature")["gain"].sum().sort_values(ascending=False)
    by_bucket = (xgb_df[xgb_df["bucket"].str.startswith("m")]
                 .groupby("bucket")["gain"].sum().sort_index())

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("##### Total gain per raw feature")
        df_f = by_feature.reset_index()
        df_f.columns = ["feature", "gain"]
        fig = px.bar(df_f.head(20), x="gain", y="feature", orientation="h",
                     color="gain", color_continuous_scale="Viridis")
        fig.update_layout(height=420, yaxis_title="", coloraxis_showscale=False,
                          yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, width="stretch")

    with col_b:
        st.markdown("##### Gain by month-of-season bucket")
        df_b = by_bucket.reset_index()
        df_b.columns = ["bucket", "gain"]
        fig = px.bar(df_b, x="bucket", y="gain",
                     color="gain", color_continuous_scale="Magma")
        fig.update_layout(height=420, xaxis_title="Month bucket (m01 = first 1/12 of season)",
                          coloraxis_showscale=False)
        st.plotly_chart(fig, width="stretch")

    with st.expander("Raw (feature × bucket × stat) gain table"):
        st.dataframe(xgb_df.sort_values("gain", ascending=False),
                     width="stretch", hide_index=True)

# --------------------------------------------------------------------------- #
# Side-by-side comparison helper
# --------------------------------------------------------------------------- #
divider("Side-by-side: top features per model")
st.caption("All four models in one place — do they agree on what matters? "
           "Rankings normalised within each model to make them comparable.")
c_t, c_p, c_i, c_x = st.columns(4)

with c_t:
    st.markdown("**TFT — past-observed**")
    if tft and not tft["encoder_importance"].empty:
        s = tft["encoder_importance"].head(10)
        s_norm = (s / s.sum()).round(3)
        st.dataframe(s_norm, width="stretch")
    else:
        st.info("TFT data not available.")

with c_p:
    st.markdown("**PatchTST — peak patches**")
    if p is not None and not p["attention"].empty:
        s = p["attention"].sort_values(ascending=False).head(10)
        st.dataframe(s.round(4), width="stretch")
    else:
        st.info("PatchTST not available.")

with c_i:
    st.markdown("**iTransformer — variables**")
    if it is not None and not it["attention"].empty:
        s = it["attention"].head(10)
        st.dataframe(s.round(3), width="stretch")
    else:
        st.info("iTransformer not available.")

with c_x:
    st.markdown("**XGBoost — gain**")
    if xgb_df is not None and not xgb_df.empty:
        s = xgb_df.groupby("feature")["gain"].sum().sort_values(ascending=False).head(10)
        st.dataframe(s.round(1), width="stretch")
    else:
        st.info("XGBoost not available.")

st.markdown("---")
st.caption(
    "LSTM has no native attention mechanism; if you need per-feature "
    "importance for it, use permutation importance on the test set. DLinear "
    "is by construction a linear projection — its weights are directly "
    "inspectable from the saved checkpoint."
)

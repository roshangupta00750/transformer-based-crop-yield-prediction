"""Interpretability — which inputs did the models actually use?

The PPT's research question is the "accuracy-interpretability trade-off";
this page surfaces the interpretability half. One question, one chart: pick a
crop and a model, see what that model attended to.

Everything the previous version showed simultaneously is still here — TFT's
static/decoder/attention panels, PatchTST's patch curve, iTransformer's
variate attention, XGBoost's month-bucket split and its raw gain table, and
the four-way side-by-side — but behind three collapsed expanders instead of
thirteen charts and tables competing for the same screen.

The extractors in ``dashboard/_interpret.py`` run forward passes over the
test split, so they are wrapped in ``st.cache_data`` here; without that every
widget interaction re-ran four model loads.

LSTM has no native attention mechanism and DLinear is a linear projection
whose weights are directly inspectable from the checkpoint — neither is
extracted here (see Method notes).
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
from utils import kpi, load_unified, page_header

st.set_page_config(page_title="Interpretability — Crop Yield",
                   layout="wide")
page_header("Interpretability",
            "Which inputs did the models actually use?")


# --------------------------------------------------------------------------- #
# Cached extractors — each of these loads a checkpoint and runs a forward pass
# over the test split, which is far too expensive to repeat on every rerun.
# --------------------------------------------------------------------------- #
@st.cache_data(ttl=1800, show_spinner="Extracting TFT variable selection…")
def _tft(crop: str):
    return tft_interpretation(crop)


@st.cache_data(ttl=1800, show_spinner="Running PatchTST attention pass…")
def _patchtst(crop: str):
    return patchtst_attention(crop)


@st.cache_data(ttl=1800, show_spinner="Running iTransformer attention pass…")
def _itransformer(crop: str):
    return itransformer_attention(crop)


@st.cache_data(ttl=1800, show_spinner="Reading XGBoost booster gain…")
def _xgboost():
    return xgboost_importance()


def _bar(series: pd.Series, value_name: str, *, horizontal: bool,
         scale: str, height: int, axis_title: str = "") -> None:
    """One bar chart from a name -> score Series."""
    df = series.reset_index()
    df.columns = ["name", value_name]
    if horizontal:
        fig = px.bar(df, x=value_name, y="name", orientation="h",
                     color=value_name, color_continuous_scale=scale)
        fig.update_layout(yaxis_title=axis_title,
                          yaxis={"categoryorder": "total ascending"})
    else:
        fig = px.bar(df, x="name", y=value_name,
                     color=value_name, color_continuous_scale=scale)
        fig.update_layout(xaxis_title=axis_title)
    fig.update_layout(height=height, coloraxis_showscale=False,
                      margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, width="stretch")


# --------------------------------------------------------------------------- #
# Controls — both drive the primary chart
# --------------------------------------------------------------------------- #
_unified = load_unified()
crops = sorted(_unified["crop"].unique().tolist()) if not _unified.empty \
    else ["winter_wheat"]

MODEL_LABELS = {
    "tft": "TFT — variable selection",
    "patchtst": "PatchTST — patch attention",
    "itransformer": "iTransformer — variate attention",
    "xgboost": "XGBoost — gain",
}

c_crop, c_model = st.columns([1, 3])
with c_crop:
    crop = st.selectbox(
        "Crop", crops,
        index=crops.index("winter_wheat") if "winter_wheat" in crops else 0,
    )
with c_model:
    model = st.radio("Model", list(MODEL_LABELS),
                     format_func=lambda m: MODEL_LABELS[m],
                     horizontal=True, key="interp_model")

# Extract once; the expanders below reuse these objects.
tft = _tft(crop)
patch = _patchtst(crop)
itr = _itransformer(crop)
xgb_df = _xgboost()

# --------------------------------------------------------------------------- #
# Resolve the selected model to one headline series
# --------------------------------------------------------------------------- #
series: pd.Series | None = None
unit = "importance"
horizontal = True
scale = "Viridis"
axis_title = ""
averaged_over = "—"
missing = ""

if model == "tft":
    if tft is None:
        missing = ("No TFT interpretation available. Train it "
                   "(`python scripts/03_train.py --model tft`) and ensure "
                   "`data/processed/dataset.parquet` exists.")
    elif tft["encoder_importance"].empty:
        missing = "This TFT checkpoint carries no encoder-variable scores."
    else:
        series = tft["encoder_importance"]
        unit = "selection weight"
        scale = "Plasma"
        averaged_over = f"{tft['n_samples']:,} prediction points"

elif model == "patchtst":
    if patch is None:
        missing = "No PatchTST attention available — checkpoint missing."
    elif patch["attention"].empty:
        missing = "PatchTST returned no attention weights."
    else:
        s = patch["attention"]
        order = s.index.to_series().str.extract(r"(\d+)")[0].astype(int)
        series = s.reindex(order.sort_values().index)
        unit = "attention"
        horizontal = False
        scale = "Cividis"
        axis_title = "Patch (early → late season)"
        averaged_over = f"{patch['n_samples']:,} test samples"

elif model == "itransformer":
    if itr is None:
        missing = "No iTransformer attention available — checkpoint missing."
    elif itr["attention"].empty:
        missing = "iTransformer returned no attention weights."
    else:
        series = itr["attention"]
        unit = "attention"
        scale = "Plasma"
        averaged_over = f"{itr['n_samples']:,} test samples"

else:  # xgboost
    if xgb_df is None or xgb_df.empty:
        missing = ("No XGBoost importance available. Train it "
                   "(`python scripts/03_train.py --model xgboost`).")
    else:
        series = (xgb_df.groupby("feature")["gain"].sum()
                  .sort_values(ascending=False).head(20))
        unit = "total gain"
        averaged_over = "booster leaf gain (all crops pooled)"

# --------------------------------------------------------------------------- #
# PRIMARY 1 — what this chart is, in four numbers
# --------------------------------------------------------------------------- #
if series is None:
    st.warning(missing)
else:
    total = float(series.sum())
    top_name = str(series.idxmax())
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        kpi("Inputs scored", f"{len(series)}")
    with k2:
        kpi("Top-ranked input", top_name)
    with k3:
        kpi("Its share of the total",
            f"{float(series.max()) / total:.1%}" if total else "—")
    with k4:
        kpi("Averaged over", averaged_over)

    # --------------------------------------------------------------------- #
    # PRIMARY 2 — the one chart
    # --------------------------------------------------------------------- #
    st.markdown(f"##### {MODEL_LABELS[model]} — {crop}")
    _bar(series, unit, horizontal=horizontal, scale=scale,
         height=460 if horizontal else 380, axis_title=axis_title)

# --------------------------------------------------------------------------- #
# SECONDARY 1 — the rest of the selected model's panels
# --------------------------------------------------------------------------- #
with st.expander(f"Detail — other views of {model}", expanded=False):
    if model == "tft":
        if tft is None:
            st.info("No TFT interpretation loaded.")
        else:
            d1, d2 = st.columns(2)
            with d1:
                st.markdown("**Static features**")
                if tft["static_importance"].empty:
                    st.info("No static-variable scores in this checkpoint.")
                else:
                    _bar(tft["static_importance"], "importance",
                         horizontal=True, scale="Viridis", height=300)
            with d2:
                st.markdown("**Known-future features (decoder)**")
                if tft["decoder_importance"].empty:
                    st.info("No decoder-variable scores available.")
                else:
                    _bar(tft["decoder_importance"], "importance",
                         horizontal=True, scale="Inferno", height=300)
            st.markdown("**Encoder attention over time**")
            if tft["encoder_attention_mean"].empty:
                st.info("No attention weights available.")
            else:
                _bar(tft["encoder_attention_mean"], "weight",
                     horizontal=False, scale="Cividis", height=300,
                     axis_title="Past time-step "
                                "(t-N = N days before prediction)")

    elif model == "xgboost":
        if xgb_df is None or xgb_df.empty:
            st.info("No XGBoost importance loaded.")
        else:
            st.markdown("**Gain by month-of-season bucket**")
            by_bucket = (xgb_df[xgb_df["bucket"].str.startswith("m")]
                         .groupby("bucket")["gain"].sum().sort_index())
            if by_bucket.empty:
                st.info("No month-bucket splits in this booster.")
            else:
                _bar(by_bucket, "gain", horizontal=False, scale="Magma",
                     height=340,
                     axis_title="Month bucket (m01 = first 1/12 of season)")
            st.markdown("**Raw (feature × bucket × stat) gain**")
            st.dataframe(xgb_df.sort_values("gain", ascending=False),
                         width="stretch", hide_index=True)

    elif model == "patchtst":
        st.info(
            "PatchTST exposes a single attention axis — the patch axis shown "
            "above. Patch_1 is the earliest window of the season, the "
            "rightmost patch is closest to harvest; there is no second view "
            "to unfold."
        )
        if patch is not None and not patch["attention"].empty:
            st.dataframe(patch["attention"].sort_values(ascending=False)
                         .round(4).rename("attention"), width="stretch")

    else:  # itransformer
        st.info(
            "iTransformer tokenises variables rather than timesteps, so the "
            "variate attention shown above is its only interpretation axis."
        )
        if itr is not None and not itr["attention"].empty:
            st.dataframe(itr["attention"].round(4).rename("attention"),
                         width="stretch")

# --------------------------------------------------------------------------- #
# SECONDARY 2 — do the four models agree?
# --------------------------------------------------------------------------- #
with st.expander("Cross-model comparison — top inputs per model",
                 expanded=False):
    st.caption(
        "Rankings normalised within each model. The axes are genuinely "
        "different — TFT and iTransformer score features, PatchTST scores "
        "time-chunks, XGBoost scores booster gain over aggregated features — "
        "so read this as agreement on *what kind of signal matters*, not as "
        "one shared scale."
    )
    c_t, c_p, c_i, c_x = st.columns(4)

    with c_t:
        st.markdown("**TFT — past-observed**")
        if tft and not tft["encoder_importance"].empty:
            s = tft["encoder_importance"].head(10)
            st.dataframe((s / s.sum()).round(3), width="stretch")
        else:
            st.info("TFT data not available.")

    with c_p:
        st.markdown("**PatchTST — peak patches**")
        if patch is not None and not patch["attention"].empty:
            st.dataframe(patch["attention"].sort_values(ascending=False)
                         .head(10).round(4), width="stretch")
        else:
            st.info("PatchTST not available.")

    with c_i:
        st.markdown("**iTransformer — variables**")
        if itr is not None and not itr["attention"].empty:
            st.dataframe(itr["attention"].head(10).round(3), width="stretch")
        else:
            st.info("iTransformer not available.")

    with c_x:
        st.markdown("**XGBoost — gain**")
        if xgb_df is not None and not xgb_df.empty:
            st.dataframe(xgb_df.groupby("feature")["gain"].sum()
                         .sort_values(ascending=False).head(10).round(1),
                         width="stretch")
        else:
            st.info("XGBoost not available.")

# --------------------------------------------------------------------------- #
# SECONDARY 3 — method notes
# --------------------------------------------------------------------------- #
with st.expander("Method notes — how each score is produced", expanded=False):
    st.markdown(
        "**TFT.** The variable-selection networks output a soft-attention "
        "score per feature, indicating how much each input drove the "
        "prediction. Scores are averaged across the test split's prediction "
        "points.\n\n"
        "**PatchTST.** Tokenises each variable's time series into overlapping "
        "patches and runs a transformer encoder over them. The chart shows "
        "the mean attention weight received by each patch, averaged across "
        "test samples and all encoder layers — high values mean the model "
        "weighted that time-chunk heavily.\n\n"
        "**iTransformer.** Inverts the usual transformer: each *variable*, "
        "not each timestep, becomes a token. High weight on a variable means "
        "the model used it heavily when building its representation.\n\n"
        "**XGBoost.** Trains on a 12-month aggregation of each past feature "
        "(mean / std / min / max per month). The booster's leaf-level gain is "
        "collapsed back to two views: total importance per raw input, and "
        "which time-of-season chunks the boosted trees actually split on.\n\n"
        "**LSTM and DLinear are absent by construction.** LSTM has no native "
        "attention mechanism — use permutation importance on the test set if "
        "you need per-feature scores. DLinear is a linear projection, so its "
        "weights are directly inspectable from the saved checkpoint."
    )
    st.caption("Extractors: `dashboard/_interpret.py`. Checkpoints: "
               "`results/checkpoints/`.")

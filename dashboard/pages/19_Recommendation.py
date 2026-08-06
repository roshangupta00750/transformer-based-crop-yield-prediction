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

Only the composite and the ranks are live. The four component scores are
frozen inside composite_ranking.csv at the moment that CSV was written,
so the page checks that file's mtime against the newest checkpoint and
against its own inputs (multicrop_summary.csv, counterfactual_daily_
summary.csv) and refuses to present a stale ranking as current.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import CHECKPOINTS_DIR, METRICS_DIR, divider, page_header

st.set_page_config(page_title="Recommendation — Crop Yield Benchmark",
                   layout="wide")
page_header("Supplementary — multi-criteria model recommendation",
            "Composite = w_acc * accuracy + w_climate * climate stability "
            "+ w_interp * interpretability + w_compute * compute cost. "
            "Drag the sliders to see how the winner shifts.")

# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
COMPOSITE_CSV = METRICS_DIR / "composite_ranking.csv"
MULTICROP_CSV = METRICS_DIR / "multicrop_summary.csv"
COUNTERFACTUAL_CSV = METRICS_DIR / "counterfactual_daily_summary.csv"


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
# Staleness — the component scores are baked into the CSV, not read live
# --------------------------------------------------------------------------- #
_comp_mt = COMPOSITE_CSV.stat().st_mtime
_ckpts = [*CHECKPOINTS_DIR.glob("*.pt"), *CHECKPOINTS_DIR.glob("*.ubj")]
_newer = [(p.name, p.stat().st_mtime)
          for p in (MULTICROP_CSV, COUNTERFACTUAL_CSV)
          if p.exists() and p.stat().st_mtime > _comp_mt]
if _ckpts:
    _newest_ckpt = max(_ckpts, key=lambda p: p.stat().st_mtime)
    if _newest_ckpt.stat().st_mtime > _comp_mt:
        _newer.append((_newest_ckpt.name, _newest_ckpt.stat().st_mtime))

if _newer:
    _lines = "\n".join(
        f"- `{n}` — {datetime.fromtimestamp(mt):%Y-%m-%d %H:%M}, "
        f"{(mt - _comp_mt) / 86400:.1f} days newer"
        for n, mt in sorted(_newer, key=lambda t: -t[1]))
    st.error(
        "**Stale ranking.** `composite_ranking.csv` was written "
        f"{datetime.fromtimestamp(_comp_mt):%Y-%m-%d %H:%M}. Its four "
        "component scores are frozen at that moment — only the weighted "
        "composite and the ranks below are recomputed live. These inputs "
        "have changed since:\n\n" + _lines +
        "\n\nRegenerate with `python scripts/15_composite_ranking.py`."
    )

# Quantify the drift rather than only asserting it. The accuracy component is
# a documented per-crop min-max of clipped R² (scripts/15_composite_ranking.py
# ::accuracy_score); recomputing it from the current multicrop_summary.csv
# shows how far the shipped ranking has moved. Diagnostic only — the shipped
# scores below are left untouched.
_acc_live = pd.DataFrame()
if MULTICROP_CSV.exists() and MULTICROP_CSV.stat().st_mtime > _comp_mt:
    _mc = pd.read_csv(MULTICROP_CSV)[["crop", "model", "r2"]].copy()
    _mc["r2_clip"] = _mc["r2"].clip(lower=0.0)
    _lo = _mc.groupby("crop")["r2_clip"].transform("min")
    _hi = _mc.groupby("crop")["r2_clip"].transform("max")
    _span = _hi - _lo
    _mc["acc_live"] = np.where(_span > 0, (_mc["r2_clip"] - _lo) / _span, 1.0)
    _acc_live = _mc[["crop", "model", "acc_live"]]

# --------------------------------------------------------------------------- #
# Weight sliders — recompute composite live
# --------------------------------------------------------------------------- #
st.subheader("Weights")
st.caption("Renormalized to sum to 1.")

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
st.dataframe(winners, width="stretch", hide_index=True)

# Drift diagnostic: how many of those winners would change if the accuracy
# component were rebuilt from the current multicrop_summary.csv?
if not _acc_live.empty:
    _chk = df.merge(_acc_live, on=["crop", "model"], how="inner")
    if not _chk.empty:
        _chk["live_composite"] = (
            w_acc_n * _chk["acc_live"]
            + w_climate_n * _chk["climate_score"]
            + w_interp_n * _chk["interp_score"]
            + w_compute_n * _chk["compute_score"]
        )
        _shipped_w = _chk.loc[_chk.groupby("crop")["composite_score"].idxmax()]
        _live_w = _chk.loc[_chk.groupby("crop")["live_composite"].idxmax()]
        _cmp = (_shipped_w[["crop", "model"]]
                .rename(columns={"model": "shipped"})
                .merge(_live_w[["crop", "model"]]
                       .rename(columns={"model": "live"}), on="crop"))
        _changed = _cmp[_cmp["shipped"] != _cmp["live"]]
        _maxdiff = float((_chk["acc_live"] - _chk["acc_score"]).abs().max())
        st.warning(
            f"**Drift check.** Rebuilding only the accuracy component from "
            f"the current `multicrop_summary.csv` moves it by up to "
            f"{_maxdiff:.2f} (component range is 0-1) and changes the winner "
            f"on {len(_changed)}/{_cmp['crop'].nunique()} crops at the "
            "current weights"
            + (": " + ", ".join(f"{r.crop} {r.shipped}→{r.live}"
                                for r in _changed.itertuples())
               if len(_changed) else "")
            + ". Diagnostic only — the table above still shows the shipped "
              "scores."
        )

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
st.plotly_chart(fig, width="stretch")

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
st.dataframe(display, width="stretch", hide_index=True)

st.caption(
    "Component-score definitions and calibration constants are in "
    "`scripts/15_composite_ranking.py`. The four component scores come "
    f"from `{COMPOSITE_CSV.name}` as written on "
    f"{datetime.fromtimestamp(_comp_mt):%Y-%m-%d %H:%M}; only the composite "
    "and the ranks are recomputed as you drag the sliders."
)

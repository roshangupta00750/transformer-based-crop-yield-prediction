"""Climate counterfactual page — CMIP6 delta-method yield projections.

Reads ``results/metrics/counterfactual_daily_summary.csv`` and
``counterfactual_daily_district.csv`` (from ``scripts/13_counterfactual_daily.py``).
The script applies additive tasmax/tasmin deltas and multiplicative pr/rsds
deltas to the 2024-25 observed daily features, then re-predicts with the
existing daily-trained models.

The rsds perturbation is INERT: feature policy v3 excludes
``radiation_global`` (see ``configs/data.yaml`` -> ``features.excluded``),
so no model consumes that column. These counterfactuals therefore measure
the response to temperature and precipitation change only. The page
verifies this live against the config and the checkpoint feature spec
rather than asserting it.

Which GCMs contributed is read from ``data/processed/cmip6_nuts3.parquet``,
not hardcoded here.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import (CHECKPOINTS_DIR, METRICS_DIR, PROJECT_ROOT, divider, kpi,
                   page_header)


def _stale_banner(*metric_paths: Path) -> None:
    """Warn when a metric file predates the newest checkpoint on disk."""
    ckpts = [*CHECKPOINTS_DIR.glob("*.pt"), *CHECKPOINTS_DIR.glob("*.ubj")]
    if not ckpts:
        return
    newest = max(ckpts, key=lambda p: p.stat().st_mtime)
    newest_mt = newest.stat().st_mtime
    stale = [(p, p.stat().st_mtime) for p in metric_paths
             if p.exists() and p.stat().st_mtime < newest_mt]
    if not stale:
        return
    lines = "\n".join(
        f"- `{p.name}` — {datetime.fromtimestamp(mt):%Y-%m-%d %H:%M}, "
        f"{(newest_mt - mt) / 86400:.1f} days older"
        for p, mt in stale)
    st.warning(
        "**Stale — these numbers are not from the models currently on "
        f"disk.** Newest checkpoint `{newest.name}` is "
        f"{datetime.fromtimestamp(newest_mt):%Y-%m-%d %H:%M}:\n\n" + lines
    )


def _excluded_features() -> set[str]:
    """Feature names the data policy excludes from every model."""
    import yaml
    p = PROJECT_ROOT / "configs" / "data.yaml"
    if not p.exists():
        return set()
    cfg = yaml.safe_load(p.read_text()) or {}
    return set((cfg.get("features") or {}).get("excluded") or [])


def _model_input_features() -> set[str]:
    """Union of every feature name across the checkpoint pipeline sidecars."""
    names: set[str] = set()
    for p in CHECKPOINTS_DIR.glob("*.pipeline.json"):
        try:
            spec = (json.loads(p.read_text()) or {}).get("spec") or {}
        except (json.JSONDecodeError, OSError):
            continue
        for v in spec.values():
            if isinstance(v, list):
                names.update(str(x) for x in v)
            elif isinstance(v, str):
                names.add(v)
    return names


st.set_page_config(page_title="Climate counterfactual — Crop Yield Benchmark",
                   layout="wide")
page_header("Climate counterfactual (CMIP6 SSP2-4.5 / SSP5-8.5)",
            "Delta-method: apply (SSP future window - historical baseline) "
            "climate deltas to 2024-25 observed features, predict with the "
            "existing daily-trained models.")

summary_path = METRICS_DIR / "counterfactual_daily_summary.csv"
district_path = METRICS_DIR / "counterfactual_daily_district.csv"
if not summary_path.exists():
    st.warning(
        "`results/metrics/counterfactual_daily_summary.csv` not found. "
        "Run `python scripts/13_counterfactual_daily.py`."
    )
    st.stop()

_stale_banner(summary_path, district_path)

summary = pd.read_csv(summary_path)
district = pd.read_csv(district_path) if district_path.exists() else pd.DataFrame()

# --------------------------------------------------------------------------- #
# What this experiment can and cannot show
# --------------------------------------------------------------------------- #
_excluded = _excluded_features()
_model_feats = _model_input_features()
_rad_perturbed = "radiation_global"          # the column scripts/13 scales by rsds
_rad_reaches_models = (_rad_perturbed in _model_feats
                       and _rad_perturbed not in _excluded)

_limits = []
if not _rad_reaches_models:
    _limits.append(
        f"**Solar radiation is inert.** `scripts/13_counterfactual_daily.py` "
        f"scales `{_rad_perturbed}` by the CMIP6 rsds delta, but that column "
        + ("is in `configs/data.yaml` -> `features.excluded`"
           if _rad_perturbed in _excluded else
           "is absent from every checkpoint's feature spec")
        + ", so it reaches no model. These results are a response to "
          "temperature and precipitation change only."
    )
_veg = sorted({"ndvi", "evi", "lai"} & _model_feats)
if _veg:
    _limits.append(
        f"**No vegetation feedback.** Satellite indices ({', '.join(_veg)}) "
        "are model inputs but are held at their observed 2024-25 values, so "
        "the crop's own response to the perturbed climate is not simulated."
    )
if "humidity" in _model_feats:
    _limits.append("**Humidity is held constant** — no CMIP6 counterpart is "
                   "applied.")
if _limits:
    st.info("\n\n".join(_limits))

# --------------------------------------------------------------------------- #
# Headline KPIs
# --------------------------------------------------------------------------- #
divider("Applied climate signal")

deltas = summary.groupby("scenario")[
    ["tasmax_delta_C", "pr_pct_delta"]].first()
k1, k2, k3, k4 = st.columns(4)
if "ssp2_4_5" in deltas.index:
    d = deltas.loc["ssp2_4_5"]
    with k1: kpi("SSP2-4.5 tasmax Δ", f"{d['tasmax_delta_C']:+.2f} °C")
    with k2: kpi("SSP2-4.5 precip Δ", f"{d['pr_pct_delta']:+.1f} %")
if "ssp5_8_5" in deltas.index:
    d = deltas.loc["ssp5_8_5"]
    with k3: kpi("SSP5-8.5 tasmax Δ", f"{d['tasmax_delta_C']:+.2f} °C")
    with k4: kpi("SSP5-8.5 precip Δ", f"{d['pr_pct_delta']:+.1f} %")

if {"ssp2_4_5", "ssp5_8_5"} <= set(deltas.index):
    _t2 = float(deltas.loc["ssp2_4_5", "tasmax_delta_C"])
    _t5 = float(deltas.loc["ssp5_8_5", "tasmax_delta_C"])
    if _t5 <= _t2:
        st.error(
            f"**Applied signal is physically inconsistent.** SSP5-8.5 warms "
            f"less ({_t5:+.2f} °C) than SSP2-4.5 ({_t2:+.2f} °C) in this "
            "run. The high-forcing scenario must warm more, so the delta "
            "extraction in `scripts/13_counterfactual_daily.py` is suspect "
            "and every per-scenario comparison below inherits the fault."
        )

# --------------------------------------------------------------------------- #
# Per-crop mean predicted yield delta (across models)
# --------------------------------------------------------------------------- #
divider("Per-crop predicted yield delta")

crop_scenario = summary.pivot_table(
    index="crop", columns="scenario", values="delta_pct", aggfunc="mean"
).round(2)
crop_scenario = crop_scenario.sort_values(
    crop_scenario.columns[-1], ascending=False) if not crop_scenario.empty else crop_scenario

st.dataframe(
    crop_scenario.reset_index().rename(columns={"crop": "Crop"})
        .style.format({c: "{:+.2f} %" for c in crop_scenario.columns}),
    hide_index=True, width="stretch",
)

# Bar chart
long = summary.groupby(["crop", "scenario"])["delta_pct"].mean().reset_index()
fig = px.bar(
    long.sort_values("delta_pct"), x="delta_pct", y="crop",
    color="scenario", barmode="group", orientation="h",
    labels={"delta_pct": "Predicted yield delta (%)", "crop": ""},
    title=f"Predicted yield change per crop, mean across "
          f"{summary['model'].nunique()} models",
)
fig.add_vline(x=0, line_dash="dot", line_color="grey")
fig.update_layout(height=440)
st.plotly_chart(fig, width="stretch")

# --------------------------------------------------------------------------- #
# Per-model behavior — where the disagreement is
# --------------------------------------------------------------------------- #
divider("Per-model response — cross-model divergence is itself the signal")


model_pivot = summary.pivot_table(
    index="model", columns="scenario", values="delta_pct", aggfunc="mean"
).round(2)
st.dataframe(
    model_pivot.reset_index().rename(columns={"model": "Model"})
        .style.format({c: "{:+.2f} %" for c in model_pivot.columns}),
    hide_index=True, width="stretch",
)

# Heatmap: model × crop for one scenario
sel_scenario = st.selectbox(
    "Scenario for detailed view",
    sorted(summary["scenario"].unique()),
    index=(sorted(summary["scenario"].unique()).index("ssp5_8_5")
           if "ssp5_8_5" in summary["scenario"].unique() else 0),
    key="cf_scenario",
)
mc = summary[summary["scenario"] == sel_scenario].pivot(
    index="crop", columns="model", values="delta_pct")

fig = px.imshow(
    mc, color_continuous_scale="RdYlGn",
    color_continuous_midpoint=0, aspect="auto", text_auto=".2f",
    labels=dict(x="Model", y="Crop", color="Yield Δ (%)"),
    title=f"Yield Δ per (crop, model) — {sel_scenario}",
)
fig.update_layout(height=440)
st.plotly_chart(fig, width="stretch")

# Cross-model spread as OOD-uncertainty proxy
divider("Cross-model spread as OOD-uncertainty signal")

spread = summary[summary["scenario"] == sel_scenario].groupby("crop")[
    "delta_pct"].agg(["mean", "std", "min", "max", "count"])
spread.columns = ["Mean Δ%", "Std across models", "Min", "Max", "# models"]
spread = spread.sort_values("Std across models", ascending=False)

st.dataframe(
    spread.reset_index().rename(columns={"crop": "Crop"})
        .style.format({"Mean Δ%": "{:+.2f}",
                       "Std across models": "{:.2f}",
                       "Min": "{:+.2f}", "Max": "{:+.2f}"}),
    hide_index=True, width="stretch",
)

# --------------------------------------------------------------------------- #
# Per-district map (optional, requires geopandas)
# --------------------------------------------------------------------------- #
if not district.empty:
    divider("Per-district predicted yield delta")
    d_sel_model = st.selectbox(
        "Model", sorted(district["model"].unique()), key="cf_dist_model")
    d_sel_crop = st.selectbox(
        "Crop", sorted(district["crop"].unique()),
        index=sorted(district["crop"].unique()).index("winter_wheat")
        if "winter_wheat" in district["crop"].unique() else 0,
        key="cf_dist_crop")
    d_sel_scen = st.selectbox(
        "Scenario", sorted(district["scenario"].unique()),
        index=sorted(district["scenario"].unique()).index("ssp5_8_5")
        if "ssp5_8_5" in district["scenario"].unique() else 0,
        key="cf_dist_scen")
    sub = district[
        (district["model"] == d_sel_model)
        & (district["crop"] == d_sel_crop)
        & (district["scenario"] == d_sel_scen)
    ]
    if not sub.empty:
        # Average across 2024-25 predictions per district
        agg = sub.groupby("landkreis").agg(
            baseline=("pred_baseline", "mean"),
            counterfactual=("pred_counterfactual", "mean"),
            delta=("delta_dt_ha", "mean"),
        ).reset_index()
        agg["delta_pct"] = agg["delta"] / agg["baseline"] * 100
        st.markdown(f"##### {d_sel_model} × {d_sel_crop} × {d_sel_scen}")
        cA, cB, cC = st.columns(3)
        with cA: kpi("Districts", f"{len(agg)}")
        with cB: kpi("Median Δ%", f"{agg['delta_pct'].median():+.2f} %")
        with cC: kpi("Δ range (dt/ha)",
                     f"{agg['delta'].min():.1f} to {agg['delta'].max():.1f}")

        fig = px.histogram(
            agg, x="delta_pct", nbins=30,
            labels={"delta_pct": "Per-district yield delta (%)"},
            title="Distribution of per-district yield delta",
        )
        fig.add_vline(x=0, line_dash="dot", line_color="grey")
        fig.update_layout(height=340)
        st.plotly_chart(fig, width="stretch")

        st.dataframe(
            agg.sort_values("delta_pct", ascending=False).head(20)
               .style.format({"baseline": "{:.2f}",
                              "counterfactual": "{:.2f}",
                              "delta": "{:+.2f}",
                              "delta_pct": "{:+.2f} %"}),
            hide_index=True, width="stretch",
        )

# --------------------------------------------------------------------------- #
# Footer
# --------------------------------------------------------------------------- #
st.markdown("---")

# GCM list read from the processed CMIP6 parquet — never hardcoded.
_cmip_path = PROJECT_ROOT / "data" / "processed" / "cmip6_nuts3.parquet"
_gcm_txt = ""
if _cmip_path.exists():
    try:
        _c = pd.read_parquet(_cmip_path, columns=["model"])
        _gcms = sorted(_c["model"].astype(str).unique())
        _gcm_txt = f"CMIP6 ensemble over {len(_gcms)} GCM(s): {', '.join(_gcms)}. "
    except (ValueError, KeyError, OSError):
        _gcm_txt = ""

st.caption(
    "**Methodology:** " + _gcm_txt +
    "Monthly output; historical baseline 2003-2014, future window "
    "2046-2050. Ensemble-mean monthly deltas are applied to the daily "
    "2024-25 features — additive for tasmax/tasmin, multiplicative for pr. "
    "The rsds delta is applied to `radiation_global`, which no model "
    "consumes under feature policy v3 (see the note at the top). "
    "Regenerate: `python scripts/09_cmip6_pull.py --model <name> && python "
    "scripts/10_cmip6_process.py && python scripts/13_counterfactual_daily.py`."
)

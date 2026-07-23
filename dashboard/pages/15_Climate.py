"""Climate counterfactual page — CMIP6 delta-method yield projections.

Reads ``results/metrics/counterfactual_daily_summary.csv`` and
``counterfactual_daily_district.csv`` (from ``scripts/13_counterfactual_daily.py``).
Applies additive tasmax/tasmin deltas + multiplicative pr/rsds deltas to
the 2024-25 observed daily features, then feeds each perturbed feature set
into the existing daily-trained models. Reports predicted yield anomaly
per (model, crop, scenario).

Data source: EC-Earth3-CC monthly, r1i1p1f1 realization. See paper
methodology; single-realization caveats explicitly documented.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import METRICS_DIR, divider, kpi, page_header

st.set_page_config(page_title="Climate counterfactual — Crop Yield Benchmark",
                   layout="wide")
page_header("Climate counterfactual (CMIP6 SSP2-4.5 / SSP5-8.5)",
            "Delta-method: apply (SSP future window - historical baseline) "
            "climate deltas to 2024-25 observed features, predict with the "
            "existing daily-trained models.")

# Compute per-model SSP5-8.5 deltas live from the current counterfactual CSV,
# so the TL;DR + detail blocks below never drift from the underlying numbers.
_cc_path = METRICS_DIR / "counterfactual_daily_summary.csv"
if _cc_path.exists():
    _cc = pd.read_csv(_cc_path)
    _ssp5 = _cc[_cc["scenario"] == "ssp5_8_5"]
    _per_model_5 = _ssp5.groupby("model")["delta_pct"].mean().round(2)
    _tas5 = float(_ssp5["tasmax_delta_C"].iloc[0]) if len(_ssp5) else float("nan")
    _tas2 = float(_cc[_cc["scenario"] == "ssp2_4_5"]["tasmax_delta_C"].iloc[0]) \
        if (_cc["scenario"] == "ssp2_4_5").any() else float("nan")
    _min_model = _per_model_5.idxmin()
    _max_model = _per_model_5.idxmax()
    _min_val = _per_model_5.min()
    _max_val = _per_model_5.max()
    _per_model_str = ", ".join(f"{m} ({v:+.2f}%)" for m, v in _per_model_5.items())
else:
    _per_model_5 = pd.Series(dtype=float)
    _tas5 = _tas2 = float("nan")
    _min_model = _max_model = "—"
    _min_val = _max_val = float("nan")
    _per_model_str = "counterfactual CSV missing"

st.markdown(
    "### TL;DR\n\n"
    f"Under SSP5-8.5 warming ({_tas5:+.2f} °C growing-season tasmax delta), "
    f"the 6 models diverge from {_min_val:+.2f}% ({_min_model}) "
    f"to {_max_val:+.2f}% ({_max_model}) predicted yield change — "
    "cross-model disagreement itself is a free out-of-distribution "
    "uncertainty signal."
)

with st.expander("More details — CMIP6 ensemble caveat & interpretation",
                 expanded=False):
    st.caption(
        "Do models agree under CMIP6 SSP2-4.5 and SSP5-8.5 warming? — "
        "a three-model ensemble stress test where cross-model "
        "disagreement becomes a free OOD-uncertainty signal."
    )
    st.info(
        "**Three-model CMIP6 ensemble caveat.** We use a 3-model ensemble "
        "(EC-Earth3-CC + MPI-ESM1-2-LR + MRI-ESM2-0, all r1i1p1f1). "
        f"Ensemble-mean growing-season tasmax delta under SSP5-8.5 is "
        f"**{_tas5:+.2f} °C** (SSP2-4.5: {_tas2:+.2f} °C). "
        "Still below the IPCC AR6 multi-model multi-realization mean "
        "(~+1.5-2 °C by 2050); multi-realization analysis is left for "
        "future work. Note that MRI-ESM2-0 is the only member showing the "
        "expected SSP5-8.5 > SSP2-4.5 warming ordering for Germany in "
        "this window — the ensemble mean is dragged flatter by internal "
        "variability in the other two single-realization runs."
    )
    st.info(
        "**Cross-model disagreement itself IS the finding.** Per-model "
        f"predicted yield change under SSP5-8.5: {_per_model_str}. "
        f"Spread from {_min_val:+.2f}% ({_min_model}) to "
        f"{_max_val:+.2f}% ({_max_model}) is a **free "
        "out-of-distribution uncertainty signal** — no explicit Bayesian "
        "machinery required. Trees and attention models with explicit "
        "variable selection tend to damp small input perturbations; "
        "recurrent architectures preserve them more."
    )

summary_path = METRICS_DIR / "counterfactual_daily_summary.csv"
district_path = METRICS_DIR / "counterfactual_daily_district.csv"
if not summary_path.exists():
    st.warning(
        "`results/metrics/counterfactual_daily_summary.csv` not found. "
        "Run `python scripts/13_counterfactual_daily.py`."
    )
    st.stop()

summary = pd.read_csv(summary_path)
district = pd.read_csv(district_path) if district_path.exists() else pd.DataFrame()

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
    title="Predicted yield change per crop, mean across 5 base models",
)
fig.add_vline(x=0, line_dash="dot", line_color="grey")
fig.update_layout(height=440)
st.plotly_chart(fig, width="stretch")

# --------------------------------------------------------------------------- #
# Per-model behavior — where the disagreement is
# --------------------------------------------------------------------------- #
divider("Per-model response — cross-model divergence is itself the signal")

st.caption(
    "Under out-of-distribution weather (climate change) individual models "
    "diverge. Cross-model disagreement is a free uncertainty estimate: "
    "high spread = low confidence in the projection."
)

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
st.caption(
    "Standard deviation across models per crop. High std = models disagree "
    "on how climate change affects this crop = high projected uncertainty. "
    "In the paper: this itself becomes a novel finding — ensemble "
    "disagreement calibrates OOD confidence for free."
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
st.caption(
    "**Methodology:** CMIP6 3-model ensemble (EC-Earth3-CC + MPI-ESM1-2-LR "
    "+ MRI-ESM2-0, all r1i1p1f1), monthly output. Historical baseline "
    "2003-2014, future window 2046-2050. Delta method: ensemble-mean "
    "monthly deltas applied to daily 2024-25 features (additive for "
    "tasmax/tasmin, multiplicative for pr/rsds — the latter sidesteps "
    "the DWD J/cm²/day vs CMIP6 W/m² unit mismatch). Regenerate: "
    "`python scripts/09_cmip6_pull.py --model <name> && python scripts/"
    "10_cmip6_process.py && python scripts/13_counterfactual_daily.py`."
)

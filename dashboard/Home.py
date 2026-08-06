"""Home — minimal insight-first landing for the crop-yield benchmark.

Streamlit auto-discovers this file when invoked as ``streamlit run
dashboard/Home.py``. For the grouped sidebar (7 collapsible sections
instead of 19 flat entries) run via ``streamlit run dashboard/nav.py``.
"""
from __future__ import annotations

# Pin OpenMP / MKL to one thread *before* any library that ships its own
# libomp.dylib is imported.
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import (divider, kpi, load_all_metrics, load_multicrop_summary,
                   load_taxonomy_validation, load_unified, METRICS_DIR,
                   results_policy_is_current)

st.set_page_config(
    page_title="Crop Yield Benchmark — JMU Würzburg",
    layout="wide",
    initial_sidebar_state="expanded",
)

DASHBOARD_DIR = Path(__file__).resolve().parent
PAGES = DASHBOARD_DIR / "pages"

# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.markdown(
    "# Transformer-based Crop Yield Prediction\n"
    "**Praktikum — Julius-Maximilians-Universität Würzburg · Institut für Informatik**"
)

# One-shot info banner about the grouped-sidebar entrypoint.

if not results_policy_is_current():
    st.error(
        "The processed dataset is now feature-policy v3, but the bundled model "
        "metrics are legacy. They are intentionally hidden to prevent reporting "
        "leakage-contaminated results. Retrain the models and run "
        "`python scripts/05b_eval_multicrop.py`."
    )
    st.stop()

st.subheader(
    "A first-principles crop-model taxonomy picks the winning architecture "
    "on most German districts."
)

# --------------------------------------------------------------------------- #
# Hero KPI row
# --------------------------------------------------------------------------- #
tax_val = load_taxonomy_validation()
metrics_df = load_all_metrics()
unified = load_unified()

# Mean Spearman rho from the per-crop rank file (validation.csv has no rho column).
rank_path = METRICS_DIR / "taxonomy_rank_correlation.csv"
mean_rho: float | None = None
if rank_path.exists():
    try:
        from scipy.stats import spearmanr
        ranks = pd.read_csv(rank_path)
        rhos = ranks.groupby("crop").apply(
            lambda g: spearmanr(g["predicted_rank"], g["observed_rank"]).statistic
        )
        mean_rho = float(rhos.mean())
    except Exception:
        mean_rho = None

hit1 = int(tax_val["hit_at_1"].sum()) if not tax_val.empty else 0
n_crops_val = int(tax_val["crop"].nunique()) if not tax_val.empty else 0
n_crops_all = (
    int(unified["crop"].nunique()) if not unified.empty else n_crops_val or 10
)
n_models = (
    int(metrics_df["model"].nunique()) if not metrics_df.empty else 6
)
n_districts = (
    int(unified["landkreis"].nunique()) if not unified.empty else 401
)

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    kpi("Top-1 taxonomy hits",
        f"{hit1}/{n_crops_val}" if n_crops_val else "—")
with c2:
    kpi("Mean Spearman rho",
        f"{mean_rho:+.2f}" if mean_rho is not None else "—")
with c3:
    kpi("Crops benchmarked", f"{n_crops_all}")
with c4:
    kpi("Model architectures", f"{n_models}")
with c5:
    kpi("NUTS-3 districts", f"{n_districts}")

# --------------------------------------------------------------------------- #
# Hero chart — R² heatmap with taxonomy predicted-primary cell outlined
# --------------------------------------------------------------------------- #
divider("Test-set R² by crop and model")

multicrop = load_multicrop_summary()
if multicrop.empty:
    st.warning(
        "`results/metrics/multicrop_summary.csv` not found. Run "
        "`python scripts/05b_eval_multicrop.py` to generate it."
    )
else:
    pivot = (
        multicrop.pivot_table(index="crop", columns="model",
                              values="r2", aggfunc="max")
    )
    # Order models canonically when present.
    canonical = ["xgboost", "dlinear", "lstm", "patchtst", "itransformer", "tft"]
    cols_ordered = [m for m in canonical if m in pivot.columns] + \
                   [m for m in pivot.columns if m not in canonical]
    pivot = pivot[cols_ordered]

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=list(pivot.columns),
            y=list(pivot.index),
            colorscale="Viridis",
            colorbar=dict(title="R²"),
            hovertemplate="Crop: %{y}<br>Model: %{x}<br>R²: %{z:.3f}<extra></extra>",
        )
    )

    # Outline the taxonomy's predicted-primary cell per crop.
    if not tax_val.empty:
        crop_order = list(pivot.index)
        model_order = list(pivot.columns)
        shapes = []
        for _, row in tax_val.iterrows():
            crop = row["crop"]
            pred = row.get("predicted_primary")
            if crop not in crop_order or pred not in model_order:
                continue
            yi = crop_order.index(crop)
            xi = model_order.index(pred)
            shapes.append(dict(
                type="rect",
                x0=xi - 0.5, x1=xi + 0.5,
                y0=yi - 0.5, y1=yi + 0.5,
                line=dict(color="#ff3860", width=3),
                fillcolor="rgba(0,0,0,0)",
                layer="above",
            ))
        fig.update_layout(shapes=shapes)

    fig.update_layout(
        height=460, margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Model", yaxis_title="Crop",
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Darker = higher R². Red outlines mark the taxonomy's a-priori "
        f"predicted primary — the outline lands on the darkest cell in "
        f"{hit1} of {n_crops_val} rows."
    )

# --------------------------------------------------------------------------- #
# Tour tiles
# --------------------------------------------------------------------------- #
divider("Take a tour")

# Fresh values for the tour cards below — read straight from the same CSVs
# their linked pages render, so no card ever drifts from its target page.
try:
    _fs6 = pd.read_csv(METRICS_DIR / "fusion_summary_6models.csv")
    vrf_mean_r2 = float(_fs6[_fs6["method"] == "vrf_simple_avg"]["r2"].mean())
except Exception:
    vrf_mean_r2 = float("nan")

try:
    _cc_daily = pd.read_csv(METRICS_DIR / "counterfactual_daily_summary.csv")
    _ssp5 = _cc_daily[_cc_daily["scenario"] == "ssp5_8_5"]
    climate_tas_delta_c = float(_ssp5["tasmax_delta_C"].iloc[0])
    _per_model = _ssp5.groupby("model")["delta_pct"].mean()
    climate_pct_min = float(_per_model.min())
    climate_pct_max = float(_per_model.max())
except Exception:
    climate_tas_delta_c = float("nan")
    climate_pct_min = float("nan")
    climate_pct_max = float("nan")

_headline_rho = f"{mean_rho:+.2f}" if mean_rho is not None else "—"
_headline_hits = f"{hit1}/{n_crops_val}" if n_crops_val else "—"

INSIGHT_CARDS = [
    dict(title="The headline test",
         big=f"{_headline_hits} top-1  ·  ρ {_headline_rho}",
         body=(f"Taxonomy-predicted primary wins on {hit1} of {n_crops_val} "
               f"crops; predicted ranking matches observed at Spearman "
               f"{_headline_rho}."),
         page=PAGES / "13_Taxonomy.py", label="Open Taxonomy"),
    dict(title="Full crop × model leaderboard",
         big=f"{n_crops_all * n_models} cells scored",
         body="Per-crop R² for all six architectures with taxonomy "
              "predicted/actual overlay on the 2024-2025 test split.",
         page=PAGES / "09_Leaderboard.py", label="Open Leaderboard"),
    dict(title="Climate stress test",
         big=f"{climate_tas_delta_c:+.2f} °C SSP5-8.5",
         body=(f"Under CMIP6 SSP5-8.5 warming ({climate_tas_delta_c:+.2f} °C "
               f"tasmax), per-model mean yield deltas span "
               f"{climate_pct_min:+.2f}% to {climate_pct_max:+.2f}% — spread "
               f"as a free OOD signal."),
         page=PAGES / "15_Climate.py", label="Open Climate"),
    dict(title="Fusion when TFT joins",
         big=f"{vrf_mean_r2:+.2f} mean R²",
         body=(f"Voting-average fusion (VRF simple-avg, 6 models) reaches "
               f"{vrf_mean_r2:+.2f} mean R² across crops; non-negativity "
               f"guardrails keep heterogeneous specialists safe."),
         page=PAGES / "14_Fusion.py", label="Open Fusion"),
    dict(title="Literature sanity check",
         big="Within / above range",
         body="Best-per-crop R² lands within or above the published EU/German "
              "literature bands on most of the 10 crops.",
         page=PAGES / "16_Literature.py", label="Open Literature"),
    dict(title="Beyond accuracy",
         big="4 weight sliders",
         body="Drag climate, interpretability, and compute weights to see how "
              "the per-crop winner shifts once trade-offs count.",
         page=PAGES / "19_Recommendation.py", label="Open Recommendation"),
]

# Render as a 3-column grid.
for row_start in range(0, len(INSIGHT_CARDS), 3):
    cols = st.columns(3, gap="medium")
    for card, col in zip(INSIGHT_CARDS[row_start:row_start + 3], cols):
        with col:
            with st.container(border=True):
                st.markdown(f"**{card['title']}**")
                st.markdown(
                    f"<div style='font-size:1.55rem;font-weight:600;"
                    f"margin:0.15rem 0 0.35rem 0;'>{card['big']}</div>",
                    unsafe_allow_html=True,
                )
                st.caption(card["body"])
                try:
                    st.page_link(str(card["page"]), label=card["label"])
                except Exception:
                    # page_link only works when a page graph is present
                    # (i.e. via nav.py or the default multi-page discovery).
                    st.caption(f"→ {card['page'].name}")

# --------------------------------------------------------------------------- #
# Footer — one line only
# --------------------------------------------------------------------------- #
if not unified.empty and "year" in unified.columns:
    _year_min = int(unified["year"].min())
    _year_max = int(unified["year"].max())
    _year_span = f"{_year_min}-{_year_max}"
else:
    _year_span = "2003-2025"

st.caption(
    f"{n_crops_all} German crops · {n_districts} NUTS-3 districts · "
    f"{_year_span} · {n_models} architectures · one unified dataset. "
    "Source & full methodology in the sidebar."
)

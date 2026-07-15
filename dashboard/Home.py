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
                   load_taxonomy_validation, load_unified, METRICS_DIR)

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
st.info(
    "For the grouped sidebar (7 sections), launch with "
    "`streamlit run dashboard/nav.py`. Running "
    "`streamlit run dashboard/Home.py` directly still works but shows a "
    "flat 19-entry sidebar.",
    icon=None,
)

st.subheader(
    "A first-principles crop-model taxonomy picks the winning architecture "
    "on 9 of 10 crops across 401 German districts."
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

INSIGHT_CARDS = [
    dict(title="The headline test",
         big="9/10 top-1  ·  ρ +0.69",
         body="Taxonomy-predicted primary wins on 9 of 10 crops; predicted "
              "ranking matches observed at Spearman +0.69.",
         page=PAGES / "13_Taxonomy.py", label="Open Taxonomy"),
    dict(title="Full crop × model leaderboard",
         big="60 cells scored",
         body="Per-crop R² for all six architectures with taxonomy "
              "predicted/actual overlay on the 2022-2023 test split.",
         page=PAGES / "09_Leaderboard.py", label="Open Leaderboard"),
    dict(title="Climate stress test",
         big="+0.58 °C SSP5-8.5",
         body="Under CMIP6 warming, models diverge from -0.43 % to +0.60 % "
              "yield change — spread as a free OOD signal.",
         page=PAGES / "15_Climate.py", label="Open Climate"),
    dict(title="Fusion when TFT joins",
         big="+0.19 mean R²",
         body="Adding TFT lifts per-crop val-selection by ~0.19 R²; "
              "non-negativity guardrails keep heterogeneous specialists safe.",
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
st.caption(
    "10 German crops · 401 NUTS-3 districts · 2003-2023 · 6 architectures · "
    "one unified dataset. Source & full methodology in the sidebar."
)

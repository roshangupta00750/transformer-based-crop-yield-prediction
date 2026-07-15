"""Home page — story-arc landing for the crop-yield benchmark.

Streamlit looks for a top-level script (this file) as the landing page. All
other pages live in ``pages/`` and appear in the sidebar in numeric order.
"""
from __future__ import annotations

# Pin OpenMP / MKL to one thread *before* any library that ships its own
# libomp.dylib is imported. Without this the dashboard segfaults on macOS
# Intel when two such libraries (torch, xgboost, lightning) end up loaded
# in the same Python process.
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import streamlit as st

from utils import (divider, kpi, load_all_metrics, load_taxonomy_validation,
                   load_unified)

st.set_page_config(
    page_title="Crop Yield Benchmark — JMU Würzburg",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.markdown(
    "# Transformer-based Crop Yield Prediction\n"
    "**Praktikum — Julius-Maximilians-Universität Würzburg · Institut für Informatik**"
)
st.caption(
    "A first-principles Crop-Model Compatibility Taxonomy that picks the "
    "right model for each of ten German crops — validated on a "
    "six-architecture, 401-district, 21-year benchmark and stress-tested "
    "three ways."
)

# --------------------------------------------------------------------------- #
# KPI strip
# --------------------------------------------------------------------------- #
df = load_unified()
metrics = load_all_metrics()
tax_val = load_taxonomy_validation()

c1, c2, c3, c4, c5, c6 = st.columns(6)
with c1:
    kpi("Landkreise (NUTS-3)", f"{df['landkreis'].nunique() if not df.empty else 0}")
with c2:
    yr_range = (
        f"{int(df['year'].min())}–{int(df['year'].max())}"
        if not df.empty else "—"
    )
    kpi("Years covered", yr_range)
with c3:
    kpi("Crops", f"{df['crop'].nunique() if not df.empty else 0}")
with c4:
    kpi("Unified rows", f"{len(df):,}" if not df.empty else "—")
with c5:
    kpi("Models trained", f"{metrics['model'].nunique() if not metrics.empty else 0} / 6")
with c6:
    if not tax_val.empty:
        n = len(tax_val)
        hit1 = int(tax_val["hit_at_1"].sum())
        kpi("Taxonomy top-1",
            f"{hit1}/{n} ({hit1/n:.0%})",
            delta=f"top-3: {int(tax_val['hit_at_3'].sum())}/{n}")
    else:
        kpi("Taxonomy top-1", "—")

st.markdown("---")

# --------------------------------------------------------------------------- #
# Story arc navigator — seven chapters
# --------------------------------------------------------------------------- #
STORY_ARC = [
    {
        "num": 1,
        "title": "Why This Benchmark",
        "intro": (
            "Crop-yield forecasting steers food-security policy, farmer "
            "decisions, and climate-adaptation planning — yet the ML "
            "literature offers a dozen candidate architectures with no "
            "principled way to pick among them. This chapter frames the "
            "gap the benchmark closes: matching model to crop from first "
            "principles rather than shopping the leaderboard after the fact."
        ),
        "pages": ["Why This Benchmark"],
    },
    {
        "num": 2,
        "title": "The Data",
        "intro": (
            "Every claim downstream rests on one unified dataset — 10 "
            "German crops, 401 NUTS-3 districts, and 21 years (2003-2023) — "
            "assembled from weather, satellite, soil, and yield sources. "
            "This chapter documents provenance, licensing, and coverage, "
            "and lets you preview each modality live before any model "
            "touches it."
        ),
        "pages": ["The Data"],
    },
    {
        "num": 3,
        "title": "Six Architectures",
        "intro": (
            "Six model families sit head-to-head: XGBoost, DLinear, LSTM, "
            "PatchTST, iTransformer, and the Temporal Fusion Transformer. "
            "Each page opens the box on hyperparameters, in-season horizon "
            "curves, predicted-vs-actual scatter, residual diagnostics — "
            "and, for TFT alone, quantile calibration on the P10-P90 band."
        ),
        "pages": ["XGBoost", "DLinear", "LSTM", "PatchTST", "iTransformer", "TFT"],
    },
    {
        "num": 4,
        "title": "Empirical Results",
        "intro": (
            "No single architecture wins every crop — the best-per-model "
            "leaderboard, per-crop matrix, in-season decay curves, spatial "
            "holdout maps, and attention-based interpretability comparisons "
            "all say so. This chapter is the empirical bedrock the taxonomy "
            "will have to explain."
        ),
        "pages": ["Head-to-head Leaderboard", "In-season Horizon",
                  "Spatial Holdout", "Interpretability"],
    },
    {
        "num": 5,
        "title": "The Taxonomy",
        "intro": (
            "The paper's headline contribution: a first-principles "
            "Crop-Model Compatibility Taxonomy that predicts each crop's "
            "winning architecture from crop biology and model inductive "
            "bias — a testable hypothesis, not a post-hoc rationalization. "
            "It hits top-1 on 9 of 10 crops and top-3 on all 10."
        ),
        "pages": ["Crop-Model Taxonomy"],
    },
    {
        "num": 6,
        "title": "Stress Tests",
        "intro": (
            "If the taxonomy story is real it should survive three "
            "independent stress tests — cross-model fusion (does the "
            "taxonomy's winner still lead once ensembles enter the arena?), "
            "CMIP6 climate counterfactuals (do models agree under OOD "
            "warming?), and comparison against published EU crop-yield "
            "benchmarks. Each section carries its own caveats banner so "
            "the reader sees where the evidence is directional versus "
            "conclusive."
        ),
        "pages": ["Cross-model Fusion", "Climate Counterfactual",
                  "Literature Comparison"],
    },
    {
        "num": 7,
        "title": "Reproduce",
        "intro": (
            "Everything above is open-source and re-runnable — this "
            "chapter is the entry point. Launch training runs from the "
            "browser (routed to the GPU box when reachable, falling back "
            "to local CPU for TFT), then drill into any trained model's "
            "per-district predictions on the German NUTS-3 choropleth."
        ),
        "pages": ["Train Models", "Geographic Predictions"],
    },
]

divider("The story in seven chapters")
st.caption(
    "The dashboard is ordered like a research paper — motivation, data, "
    "methods, results, headline claim, robustness, reproducibility. Read "
    "top-to-bottom for the full argument, or jump to any chapter below."
)

for chap in STORY_ARC:
    divider(f"Chapter {chap['num']} — {chap['title']}")
    st.markdown(chap["intro"])
    st.markdown("**Pages in this chapter:**")
    for pg in chap["pages"]:
        st.markdown(f"- {pg}")

# --------------------------------------------------------------------------- #
# Project context
# --------------------------------------------------------------------------- #
divider("About the project")
st.markdown("""
This dashboard wraps the reproducible benchmark described in
`Crop_Yield_Transformer_Preliminary.pptx`. The research question is:

> *On a single standardized German crop-yield dataset, which transformer
> family delivers the best accuracy–interpretability trade-off for
> **in-season** prediction (30 / 60 / 90 / 120 days before harvest)?*

**Why anomaly framing?** 97 % of yield variance is between-district. Models
predict the (district, year) **anomaly** rather than the absolute yield, so
R² measures the part of the signal models actually contribute — not just
"knowing the district".
""")

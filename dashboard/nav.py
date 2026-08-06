"""Grouped-sidebar entrypoint for the Crop-Yield Benchmark dashboard.

Streamlit's automatic multi-page discovery gives you one flat list of 19
sidebar entries, which is unusable. This file builds an ``st.navigation``
object with named sections so the sidebar collapses into ~7 groups.

Run with::

    streamlit run dashboard/nav.py

The direct ``streamlit run dashboard/Home.py`` entrypoint also still
works, but that path falls back to the flat auto-discovered sidebar.
"""
from __future__ import annotations

# Pin OpenMP / MKL to one thread *before* any library that ships its own
# libomp.dylib is imported. Same rationale as Home.py / utils.py.
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path

import streamlit as st

DASH = Path(__file__).resolve().parent
PAGES = DASH / "pages"


def _page(path: Path, title: str, *, default: bool = False) -> "st.navigation.Page":
    return st.Page(str(path), title=title, default=default)


SECTIONS = {
    "Overview": [
        _page(DASH / "Home.py", "Home", default=True),
        _page(PAGES / "02_Data.py", "Data"),
    ],
    "Models": [
        _page(PAGES / "03_XGBoost.py", "XGBoost"),
        _page(PAGES / "04_DLinear.py", "DLinear"),
        _page(PAGES / "05_LSTM.py", "LSTM"),
        _page(PAGES / "06_PatchTST.py", "PatchTST"),
        _page(PAGES / "07_iTransformer.py", "iTransformer"),
        _page(PAGES / "08_TFT.py", "TFT"),
    ],
    "Findings": [
        _page(PAGES / "09_Leaderboard.py", "Leaderboard"),
        _page(PAGES / "10_InSeason.py", "In-season Horizon"),
        _page(PAGES / "11_Spatial.py", "Spatial Holdout"),
        _page(PAGES / "12_Interpret.py", "Interpretability"),
    ],
    "Headline": [
        _page(PAGES / "13_Taxonomy.py", "Crop-Model Taxonomy"),
    ],
    "Robustness": [
        _page(PAGES / "14_Fusion.py", "Cross-model Fusion"),
        _page(PAGES / "15_Climate.py", "Climate Counterfactual"),
        _page(PAGES / "16_Literature.py", "Literature Comparison"),
    ],
    "Reproduce": [
        _page(PAGES / "17_Train.py", "Train Models"),
        _page(PAGES / "18_Geographic.py", "Geographic Predictions"),
    ],
    "Supplementary": [
        _page(PAGES / "19_Recommendation.py", "Multi-criteria Recommendation"),
    ],
}


nav = st.navigation(SECTIONS, position="sidebar", expanded=True)
nav.run()

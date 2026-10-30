"""Entrypoint for the Crop-Yield dashboard: results first, background, then the archive.

Run with::

    streamlit run dashboard/nav.py

Use this entrypoint, not ``dashboard/Home.py``: running Home.py directly makes
Streamlit auto-discover every file in ``pages/`` and show them all as one flat
list.

Pages that left the menu are listed in ``ARCHIVED`` below. Nothing was
deleted; moving an entry from ``ARCHIVED`` into ``PAGE_LIST`` restores it.
Their key results now live on Predictions (best model per crop) and Evidence
(taxonomy test, fusion, literature, spatial holdout).
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
VIEWS = DASH / "views"


def _page(path: Path, title: str, *, default: bool = False) -> "st.navigation.Page":
    if default:
        return st.Page(str(path), title=title, default=True)
    return st.Page(str(path), title=title, url_path=title.lower().replace(" ", "-"))


PAGE_LIST = {
    "Dashboard": [
        _page(VIEWS / "bi.py", "Dashboard", default=True),
    ],
    "Report": [
        _page(VIEWS / "home.py", "Overview"),
        _page(VIEWS / "insights.py", "Key insights"),
        _page(VIEWS / "explorer.py", "Predictions vs actual"),
        _page(VIEWS / "accuracy.py", "Accuracy"),
        _page(VIEWS / "bad_years.py", "Bad years"),
        _page(VIEWS / "papers.py", "Compared with papers"),
        _page(VIEWS / "journey.py", "Model journey"),
        _page(VIEWS / "forecast.py", "Forecasts"),
    ],
    "Report: analytics": [
        _page(VIEWS / "where_grown.py", "Where crops are grown"),
        _page(VIEWS / "errors.py", "Error analysis"),
        _page(VIEWS / "weather.py", "Weather and yield"),
    ],
    "Background": [
        _page(VIEWS / "method.py", "How it works"),
        _page(VIEWS / "data.py", "Data"),
    ],
    # The earlier six-model benchmark (legacy, pre-policy-v3 results), kept for reference.
    "Archive (older work)": [
        _page(VIEWS / "overview.py", "6-model test"),
        _page(VIEWS / "geographic.py", "6-model map"),
        _page(VIEWS / "predictions.py", "6-model results"),
        _page(PAGES / "17_Train.py", "Training tool"),
        _page(PAGES / "20_Evidence.py", "Evidence notes"),
    ],
}

# Off the menu, kept on disk. Move an entry into PAGE_LIST to restore it.
ARCHIVED = {
    "Problem": PAGES / "01_Problem.py",
    "XGBoost": PAGES / "03_XGBoost.py",
    "DLinear": PAGES / "04_DLinear.py",
    "LSTM": PAGES / "05_LSTM.py",
    "PatchTST": PAGES / "06_PatchTST.py",
    "iTransformer": PAGES / "07_iTransformer.py",
    "TFT": PAGES / "08_TFT.py",
    "Leaderboard": PAGES / "09_Leaderboard.py",
    "In-season Horizon": PAGES / "10_InSeason.py",
    "Interpretability": PAGES / "12_Interpret.py",
    "Crop-Model Taxonomy": PAGES / "13_Taxonomy.py",
    "Cross-model Fusion": PAGES / "14_Fusion.py",
    "Climate Counterfactual": PAGES / "15_Climate.py",
    "Literature Comparison": PAGES / "16_Literature.py",
    "Geographic (stub)": PAGES / "18_Geographic.py",
    "Multi-criteria Recommendation": PAGES / "19_Recommendation.py",
}


st.set_page_config(page_title="YieldFormer-DE", page_icon="🌾", layout="wide")
nav = st.navigation(PAGE_LIST, position="top")
nav.run()

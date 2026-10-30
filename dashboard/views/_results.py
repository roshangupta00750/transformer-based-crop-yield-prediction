"""Shared loaders, labels and plain-language help for the results pages.

All numbers come from results/dashboard/, built by scripts/63_dashboard_data.py from
the pre-registered evaluations (see that script for every definition). Colour rule:
our model is always blue, the 5-year average always grey, bad years red, good years
green-blue, and "better / worse than the 5-year average" is a blue <-> red scale.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import CROP_LABEL, CROPS, DIVERGING, MUTED, SEQ_BLUE, district_map  # noqa: E402,F401

ROOT = Path(__file__).resolve().parents[2]
DASH = ROOT / "results" / "dashboard"

MODEL = "#2a78d6"        # our model
AVG5 = MUTED              # 5-year average
AVG_LONG = "#bdbcb6"      # long-term average
BAD = "#e34948"
GOOD = "#1baf7a"
PAPER = "#eda100"         # published papers
GRID = "#e1e0d9"
MAIN = "v18"              # final model: ensemble of iTransformer + TFT + PatchTST + district error memory
MODEL_NAME = "YieldFormer-DE"   # name of the final model (= v18; v15 was the iTransformer-only version)
MAIN_LABEL = f"{MODEL_NAME} ({MAIN})"   # column/legend label of the final model everywhere
# single networks + error memory (the ensemble's members); names chosen by the user 2026-10-12
NAME_I, NAME_T, NAME_P = "YieldFormer-DE-i (v15)", "YieldFormer-DE-T (v17)", "YieldFormer-DE-P (v17)"
WEATHER_LABEL = {"rain_autumn_prev(OND)": "Rain in the autumn before (Oct–Dec)",
                 "rain_spring(MAM)": "Rain in spring (Mar–May)", "rain_summer(JJ)": "Rain in June–July",
                 "rain_harvest(Jul15-Aug15)": "Rain at harvest (mid-Jul – mid-Aug)",
                 "cwb_spring(MAM)": "Water balance in spring (rain minus evaporation)",
                 "cwb_summer(JJ)": "Water balance in June–July", "tmax_summer(JJ)": "Daily max. temperature, June–July",
                 "heatdays_JJ": "Hot days, June–July", "soilmoist_JJ": "Soil moisture, June–July"}
PAPERS_MODEL = "v12"      # latest model run on the papers' own splits

HELP = {
    "accuracy": "Accuracy = 100% minus the average forecast error in percent. "
                "88% means a typical district forecast is about 12% away from the real yield.",
    "avg5": "The simplest honest forecast: the average yield of the previous five years in that "
            "district. Any useful model must beat it.",
    "avg_long": "The district's long-term average with its trend (yields rise over time).",
    "unseen": "Every year is estimated by a model that never saw that year's yields: it is trained "
              "on earlier years only, exactly as it would be used in practice.",
    "national": "National yield = plain average over all districts that reported the crop that year (not weighted by area).",
    "within10": "Share of district forecasts that land within ±10% of the real yield.",
    "bad_year": "A crop-year whose national yield was at least 10% below its long-term average.",
    "hindcast": "2010-2015: kept aside while ideas were developed and used only to decide whether to adopt each "
                "version (from v6 on). Because about 9 adoption decisions were made on these same years, it is the "
                "cleanest test we have, but not a perfectly fresh one.",
    "rule95": "A change is adopted only if, when the test years are resampled 10,000 times, it comes out better in "
              "at least 95% of them. With about 9 decisions on the same years, one lucky pass is possible; a "
              "stricter correction for that would ask for about 99.4%.",
    "dt_ha": "dt/ha = decitonnes per hectare = 100 kg per hectare. 80 dt/ha = 8 tonnes per hectare.",
}


def header(title: str, lead: str) -> None:
    """Page title plus a one-paragraph plain-language answer."""
    st.markdown(f"## {title}")
    st.markdown(f"<div style='font-size:1.08rem;line-height:1.55;margin:0.2rem 0 0.8rem 0;max-width:64rem'>{lead}</div>",
                unsafe_allow_html=True)


def plain_layout(fig, height: int = 420, **kw):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      legend=dict(orientation="h", yanchor="top", y=-0.22, x=0), **kw)
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def crop_label(c: str) -> str:
    return CROP_LABEL.get(c, c.replace("_", " ").capitalize())


def missing(name: str) -> None:
    st.info(f"`results/dashboard/{name}` not found. Run `python scripts/63_dashboard_data.py`.")


@st.cache_data(show_spinner=False)
def load_summary() -> dict:
    f = DASH / "summary.json"
    return json.loads(f.read_text()) if f.exists() else {}


@st.cache_data(show_spinner=False)
def load_table(name: str) -> pd.DataFrame:
    f = DASH / name
    if not f.exists():
        return pd.DataFrame()
    return pd.read_parquet(f) if f.suffix == ".parquet" else pd.read_csv(f)

"""Dashboard contract (dashboard/SPEC.md): every page runs and keeps its required elements.

Fails when a page raises or when a required heading/text disappears, so later edits cannot
silently drop features. Skipped when streamlit or the built dashboard data are missing.
"""
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
VIEWS = ROOT / "dashboard" / "views"
DATA = ROOT / "results" / "dashboard" / "summary.json"

st_testing = pytest.importorskip("streamlit.testing.v1")
pytestmark = pytest.mark.skipif(not DATA.exists(), reason="run scripts/63_dashboard_data.py first")

REQUIRED = {
    "bi.py": ["YieldFormer-DE dashboard", "Filters", "Accuracy", "Typical error", "Within ±10%", "National accuracy",
              "Bad years warned", "Estimates", "Overview", "Drill-down table", "Errors", "Model comparison", "Weather",
              "Crop areas", "Trends", "Click a row to see", "All crops",
              "Accuracy by crop", "How close the estimates are", "Gain over the comparison by year", "Accuracy by state",
              "Data-quality check", "comparisons won by YieldFormer-DE", "YieldFormer-DE vs AgroForecast-DE",
              "Which weather matters for", "Real vs estimate by crop", "Top 12 districts"],
    "home.py": ["YieldFormer-DE", "Which model is which", "Why the counts differ", "How it works",
                "Germany's harvest: real vs estimate", "Top insights"],
    "insights.py": [f"#### {i}." for i in range(1, 11)],
    "explorer.py": ["Year by year: Germany", "Year × crop table", "Every estimate"],
    "where_grown.py": ["Crop area by district", "Share of farmland", "Average yield by district", "Area by state",
                       "Top districts"],
    "accuracy.py": ["By crop", "By year", "Map", "Forecast vs real", "One district"],
    "errors.py": ["Error distribution", "Error by state", "Error vs crop area", "Over- or under-estimate by year",
                  "Where the error comes from", "Errors repeat", "Cautious estimates", "Largest misses"],
    "weather.py": ["Which weather matters for which crop", "One crop, one weather signal", "Drought years in the weather"],
    "bad_years.py": ["three rules side by side", "Every bad harvest", "All years at a glance", "Limits, stated plainly"],
    "papers.py": ["What this means", "Every version, every comparison", "How the comparison changed",
                  "JRC MARS, Germany 2025"],
    "journey.py": ["All versions on the same forecasts", "How sure is the gain", "Is the transformer needed?",
                   "Many decisions, same test years", "Every version", "What we learned"],
    "forecast.py": ["Forecasts", "YieldFormer-DE forecasts", "Earlier forecast from v1", "not the final model",
                    "Germany: 2026 forecast vs normal", "Per district", "How the forecasts did in 2024 and 2025", "first check against preliminary official yields"],
    "method.py": ["The question", "The data", "The model", "How we test it honestly", "Glossary"],
    "data.py": ["Yield trend per crop", "One district, one season", "Data coverage"],
}


def _texts(at) -> str:
    out = []
    for kind in ("markdown", "caption", "title", "header", "subheader", "info", "warning", "error"):
        out += [str(getattr(e, "value", "")) for e in getattr(at, kind, [])]
    out += [str(getattr(t, "label", "")) for t in at.tabs]
    out += [str(getattr(m, "label", "")) for m in at.metric]
    out += [str(getattr(x, "label", "")) for x in getattr(at, "expander", [])]
    out += [str(getattr(x, "label", "")) for x in getattr(at, "button", [])]
    return "\n".join(out)


@pytest.mark.parametrize("page", sorted(REQUIRED))
def test_page_runs_and_keeps_required_elements(page):
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    at = st_testing.AppTest.from_file(str(VIEWS / page), default_timeout=240)
    at.run()
    assert not at.exception, f"{page} raised: {[e.value for e in at.exception][:2]}"
    text = _texts(at)
    missing = [r for r in REQUIRED[page] if r not in text]
    assert not missing, f"{page} lost required elements: {missing}"
    import re
    raw = [m for m in re.findall(r"\{[A-Za-z_][^{}\n]{0,60}(?::[.,+\d]*[dfg%])?\}", text)
           if ";" not in m]                                    # CSS rules inside markdown are fine
    assert not raw, f"{page} shows unformatted f-string placeholders (missing f prefix?): {raw[:3]}"


def test_every_page_in_navigation():
    nav = (ROOT / "dashboard" / "nav.py").read_text()
    lost = [p for p in REQUIRED if f'"{p}"' not in nav]
    assert not lost, f"pages missing from the navigation: {lost}"


def test_no_heatmaps_or_colour_coded_tables():
    """SPEC: chart form follows the question — no heatmaps of numbers or gradient-coloured tables."""
    banned = ("go.Heatmap", "Histogram2d", "background_gradient", "px.imshow")
    hits = [f"{p}: {b}" for p in REQUIRED for b in banned if b in (VIEWS / p).read_text()]
    assert not hits, f"heatmap-style visuals are not allowed: {hits}"


def test_crops_are_not_ranked():
    """SPEC: every crop is treated the same — fixed crop order, no weakest/strongest framing."""
    banned = ("Weakest", "Strongest", "weakest first", "Biggest gain", "Largest gain for", "least for")
    hits = [f"{p}: {b}" for p in REQUIRED for b in banned if b in (VIEWS / p).read_text()]
    assert not hits, f"ranking language/sorting is not allowed: {hits}"


def test_no_month_labels_in_navigation():
    """SPEC: page labels carry no month (user request 2026-10-09: no 'Aug' model labels)."""
    nav = (ROOT / "dashboard" / "nav.py").read_text()
    import re
    hits = re.findall(r'_page\([^)]*"((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*:?[^"]*)"\)', nav)
    assert not hits, f"month in navigation labels: {hits}"


def test_only_transformer_family_and_one_reference():
    """SPEC model family rule (user, 2026-10-10): transformer candidates, simple baselines, and exactly one
    non-transformer reference ("XGBoost + memory (reference)", comparison only); no other tree variant."""
    ref = "XGBoost + memory (reference)"
    for p in REQUIRED:
        txt = (VIEWS / p).read_text()
        assert "Gradient boosting" not in txt and '"G16"' not in txt, f"{p}: non-reference tree variant"
    bi = (VIEWS / "bi.py").read_text()
    assert 'st.selectbox("Model", [m for m in MODELS if m != REFERENCE]' in bi, "reference must never be the model"
    d = ROOT / "results" / "dashboard"
    rows = d / "model_rows.parquet"
    if rows.exists():
        import pandas as pd
        trees = [c for c in pd.read_parquet(rows).columns if "xgboost" in c.lower() or "boosting" in c.lower()]
        assert trees == [ref], trees
    for f in d.glob("v16_*.csv"):
        assert ",G16," not in f.read_text(), f.name


def test_yieldformer_name_only_for_the_ensemble():
    """SPEC naming (user, 2026-10-12): 'YieldFormer-DE' alone = the ensemble; single networks are -i / -T / -P."""
    import re
    bad = []
    for p in REQUIRED:
        txt = (VIEWS / p).read_text()
        bad += [f"{p}: {m}" for m in re.findall(r"YieldFormer-DE \(?v1[0-7]\b[^\"]{0,20}", txt)]
    assert not bad, f"single-network versions labelled as YieldFormer-DE: {bad}"

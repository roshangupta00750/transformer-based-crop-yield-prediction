"""Contract of the one-screen dashboard (dashboard/app.py, dashboard/SPEC.md part A).

Every view must run without an exception and keep its required elements, also with a district and a crop selected,
so later edits cannot silently drop the forecasts, the district card or the downloads.
"""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "dashboard" / "app.py"
DATA = ROOT / "results" / "dashboard" / "model_rows.parquet"

st_testing = pytest.importorskip("streamlit.testing.v1")
pytestmark = pytest.mark.skipif(not DATA.exists(), reason="run scripts/63_dashboard_data.py first")

CARDS = ["Accuracy", "Better in", "Within ±10%", "Poor years detected", "2026 forecast"]
VIEWS = ["Map", "District look-up", "Table & download", "Forecasts", "Compare"]
REQUIRED = {
    "Map": ["Colour the map by", "Click a district to open its card"],
    "District look-up": ["Year by year", "All crops in this district", "Download this district"],
    "Table & download": ["Estimates in the selection", "Estimates in this selection (CSV)",
                         "2026 forecasts in this selection (CSV)", "Data dictionary (CSV)"],
    "Forecasts": ["Forecast shown", "District forecasts", "preliminary official national yields",
                  "Also sealed: 2027 and 2028"],
    "Compare": ["Accuracy on the same"],
}


def _text(at) -> str:
    parts = [m.value for m in at.markdown] + [c.value for c in at.caption] + [i.value for i in at.info]
    parts += [b.label for b in at.button] + [r.label for r in at.radio] + [str(o) for r in at.radio for o in r.options]
    parts += [e.label for e in at.expander] + [h.value for h in at.subheader] + [h.value for h in at.header]
    parts += [t.label for t in at.text_input] + [c.label for c in at.checkbox]
    parts += [getattr(d.proto, "label", "") for d in at.get("download_button")]
    return "\n".join(str(p) for p in parts)


@pytest.fixture(scope="module")
def app():
    at = st_testing.AppTest.from_file(str(APP), default_timeout=180)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def test_cards_and_views_present(app):
    text = _text(app)
    for c in CARDS:
        assert c.upper() in text.upper(), f"card missing: {c}"
    assert app.radio(key="view").options == VIEWS


def test_only_final_model_is_default_and_reference_is_not_a_model(app):
    assert app.selectbox(key="f_model").value == "YieldFormer-DE (v18)"
    assert "XGBoost + memory (reference)" not in app.selectbox(key="f_model").options


@pytest.mark.parametrize("district,crop", [("All districts", "All crops"), ("Uckermark", "winter_wheat")])
@pytest.mark.parametrize("view", VIEWS)
def test_each_view_runs_and_keeps_its_elements(view, district, crop):
    at = st_testing.AppTest.from_file(str(APP), default_timeout=180)
    at.run()
    at.selectbox(key="f_crop").set_value(crop).run()
    at.selectbox(key="f_district").set_value(district).run()
    at.radio(key="view").set_value(view).run()
    assert not at.exception, [e.value for e in at.exception]
    assert not [w for w in at.warning if "deprecated" in str(w.value).lower()]
    text = _text(at)
    need = REQUIRED[view] if (view != "District look-up" or district != "All districts") else ["Choose a district above"]
    for item in need:
        assert item in text, f"{view}: missing '{item}'"


@pytest.mark.parametrize("part", ["Simple averages and members", "Seasons & regions", "Published models",
                                  "JRC MARS 2025", "Poor years", "Computing cost"])
def test_compare_parts(part):
    at = st_testing.AppTest.from_file(str(APP), default_timeout=180)
    at.run()
    at.radio(key="view").set_value("Compare").run()
    at.radio(key="cmp_part").set_value(part).run()
    assert not at.exception, [e.value for e in at.exception]


def test_crops_are_listed_in_fixed_order(app):
    assert app.selectbox(key="f_crop").options[1:] == [
        "Winter wheat", "Winter barley", "Rye", "Triticale", "Winter rapeseed", "Spring barley", "Oats",
        "Silage maize", "Potatoes", "Sugar beet"]


@pytest.mark.parametrize("crop", ["All crops", "winter_wheat"])
@pytest.mark.parametrize("layer", ["vs_normal", "error", "accuracy"])
def test_map_shows_past_harvests(crop, layer):
    """User, 2026-10-16: the map must not be limited to 2026; any harvest 2010-2025 can be mapped."""
    at = st_testing.AppTest.from_file(str(APP), default_timeout=180)
    at.run()
    at.selectbox(key="f_crop").set_value(crop).run()
    at.selectbox(key="map_year").set_value(2018).run()
    at.radio(key="map_layer").set_value(layer).run()
    assert not at.exception, [e.value for e in at.exception]
    assert "2018 harvest" in at.selectbox(key="map_year").options      # AppTest lists the formatted labels


def test_region_filter_narrows_states_and_selection():
    """User, 2026-10-16: North / East / South / West in the filters."""
    at = st_testing.AppTest.from_file(str(APP), default_timeout=180)
    at.run()
    at.selectbox(key="f_region").set_value("East").run()
    assert not at.exception
    assert set(at.selectbox(key="f_state").options) == {"All states", "Brandenburg", "Mecklenburg-Vorpommern",
                                                        "Sachsen", "Sachsen-Anhalt", "Thüringen"}
    assert "East region" in _text(at)
    at.selectbox(key="f_state").set_value("Saxony").run()
    at.selectbox(key="f_region").set_value("West").run()      # a state outside the region is cleared
    assert at.selectbox(key="f_state").value == "All states"


def test_seasons_view_has_no_winner_without_tie_rule():
    """Season leads under 0.5 pp are marked as ties (the most accurate network is mostly noise)."""
    at = st_testing.AppTest.from_file(str(APP), default_timeout=180)
    at.run()
    at.radio(key="view").set_value("Compare").run()
    at.radio(key="cmp_part").set_value("Seasons & regions").run()
    text = _text(at)
    assert "(tie)" in text and "Within 0.5 pp of the season" in text and "By region" in text


def test_region_panel_answers_both_questions(app):
    """User, 2026-10-16: show both answers to "which region does better" - most accurate and largest gain."""
    text = _text(app)
    assert "Which region does better?" in text
    assert "Most accurate:" in text and "Gains most over the" in text


def test_computing_cost_view():
    """User, 2026-10-16: GPU per model, time per model, size of the model."""
    at = st_testing.AppTest.from_file(str(APP), default_timeout=180)
    at.run()
    at.radio(key="view").set_value("Compare").run()
    at.radio(key="cmp_part").set_value("Computing cost").run()
    assert not at.exception, [e.value for e in at.exception]
    text = _text(at)
    for item in ("GPU time, final model", "Parameters", "GPU (runs)", "Training run, median (min)", "GPU hours"):
        assert item in text, item


def test_no_yellow_panels():
    """User, 2026-10-16: no yellow tints."""
    src = APP.read_text().lower()
    assert '"sand"' not in src and "#fbf5e6" not in src


def test_old_pages_are_not_reachable():
    """dashboard/main.py must use st.navigation (turns off auto-discovery of dashboard/pages/) and run.sh must
    start main.py, so the archive pages and the training console cannot be opened from this dashboard."""
    main = (ROOT / "dashboard" / "main.py").read_text()
    assert "st.navigation(" in main and 'position="hidden"' in main
    assert "dashboard/main.py" in (ROOT / "dashboard" / "run.sh").read_text()


def test_filters_live_in_the_left_sidebar(app):
    """User, 2026-10-16: filters on the left so they can be changed from anywhere on the page."""
    for key in ("f_crop", "f_state", "f_district", "f_ref", "f_model"):
        assert app.sidebar.selectbox(key=key) is not None
    assert "Reset all filters" in [b.label for b in app.sidebar.button]


def test_reset_restores_defaults():
    at = st_testing.AppTest.from_file(str(APP), default_timeout=180)
    at.run()
    at.selectbox(key="f_crop").set_value("potatoes").run()
    at.selectbox(key="f_district").set_value("Uckermark").run()
    at.button[[b.label for b in at.button].index("Reset all filters")].click().run()
    assert at.selectbox(key="f_crop").value == "All crops"
    assert at.selectbox(key="f_district").value == "All districts"
    assert not at.exception


def test_numbers_are_computed_not_typed():
    """Counts on the Published models part come from papers.csv (they were once typed in by hand)."""
    src = APP.read_text()
    for literal in ('"5 of 8"', '"6 of 18"', "7.3% vs 8.2%"):
        assert literal not in src

"""YieldFormer-DE dashboard: one screen (contract: dashboard/SPEC.md part A, tests/test_dashboard_app.py).

Filter bar + five cards + one view at a time (Map | District look-up | Table & download | Forecasts | Compare).
Every number is read from results/dashboard/ (written by scripts/63_dashboard_data.py); nothing is typed in by hand.
Vocabulary (user, 2026-10-31): "backtest forecast" = 2010-2025, each season forecast by a model trained only on earlier
seasons; "sealed forecast" = 2026 onwards, saved before the season's official yields existed. In code `E`/"estimate".
Run: ./dashboard/run.sh (entry point dashboard/main.py, localhost only). Backup of the earlier app: run_old.sh.
"""
from __future__ import annotations

import html
import json
import re
from contextlib import nullcontext
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "dashboard"
SHAPEFILE = ROOT / "data" / "raw" / "geo" / "NUTS_RG_01M_2021_4326_LEVL_3.shp"
GEO_FILE = Path(__file__).with_name("static") / "de_nuts3.json"   # served by Streamlit (enableStaticServing)
GEO_URL = "app/static/de_nuts3.json"     # the browser loads the district shapes once instead of on every click

MODEL = "YieldFormer-DE (v18)"
MODELS = [MODEL, "YieldFormer-DE-T (v17)", "YieldFormer-DE-i (v15)", "YieldFormer-DE-P (v17)"]
MLABEL = {MODEL: "YieldFormer-DE (v18) · final",                   # short enough for the sidebar select
          "YieldFormer-DE-T (v17)": "YieldFormer-DE-T · TFT (v17)",
          "YieldFormer-DE-i (v15)": "YieldFormer-DE-i · iTransformer (v15)",
          "YieldFormer-DE-P (v17)": "YieldFormer-DE-P · PatchTST (v17)"}
COMPARE = ["5-year average", "Long-term average", "XGBoost + memory (reference)"]
CROPS = ["winter_wheat", "winter_barley", "rye", "triticale", "winter_rapeseed",
         "summer_barley", "oats", "silage_maize", "potatoes", "sugar_beet"]          # fixed order, never ranked
ORDER = {c: i for i, c in enumerate(CROPS)}
CROP_NAME = {"winter_wheat": "Winter wheat", "winter_barley": "Winter barley", "rye": "Rye",
             "triticale": "Triticale", "winter_rapeseed": "Winter rapeseed", "summer_barley": "Spring barley",
             "oats": "Oats", "silage_maize": "Silage maize", "potatoes": "Potatoes", "sugar_beet": "Sugar beet"}
REGIONS = ["North", "East", "South", "West"]                           # fixed order, never ranked
REGION_OF = {"Schleswig-Holstein": "North", "Lower Saxony": "North",
             "Mecklenburg-Vorpommern": "East", "Brandenburg": "East", "Saxony": "East", "Saxony-Anhalt": "East",
             "Thuringia": "East",
             "North Rhine-Westphalia": "West", "Hesse": "West", "Rhineland-Palatinate": "West", "Saarland": "West",
             "Bavaria": "South", "Baden-Württemberg": "South"}
REGION_HELP = ("Groups of federal states. North: Schleswig-Holstein, Niedersachsen. East: Mecklenburg-Vorpommern, "
               "Brandenburg, Sachsen, Sachsen-Anhalt, Thüringen. South: Bayern, Baden-Württemberg. West: "
               "Nordrhein-Westfalen, Hessen, Rheinland-Pfalz, Saarland.")
STATE_DE = {"Bavaria": "Bayern", "Hesse": "Hessen", "Lower Saxony": "Niedersachsen",
            "North Rhine-Westphalia": "Nordrhein-Westfalen", "Rhineland-Palatinate": "Rheinland-Pfalz",
            "Saxony": "Sachsen", "Saxony-Anhalt": "Sachsen-Anhalt", "Thuringia": "Thüringen"}
BLUE, LIGHT, GREY, LGREY, INK, RED, MUTED = "#2a78d6", "#9ec5f4", "#8f8d86", "#c9c7bf", "#1f1f1d", "#e34948", "#6b6a65"
SEQ_BLUE = [[0.0, "#e3eefc"], [0.5, "#5c9ce6"], [1.0, "#0d366b"]]      # higher = better (accuracy)
SEQ_GREY = [[0.0, "#f1f0eb"], [0.5, "#a9a69d"], [1.0, "#3b3a36"]]      # neutral quantities (dt/ha, area)
SEQ_RED = [[0.0, "#fbeaea"], [0.5, "#ec8f8e"], [1.0, "#a12a29"]]       # larger error = darker
DIV = [[0.0, RED], [0.5, "#f2f1ee"], [1.0, BLUE]]                      # below / worse = red, above / better = blue
CFG = {"displayModeBar": False, "scrollZoom": False}   # the mouse wheel scrolls the page, never zooms the map away

# Panel tints: each table / chart sits on its own light tint so the eye can tell sections apart (user, 2026-10-16).
# Tints are decoration only; data colours (blue / red / grey / black) keep their meaning on every tint.
TINTS = {"sky": ("#edf4fd", "#c6daf3"), "teal": ("#e8f6f2", "#bfe2d8"), "rose": ("#fbf0f5", "#ecccdc"),
         "violet": ("#f3f0fb", "#d8cff0"), "green": ("#ecf6e9", "#c8e1bd"), "slate": ("#f1f4f8", "#d2d9e2")}
ACCENT = {"accuracy": BLUE, "better": "#1f9e89", "within": "#7b61c9", "poor": "#c2557a", "forecast": "#3f9a4a"}


# Click spotlight. The handler is installed once into the page itself (not this iframe, which Streamlit may reload):
# a click on a panel or summary card lights it with a ring and darkens everything around it through one large
# box-shadow; a click on empty space in the main area or Esc clears it. Clicks in the sidebar never change it.
SPOTLIGHT_JS = """<script>
const P = window.parent, D = P.document;
P.__spotEnabled = __ON__;
if (!D.getElementById("spot-js")) {
  const s = D.createElement("script");
  s.id = "spot-js";
  s.textContent = "(" + function () {
    var cur = null;
    function styleEl() {
      var el = document.getElementById("spot-style");
      if (!el) { el = document.createElement("style"); el.id = "spot-style"; document.head.appendChild(el); }
      return el;
    }
    function keyOf(el) {
      if (el.classList.contains("kcard")) {
        var k = el.getAttribute("data-spot");
        return k ? '.kcard[data-spot="' + k + '"]' : null;
      }
      for (var i = 0; i < el.classList.length; i++) {
        if (el.classList[i].indexOf("st-key-pnl-") === 0) { return "." + el.classList[i]; }
      }
      return null;
    }
    window.__spotApply = function () {
      var el = styleEl();
      if (!window.__spotEnabled || !cur) { el.textContent = ""; return; }
      el.textContent = cur + "{position:relative;z-index:60;box-shadow:0 0 0 3px #2a78d6," +
                       "0 0 0 200vmax rgba(15,23,42,.45)!important;}";
    };
    document.addEventListener("click", function (e) {
      if (!window.__spotEnabled || !e.target.closest || !e.target.closest('[data-testid="stMain"]')) { return; }
      var t = e.target.closest('[class*="st-key-pnl-"], .kcard:not(.small)');
      var k = t ? keyOf(t) : null;
      if (k !== cur) { cur = k; window.__spotApply(); }
    }, true);
    document.addEventListener("keydown", function (e) {
      if (e.key === "Escape" && cur) { cur = null; window.__spotApply(); }
    });
  }.toString() + ")();";
  D.head.appendChild(s);
}
if (P.__spotApply) { P.__spotApply(); }
</script>"""


def panel(name: str, tint: str):
    """A tinted section; its CSS class (st-key-pnl-...) is also what the click spotlight lights up."""
    return st.container(key=f"pnl-{tint}-{name}")


st.set_page_config(page_title="YieldFormer-DE dashboard", page_icon="🌾", layout="wide",
                   initial_sidebar_state="expanded")
st.markdown("""
<style>
[data-testid="stSidebarNav"] {display: none;}
[data-testid="stAppViewContainer"], [data-testid="stMain"] {background: #f4f6f9;}
[data-testid="stHeader"] {background: #f4f6f9;}   /* solid, so the hosting toolbar never sits on top of a chart */
/* filter panel: fixed on the left, stays in view while the page scrolls */
section[data-testid="stSidebar"] {width: 310px !important; min-width: 310px !important; background: #e8f0f9;
                                  border-right: 1px solid #cddcee;}
section[data-testid="stSidebar"] [data-testid="stSidebarHeader"] {padding: 0.6rem 1rem 0 1rem; height: 2.2rem;}
section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {padding-top: 0.2rem;}
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {gap: 0.55rem;}
.ftitle {font-size: 0.78rem; color: #184f95; text-transform: uppercase; letter-spacing: .04em; font-weight: 700;}
.block-container {padding-top: 3.4rem; padding-bottom: 5rem; max-width: 1500px;}
.block-container [data-testid="stVerticalBlock"] {gap: 0.6rem;}
.band {background: linear-gradient(100deg, #12345a 0%, #1d5a9c 62%, #2a78d6 100%); border-radius: 14px;
       padding: 12px 196px 9px 20px; margin-bottom: 1rem; position: relative; min-height: 70px;}   /* offsets Streamlit's -1rem under markdown */
.band .logo {position: absolute; right: 16px; top: 50%; transform: translateY(-50%); height: 52px;
             background: #fff; border-radius: 8px; padding: 3px 6px;}   /* university logo from the final deck */
@media (max-width: 760px) {.band {padding-right: 20px;} .band .logo {display: none;}}
.band.nologo {padding-right: 20px;}
.title {font-size: 1.35rem; font-weight: 700; color: #fff; margin-right: 10px;}
.chip {display: inline-block; background: rgba(255,255,255,.16); color: #fff; border-radius: 12px; padding: 1px 10px;
       font-size: 0.78rem; margin: 0 6px 0 0; white-space: nowrap;}
.limits {color: #d3e2f4; font-size: 0.78rem; margin: 4px 0 0 0;}
.kcard {border: 1px solid #dfe3ea; border-radius: 10px; padding: 8px 12px; background: #fff; height: 114px;
        position: relative;
        box-shadow: 0 1px 2px rgba(20,40,70,.06);}
.kcard .t {font-size: 0.72rem; color: #5d6470; text-transform: uppercase; letter-spacing: .03em; white-space: nowrap;}
.kcard .v {font-size: clamp(1.3rem, 1.9vw, 1.85rem); font-weight: 700; color: #1f1f1d; line-height: 1.3; white-space: nowrap;}
.kcard .s {font-size: 0.76rem; color: #5d6470; line-height: 1.3; white-space: nowrap; overflow: hidden;
           text-overflow: ellipsis;}
.kcard .info {color: #2a78d6; cursor: help; margin-left: 3px;}
.kcard .tip {display: none; position: absolute; left: 6px; right: 6px; top: calc(100% + 4px); z-index: 100;
             background: #1f2937; color: #fff; font-size: 0.76rem; line-height: 1.35; padding: 7px 10px;
             border-radius: 8px; text-transform: none; letter-spacing: 0; white-space: normal;
             box-shadow: 0 6px 18px rgba(0,0,0,.22);}
.kcard .info:hover .tip {display: block;}
.kcard.small {height: 92px; margin-bottom: 1rem; box-shadow: none;}   /* offsets Streamlit's -1rem under markdown */
.kcard.small .v {font-size: 1.2rem;}
.kcard.hero {background: #f5f9fe;}
.gain {display: inline-block; border-radius: 10px; padding: 0 8px; font-size: 0.78rem; font-weight: 600;
       vertical-align: middle;}
.sub {color: #5d6470; font-size: 0.84rem;}
[class*="st-key-pnl-"] {border-radius: 12px; padding: 12px 14px 14px 14px; transition: box-shadow .2s;}
[class*="st-key-pnl-"] h5 {font-size: 0.98rem !important; font-weight: 650 !important; padding-bottom: 0.3rem;}
/* panels side by side end at the same height: the last panel in a column grows (Streamlit wraps each panel) */
/* only columns that hold a single panel (the map column keeps its natural height, so the map is never resized) */
[data-testid="stColumn"] > [data-testid="stVerticalBlock"]:has(> [data-testid="stLayoutWrapper"]:only-child > [class*="st-key-pnl-"]) {height: 100%;}
[data-testid="stColumn"] > [data-testid="stVerticalBlock"] > [data-testid="stLayoutWrapper"]:only-child:has(> [class*="st-key-pnl-"]) {flex-grow: 1;}
[data-testid="stColumn"] > [data-testid="stVerticalBlock"] > [data-testid="stLayoutWrapper"]:only-child > [class*="st-key-pnl-"] {height: 100%;}
[data-testid="stCaptionContainer"] p {max-width: 900px;}
html, [data-testid="stMain"] {scrollbar-gutter: stable;}
div[role="radiogroup"] {gap: 0.35rem; flex-wrap: wrap;}
div[role="radiogroup"] > label {border: 1px solid #d3d9e2; border-radius: 16px; padding: 1px 12px; background: #fff;
                                margin: 0;}
div[role="radiogroup"] > label p {font-size: 0.84rem;}   /* secondary pills smaller than the main view tabs */
div[role="radiogroup"] > label > div:first-child {display: none;}
div[role="radiogroup"] > label:has(input:checked) {border-color: #2a78d6; background: #2a78d6;}
div[role="radiogroup"] > label:has(input:checked) p {color: #fff;}
.st-key-view div[role="radiogroup"] > label {padding: 3px 16px;}
.st-key-view div[role="radiogroup"] > label p {font-size: 0.95rem; font-weight: 600;}
table.tbl {width: 100%; border-collapse: collapse; font-size: 0.82rem; margin-bottom: 6px;}
.tblwrap {overflow-x: auto;}
table.tbl th {text-align: left; color: #4f5662; font-weight: 600; border-bottom: 1px solid rgba(0,0,0,.14);
              padding: 5px 8px; background: rgba(255,255,255,.55); vertical-align: bottom;}
table.tbl td {border-bottom: 1px solid rgba(0,0,0,.06); padding: 4px 8px; white-space: nowrap;}
table.tbl tbody tr:hover td {background: rgba(255,255,255,.7);}
table.tbl td.n, table.tbl th.n {text-align: right; font-variant-numeric: tabular-nums;}
table.tbl td.n {white-space: nowrap;}
""" + "".join(f'[class*="st-key-pnl-{t}-"] {{background: {bg}; border: 1px solid {bd};}}\n'
              for t, (bg, bd) in TINTS.items()) + """
</style>""", unsafe_allow_html=True)


# --------------------------------------------------------------------------- data
@st.cache_data(show_spinner=False)
def load_estimates() -> pd.DataFrame:
    d = pd.read_parquet(DATA / "model_rows.parquet")
    keep = ["year", "crop", "landkreis", "district", "state", "actual"] + MODELS + COMPARE
    return d[[c for c in keep if c in d]]


@st.cache_data(show_spinner=False)
def load_csv(name: str) -> pd.DataFrame:
    f = DATA / name
    return pd.read_csv(f) if f.exists() else pd.DataFrame()


@st.cache_resource(show_spinner=False)
def load_geo():
    """Simplified district polygons (rounded coordinates) and the Germany outline.

    Built from the GISCO shapefile when it is there; otherwise (a fresh clone, the hosted dashboard) from the
    simplified copy kept in dashboard/static/, so the map works without downloading the raw boundaries."""
    from shapely.geometry import mapping, shape
    from shapely.ops import unary_union
    if not SHAPEFILE.exists() and GEO_FILE.exists():
        gj = json.loads(GEO_FILE.read_text())
        outline = unary_union([shape(f["geometry"]).buffer(0) for f in gj["features"]]).simplify(0.01)
        lines = [(list(np.round(p.exterior.coords.xy[0], 3)), list(np.round(p.exterior.coords.xy[1], 3)))
                 for p in getattr(outline, "geoms", [outline])]
        return gj, lines
    import geopandas as gpd
    g = gpd.read_file(SHAPEFILE)
    g = g[g.CNTR_CODE == "DE"][["NUTS_ID", "geometry"]].copy()
    g["geometry"] = g.geometry.simplify(0.004, preserve_topology=True)

    def rnd(geom):
        def r(c):
            return [r(x) for x in c] if isinstance(c[0], (list, tuple)) else [round(c[0], 3), round(c[1], 3)]
        m = mapping(geom)
        return {"type": m["type"], "coordinates": r(m["coordinates"])}
    feats = [{"type": "Feature", "id": row.NUTS_ID, "properties": {}, "geometry": rnd(row.geometry)}
             for row in g.itertuples()]
    outline = g.geometry.union_all().simplify(0.01)
    lines = [(list(np.round(p.exterior.coords.xy[0], 3)), list(np.round(p.exterior.coords.xy[1], 3)))
             for p in getattr(outline, "geoms", [outline])]
    gj = {"type": "FeatureCollection", "features": feats}
    GEO_FILE.parent.mkdir(exist_ok=True)
    GEO_FILE.write_text(json.dumps(gj, separators=(",", ":")))
    return gj, lines


@st.cache_data(show_spinner=False)
def sealed_files(stamp: float) -> pd.DataFrame:
    """Every file listed in FROZEN.sha256, its fingerprint recomputed now. `stamp` (newest mtime) refreshes the cache."""
    import hashlib
    rows = []
    for line in (SEALED / "FROZEN.sha256").read_text().splitlines():
        if not line.strip():
            continue
        want, name = line.split()
        f = SEALED / name
        got = hashlib.sha256(f.read_bytes()).hexdigest() if f.exists() else None
        yr, issued = re.search(r"forecast_(\d{4})", name), re.search(r"issued_(\d{4}-\d{2}-\d{2})", name)
        rows.append({"file": name, "year": int(yr.group(1)) if yr else None, "issued": issued.group(1) if issued else "",
                     "recorded": want, "check": "missing" if got is None else ("ok" if got == want else "changed")})
    return pd.DataFrame(rows).sort_values(["year", "issued"], ignore_index=True)


SEALED = ROOT / "results" / "forecasts"
E_ALL = load_estimates().assign(region=lambda x: x.state.map(REGION_OF))
F_ALL = load_csv("forecast_yieldformer.csv")
OFFICIAL = load_csv("official_sealed_years.csv")   # official yields of sealed years; empty until Destatis publishes
AREA = load_csv("crop_area.csv")
NAME_OF = F_ALL.dropna(subset=["district"]).drop_duplicates("landkreis").set_index("landkreis")["district"].to_dict()
NAME_OF.update(E_ALL.drop_duplicates("landkreis").set_index("landkreis")["district"].to_dict())
CODE_OF = {v: k for k, v in NAME_OF.items()}
STATE_OF = AREA.dropna(subset=["state"]).drop_duplicates("landkreis").set_index("landkreis")["state"].to_dict()
STATE_OF.update(E_ALL.drop_duplicates("landkreis").set_index("landkreis")["state"].to_dict())
# years come from the data, not from the code: estimates FIRST_YEAR..LAST_YEAR, sealed forecast FC_YEAR
FIRST_YEAR, LAST_YEAR = int(E_ALL.year.min()), int(E_ALL.year.max())
FC_YEAR = int(F_ALL.loc[F_ALL.model == MODEL, "year"].max())
MAP_YEARS = [FC_YEAR] + list(range(LAST_YEAR, FIRST_YEAR - 1, -1))   # sealed forecast first, then harvests
EARLY_END = FIRST_YEAR + 5   # the report's two test periods: the first six seasons, then the rest
DROUGHT_YEAR = 2018          # the drought season, left out once to check that a region's gain is not one year
F26_ALL = F_ALL[(F_ALL.year == FC_YEAR) & (F_ALL.model == MODEL)]
META = F26_ALL.iloc[0] if len(F26_ALL) else None
META_FILE = META.file if META is not None else ""


def cname(c: str) -> str:
    return CROP_NAME.get(c, c)


def sname(s) -> str:
    return STATE_DE.get(s, s) if isinstance(s, str) else ""


SHOW = {"Long-term average": "District normal"}   # one name on screen (reviewers, 2026-10-31); data column unchanged


def dn(r: str) -> str:
    return SHOW.get(r, r)


def rshort(r: str) -> str:
    r = dn(r)
    return r if "XGBoost" in r else r.lower()


def dfmt(d) -> str:
    try:
        return pd.Timestamp(d).strftime("%-d %b %Y")
    except (ValueError, TypeError):
        return str(d)


def mape(p, y) -> float:
    p, y = np.asarray(p, float), np.asarray(y, float)
    ok = np.isfinite(p) & np.isfinite(y) & (y > 0)
    return float(100 * np.mean(np.abs(p[ok] - y[ok]) / y[ok])) if ok.any() else float("nan")


def acc(p, y) -> float:
    return 100 - mape(p, y)


def acc_by(d: pd.DataFrame, col: str, by: str = "landkreis") -> pd.Series:
    """Accuracy per group without a Python loop (same definition as acc)."""
    ape = (100 * (d[col] - d.actual).abs() / d.actual).where(d.actual > 0)
    return 100 - ape.groupby(d[by]).mean()


def gain_range(d: pd.DataFrame, m: str, r: str, n: int = 10000) -> tuple:
    """90% range of (accuracy of m - accuracy of r) with whole seasons resampled (districts share a season's weather)."""
    d = d.dropna(subset=["actual", m, r])
    d = d[d.actual > 0]
    y = (d.assign(em=(d[m] - d.actual).abs() / d.actual * 100, er=(d[r] - d.actual).abs() / d.actual * 100)
         .groupby("year").agg(em=("em", "sum"), er=("er", "sum"), n=("em", "size")))
    if len(y) < 3:
        return float("nan"), float("nan")
    idx = np.random.default_rng(0).integers(0, len(y), size=(n, len(y)))
    g = (y.er.to_numpy()[idx].sum(1) - y.em.to_numpy()[idx].sum(1)) / y.n.to_numpy()[idx].sum(1)
    return tuple(np.percentile(g, [5, 95]))


def f1(x, d=1) -> str:
    """Number with d decimals, '–' if missing, real minus sign."""
    return "–" if x is None or x != x else f"{x:,.{d}f}".replace("-", "−")


def fs(x, d=1) -> str:
    """Signed number (+/−), '–' if missing; a value that rounds to zero is shown as 0.0, not −0.0."""
    if x is None or x != x:
        return "–"
    return f"{0:.{d}f}" if round(x, d) == 0 else f"{x:+.{d}f}".replace("-", "−")


def plural(n: int, word: str, many: str | None = None) -> str:
    return f"{n:,} {word if n == 1 else (many or word + 's')}"


def card(col, title, value, sub="", *, hero=False, chip=None, help_="", small=False, accent=None):
    e = html.escape
    chip_html = ""
    if chip is not None:
        txt, good = chip
        bg, fg = ("#e3eefc", "#184f95") if good else ("#fde8e8", "#a12a29")
        chip_html = f" <span class='gain' style='background:{bg};color:{fg}'>{e(txt)}</span>"
    subs = sub if isinstance(sub, (list, tuple)) else [sub]
    info = f"<span class='info'>ⓘ<span class='tip'>{e(help_)}</span></span>" if help_ else ""
    cls = "kcard" + (" hero" if hero else "") + (" small" if small else "")
    top = f" style='border-top:4px solid {accent}'" if accent else ""
    spot = e(re.sub(r"[^a-z0-9]+", "-", str(title).lower()), quote=True)
    col.markdown(f"<div class='{cls}' data-spot='{spot}'{top}>"
                 f"<div class='t'>{e(title)}{info}</div>"
                 f"<div class='v'>{e(str(value))}{chip_html}</div>"
                 + "".join(f"<div class='s'>{e(s)}</div>" for s in subs if s) + "</div>", unsafe_allow_html=True)


def table(df: pd.DataFrame, numeric: tuple = ()):
    """Plain HTML table: no internal scrolling, no header-click sorting; `numeric` columns right-aligned."""
    e = html.escape
    head = "".join(f"<th class='{'n' if c in numeric else ''}'>{e(str(c))}</th>" for c in df.columns)
    rows = "".join("<tr>" + "".join(f"<td class='{'n' if c in numeric else ''}'>{e(str(v))}</td>"
                                    for c, v in zip(df.columns, r)) + "</tr>" for r in df.itertuples(index=False))
    st.markdown(f"<div class='tblwrap'><table class='tbl'><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table>"
                "</div>",
                unsafe_allow_html=True)


def _step(key: str, d: int, pages: int):
    st.session_state[key] = int(min(max(1, st.session_state.get(key, 1) + d), pages))


def paged(df: pd.DataFrame, key: str, numeric: tuple = (), per: int = 15):
    """Fixed rows per page with ◀ page ▶ controls. Page keys start with 'pg_' and go back to 1 when filters change."""
    n = len(df)
    pages = max(1, int(np.ceil(n / per)))
    if st.session_state.get(key, 1) > pages:
        st.session_state[key] = pages
    p = int(st.session_state.get(key, 1))
    table(df.iloc[(p - 1) * per:p * per], numeric)
    if pages == 1:
        return
    a, b, c, d = st.columns([1, 0.8, 2, 1])
    a.button("◀ Previous", key=f"{key}_prev", on_click=_step, args=(key, -1, pages), disabled=p <= 1, width="stretch")
    b.number_input("Page", 1, pages, key=key, label_visibility="collapsed")
    c.markdown(f"<div class='sub' style='padding-top:8px'>of {pages:,} pages · {plural(n, 'row')}</div>",
               unsafe_allow_html=True)
    d.button("Next ▶", key=f"{key}_next", on_click=_step, args=(key, 1, pages), disabled=p >= pages, width="stretch")


LEVELS = ["District", "State", "Region", "Germany"]


def by_level(T: pd.DataFrame, cols: list, level: str) -> pd.DataFrame:
    """Rows per year, crop and area unit; yields averaged with each district's census crop area as weight (as the
    official state and national yields are), plain mean where no area is known. District level returns T as is."""
    if level == "District":
        return T.assign(unit=T.district, n=1)
    unit = {"State": T.state.map(sname), "Region": T.region, "Germany": pd.Series("Germany", index=T.index)}[level]
    d = T.assign(unit=unit).merge(AREA[["landkreis", "crop", "area_ha"]], on=["landkreis", "crop"], how="left")
    d["w"] = d.area_ha.fillna(0.0)
    keys = ["year", "crop", "unit"]
    g = d.groupby(keys)
    out = g.size().rename("n").to_frame()
    wsum = g.w.sum()
    for c in cols:
        ok = d[c].notna()
        wx = (d[c] * d.w).where(ok).groupby([d[k] for k in keys]).sum()
        out[c] = (wx / wsum.where(wsum > 0)).fillna(g[c].mean())
    return out.reset_index()


def search_box(key: str, label: str = "Search district or state") -> str:
    return st.text_input(label, key=key, placeholder="type part of a name").strip().lower()


def layout(fig, h=340, **kw):
    fig.update_layout(height=h, margin=kw.pop("margin", dict(l=8, r=8, t=34, b=8)), plot_bgcolor="rgba(0,0,0,0)",
                      paper_bgcolor="rgba(0,0,0,0)",
                      font=dict(size=12, color=INK), legend=dict(orientation="h", y=-0.2, x=0, font=dict(size=11)),
                      hoverlabel=dict(font_size=12), **kw)
    fig.update_xaxes(gridcolor="rgba(0,0,0,.08)", zeroline=False, tickfont=dict(size=11, color="#555"))
    fig.update_yaxes(gridcolor="rgba(0,0,0,.08)", zeroline=False, tickfont=dict(size=11, color="#555"))
    return fig


def csv_bytes(df: pd.DataFrame) -> bytes:
    if st.session_state.get("csv_de"):
        return df.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig")
    return df.to_csv(index=False).encode("utf-8-sig")


# --------------------------------------------------------------------------- filter state and callbacks
ss = st.session_state
DEFAULTS = {"f_crop": "All crops", "f_years": (FIRST_YEAR, LAST_YEAR), "f_region": "All regions", "f_state": "All states", "f_district": "All districts",
            "f_ref": "5-year average", "f_model": MODEL, "map_layer": "vs_normal", "map_year": FC_YEAR,
            "cmp_part": "Simple averages and members"}
for k_, v_ in DEFAULTS.items():
    if k_ != "f_years":                    # the slider gets its range from `value=`; Reset and year clicks set it
        ss.setdefault(k_, v_)


def _reset():
    for k, v in DEFAULTS.items():
        ss[k] = v
    for k in [k for k in ss.keys() if str(k).startswith("pg_")]:
        ss[k] = 1


def _points(key: str) -> list:
    ev = ss.get(key)
    try:
        return list(ev["selection"]["points"])
    except (KeyError, TypeError, AttributeError):
        return []


def _map_click():
    pts = _points("map_main")
    loc = pts[0].get("location") if pts else None
    if loc in NAME_OF and (E_ALL.landkreis == loc).any():
        ss["f_district"] = NAME_OF[loc]
        if ss.get("f_state") != "All states" and STATE_OF.get(loc) != ss.get("f_state"):
            ss["f_state"] = "All states"
        if ss.get("f_region") != "All regions" and REGION_OF.get(STATE_OF.get(loc)) != ss.get("f_region"):
            ss["f_region"] = "All regions"


def _year_click(key: str):
    pts = _points(key)
    x = pts[0].get("x") if pts else None
    if x is not None and FIRST_YEAR <= int(x) <= LAST_YEAR:
        ss["f_years"] = (int(x), int(x))


def _set_from(target: str, key: str):
    ss[target] = ss[key]


# --------------------------------------------------------------------------- header + filter panel (left sidebar)
sealed = f"2026 forecast sealed {dfmt(META.issued)} · data to {dfmt(META.data_end)}" if META is not None else ""
LOGO = ("<img class='logo' src='app/static/jmu_logo.png' alt='Universität Würzburg'>"   # local only: the logo
        if (GEO_FILE.parent / "jmu_logo.png").exists() else "")                          # is not in the public repo
BAND = "band" if LOGO else "band nologo"
st.markdown(f"<div class='{BAND}'>{LOGO}"
            f"<span class='title'>YieldFormer-DE · district crop yields in Germany</span>"
            f"<span class='chip'>backtest forecasts 2010–2025: each season from a model that never saw it</span>"
            f"<span class='chip'>{html.escape(sealed)}</span>"
            f"<div class='limits'>Limits: made before the official statistics, not before the season · no better than the "
            f"5-year average at 30 June · underestimates large losses in single districts</div></div>", unsafe_allow_html=True)

sb = st.sidebar
# How to read the page (user, 2026-11-01): one click away, above the filters (UI review, 2026-11-01).
with sb.popover("ⓘ How to read this dashboard", width="stretch"):
    st.markdown(
        f"**The five cards**\n"
        f"- **Accuracy:** 100 minus the mean % error of the forecasts (not a hit rate), next to the comparison; "
        f"the chip is the gain in percentage points, the 90% range comes from resampling whole seasons.\n"
        f"- **Better in:** seasons in which the forecast was more accurate than the comparison.\n"
        f"- **Within ±10%:** share of forecasts within 10% of the official yield.\n"
        f"- **Poor crop-years:** crop-years at least 10% below normal, and how many the forecast flagged.\n"
        f"- **{FC_YEAR} forecast:** the sealed forecast against the district normal.\n\n"
        f"**Backtest and sealed forecasts**\n"
        f"- {FIRST_YEAR}–{LAST_YEAR}: backtest forecasts, each season from a model trained only on earlier seasons.\n"
        f"- {FC_YEAR}: sealed before the official figures; not scored until Destatis publishes the district yields.\n\n"
        f"**How the numbers are averaged**\n"
        f"- Cards 1–4: a plain mean over every district-crop-year with an official yield in the filters.\n"
        f"- {FC_YEAR} card: weighted by each crop's area.\n\n"
        f"**Colours**\n"
        f"- Blue: our forecast, above normal, better. Red: below normal, worse. Grey: comparison. Black: official.\n\n"
        f"**Limits**\n"
        f"- The design (inputs, networks, settings; versions v1–v20) was chosen while these results were visible; "
        f"a change was kept only if it passed a rule written in advance and tested on {FIRST_YEAR}–{EARLY_END}.\n"
        f"- Uses weather up to harvest: it comes before the official statistics, not before the season.\n"
        f"- Large losses in single districts are underestimated.\n\n"
        f"Each card's own ⓘ explains that card in detail.")
sb.markdown("<div class='ftitle' style='margin-top:14px'>Filters</div><div class='sub' style='font-size:0.78rem;"
            "margin:2px 0 12px 0'>apply to every card and view</div>", unsafe_allow_html=True)
crop = sb.selectbox("Crop", ["All crops"] + CROPS, format_func=lambda c: c if c == "All crops" else cname(c),
                    key="f_crop")
years = sb.select_slider("Years (backtest forecasts)", options=list(range(FIRST_YEAR, LAST_YEAR + 1)), value=DEFAULTS["f_years"],
                         key="f_years")
region = sb.selectbox("Region", ["All regions"] + REGIONS, key="f_region", help=REGION_HELP)
states = sorted((s_ for s_ in E_ALL.state.dropna().unique() if region == "All regions" or REGION_OF.get(s_) == region),
                key=sname)
if ss.get("f_state") not in ["All states"] + states:
    ss["f_state"] = "All states"
state = sb.selectbox("State", ["All states"] + states, format_func=lambda s: s if s == "All states" else sname(s),
                     key="f_state")
pool = (E_ALL[E_ALL.state == state] if state != "All states" else
        E_ALL[E_ALL.state.map(REGION_OF) == region] if region != "All regions" else E_ALL)
dnames = ["All districts"] + sorted(pool.district.dropna().unique())
if ss.get("f_district") not in dnames:
    ss["f_district"] = "All districts"
district = sb.selectbox("District (type to search)", dnames, key="f_district")
ref = sb.selectbox("Compare with", COMPARE, key="f_ref", format_func=dn,
                   help="5-year average: the district's mean yield of the five years before. District normal: the mean of "
                        "all earlier years, plus a trend if one is clear (the same normal is used everywhere on this page). "
                        "XGBoost + memory: a tree model on the same inputs, shown as a reference only.")
model = sb.selectbox("Model (backtest forecasts)", MODELS, format_func=MLABEL.get, key="f_model",
                     help=f"{MODEL} is the final model: the average of three transformer networks (TFT, iTransformer, "
                          "PatchTST) plus a district error memory. The members are each network with the memory. "
                          "The 2026 forecast is always the final model. District error memory: adds back each "
                          "district's average error over the earlier seasons. vN = version number of the project's "
                          "written plans (v1 to v20); each version was planned in writing before it was run.")
sb.button("Reset all filters", on_click=_reset, width="stretch", help="All filters back to their defaults")
sb.divider()
spot_on = sb.toggle("Spotlight", value=True, key="spot_on",
                    help="Click a card, table or chart to light it up and dim the rest of the page. Click empty space "
                         "or press Esc to switch the light back on everywhere.")
sb.caption("Spotlight: click a panel to focus on it; Esc or a click on empty space clears it.")
with sb:
    components.html(SPOTLIGHT_JS.replace("__ON__", "true" if spot_on else "false"), height=0)

RS = {"5-year average": "5-yr avg", "Long-term average": "Normal",
      "XGBoost + memory (reference)": "XGBoost ref."}.get(ref, ref)   # short form for table headers
fp = (crop, tuple(years), region, state, district, ref, model)                 # back to page 1 when the filters change
if ss.get("_fp") != fp:
    for k_ in [k for k in ss.keys() if str(k).startswith("pg_")]:
        ss[k_] = 1
    ss["_fp"] = fp


def select(df: pd.DataFrame, *, use_years=True, use_district=True, use_crop=True, use_area=True) -> pd.DataFrame:
    m = pd.Series(True, index=df.index)
    if use_crop and crop != "All crops":
        m &= df.crop == crop
    st_ = df.state if "state" in df else df.landkreis.map(STATE_OF)
    if use_area and region != "All regions":
        m &= st_.map(REGION_OF).eq(region)
    if use_area and state != "All states":
        m &= st_.eq(state)
    if use_district and district != "All districts":
        m &= df.landkreis.eq(CODE_OF.get(district))
    if use_years and "year" in df:
        m &= df.year.between(*years)
    return df[m]


E = select(E_ALL)
F26 = select(F26_ALL, use_years=False)
E_MAP = select(E_ALL, use_district=False, use_area=False)          # national: the map keeps one colour scale and
F_MAP = select(F26_ALL, use_years=False, use_district=False, use_area=False)   # only draws the districts in scope
scope = (district if district != "All districts" else sname(state) if state != "All states"
         else f"{region} region" if region != "All regions" else "Germany")
yr_txt = f"{years[0]}" if years[0] == years[1] else f"{years[0]}–{years[1]}"
if E.empty:
    st.warning("No official yields for this selection (perhaps the crop is not grown here). Change the filters or "
               "press Reset all filters.")
    st.stop()


def area_weighted(f: pd.DataFrame, col: str = "vs_normal") -> pd.Series:
    """Per district: a crop-level % (vs normal, error), weighted by each crop's census area in the district."""
    d = f.dropna(subset=[col]).merge(AREA[["landkreis", "crop", "area_ha"]], on=["landkreis", "crop"], how="left")
    d["w"] = d.area_ha.fillna(0.0)
    d["wx"] = d[col] * d.w
    g = d.groupby("landkreis")
    w = g.w.sum()
    return (g.wx.sum() / w.where(w > 0)).fillna(g[col].mean())     # plain mean where no census area is known


def poor_years(d: pd.DataFrame, col: str):
    """Poor crop-year = mean official yield ≥10% below the long-term normal; detected = estimate ≥5% below normal."""
    g = d.groupby(["crop", "year"])[["actual", col, "Long-term average"]].mean()
    real = 100 * (g.actual / g["Long-term average"] - 1)
    est = 100 * (g[col] / g["Long-term average"] - 1)
    return real, est


# --------------------------------------------------------------------------- cards
m_mod, m_ref = mape(E[model], E.actual), mape(E[ref], E.actual)
gain = m_ref - m_mod
yr = acc_by(E, model, "year") - acc_by(E, ref, "year")
k = st.columns(5)
RN = dn(ref).replace(" (reference)", "")                                     # comparison name on the cards
lo_, hi_ = gain_range(E, model, ref)
card(k[0], "Accuracy", f"{100 - m_mod:.1f}%",
     [f"{RN} {100 - m_ref:.1f}%", f"90% range {fs(lo_)} to {fs(hi_)} pp" if lo_ == lo_ else "range: too few seasons"],
     hero=True, chip=(f"{fs(gain)} pp", gain >= 0), accent=ACCENT["accuracy"],
     help_="Accuracy = 100 minus the mean absolute % error of the district backtest forecasts (not a hit rate). "
           "pp = percentage points. 90% range of the gain: whole seasons resampled 10,000 times, because districts "
           "in one season share the weather.")
card(k[1], "Better in", f"{int((yr > 0).sum())} of {len(yr)}", f"{'season' if len(yr) == 1 else 'seasons'} vs {RN}",
     accent=ACCENT["better"],
     help_="Seasons (harvest years) in which the backtest forecast was more accurate than the comparison.")
w_mod = 100 * np.mean(np.abs(E[model] / E.actual - 1) <= 0.10)
w_ref = 100 * np.mean(np.abs(E[ref] / E.actual - 1) <= 0.10)
card(k[2], "Within ±10%", f"{w_mod:.0f}%", f"{RN} {w_ref:.0f}%", accent=ACCENT["within"],
     help_="Share of backtest forecasts within 10% of the official yield.")
real, est = poor_years(E, model)
poor, sig = real <= -10, est <= -5
n_poor, hit, false = int(poor.sum()), int((poor & sig).sum()), int((~poor & sig).sum())
near = int((~poor & sig & (real <= -5)).sum())
card(k[3], "Poor crop-years", f"{hit} of {n_poor}" if n_poor else "–", f"detected · {plural(false, 'false alarm')}",
     accent=ACCENT["poor"],
     help_="Counted per crop-year (one crop in one season, averaged over the districts in the selection). Poor: "
           "yield ≥10% below normal. Detected: backtest forecast ≥5% below normal; the forecast uses weather up "
           "to harvest."
           + (f" {near} of the false alarms were years 5–10% below normal." if near else ""))
if F26.empty:
    card(k[4], "2026 forecast", "–", "none for this selection", accent=ACCENT["forecast"])
elif crop != "All crops":
    vs = 100 * (F26.y_pred.mean() / F26.y_hist.mean() - 1)
    card(k[4], "2026 forecast", f"{F26.y_pred.mean():.1f} dt/ha", f"{fs(vs)}% vs normal", accent=ACCENT["forecast"],
         help_=f"Mean of the districts. Typical range {F26.lo80.mean():.1f}–{F26.hi80.mean():.1f} dt/ha.")
else:
    nat = F26.groupby("crop")[["y_pred", "y_hist"]].mean()
    fa = F26.dropna(subset=["vs_normal"]).merge(AREA[["landkreis", "crop", "area_ha"]], on=["landkreis", "crop"],
                                                how="left")       # all crops: weighted by each crop's census area
    wa = fa.area_ha.fillna(0.0)
    tot = float(np.average(fa.vs_normal, weights=wa)) if wa.sum() > 0 else float(fa.vs_normal.mean())
    card(k[4], "2026 forecast", f"{fs(tot)}%", "vs normal", accent=ACCENT["forecast"],
         help_=f"All crops, weighted by crop area. {int((nat.y_pred > nat.y_hist).sum())} of {len(nat)} crops above normal.")
st.markdown(f"<div class='sub' style='margin:10px 0 2px 0'>Selection: "
            f"{cname(crop) if crop != 'All crops' else 'all crops'} · {html.escape(scope)} · {yr_txt} · "
            f"{html.escape(MLABEL[model])} vs {html.escape(dn(ref))} · {plural(E.landkreis.nunique(), 'district')} · "
            f"{plural(len(E), 'backtest forecast')}</div>"
            f"<div class='sub' style='margin:0 0 2px 0'>Cards 1–4: backtest forecasts {FIRST_YEAR}–{LAST_YEAR} · "
            f"the sealed {FC_YEAR} forecast is not scored yet · <b>ⓘ How to read this dashboard</b> in the filter "
            f"panel</div><div style='height:8px'></div>",
            unsafe_allow_html=True)

VIEWS = ["Map", "District look-up", "Table & download", "Forecasts", "Compare"]
view = st.radio("View", VIEWS, horizontal=True, key="view", label_visibility="collapsed")


# --------------------------------------------------------------------------- figures
def district_figure(values: pd.Series, label: str, scale, *, diverging=False, clip=False, hover=None,
                    unit="", height=500):
    gj, outline = load_geo()
    v_all = values.dropna()                       # Germany: fixes the colour scale, so a colour means the same
    st_of = v_all.index.map(STATE_OF)             # value whichever region or state is shown
    keep = np.ones(len(v_all), bool)
    if region != "All regions":
        keep &= st_of.map(REGION_OF) == region
    if state != "All states":
        keep &= st_of == state
    v = v_all[keep]
    fig = go.Figure()
    zoom = (state != "All states" or region != "All regions") and len(v)
    if not zoom:                                  # one trace for mainland and islands (None breaks the line)
        lon = [c for xs, _ in outline for c in list(xs) + [None]]
        lat = [c for _, ys in outline for c in list(ys) + [None]]
        fig.add_trace(go.Scattergeo(lon=lon, lat=lat, mode="lines", line=dict(color=LGREY, width=0.8),
                                    hoverinfo="skip", showlegend=False))
    if len(v):
        kw = {}
        if diverging:
            lim = float(np.nanpercentile(np.abs(v_all), 97)) or 1.0
            kw.update(zmid=0, zmin=-lim, zmax=lim)
        elif clip:
            kw.update(zmin=float(np.nanpercentile(v_all, 2)), zmax=float(np.nanpercentile(v_all, 98)))
        else:
            kw.update(zmin=float(v_all.min()), zmax=float(v_all.max()))
        text = [hover(i, x) if hover else
                f"<b>{html.escape(NAME_OF.get(i, i))}</b> · {sname(STATE_OF.get(i))}<br>{label}: "
                f"{fs(x) if diverging else f1(x)}{unit}" for i, x in v.items()]
        fig.add_trace(go.Choropleth(
            geojson=GEO_URL, locations=v.index, z=v.values, colorscale=scale, marker_line_width=0.25,
            marker_line_color="white", text=text, hovertemplate="%{text}<extra></extra>",
            colorbar=dict(title=dict(text=label, side="top", font=dict(size=11)), orientation="h", thickness=9,
                          len=0.55, x=0.5, y=0.0, xanchor="center", yanchor="top", ticksuffix=unit,
                          tickfont=dict(size=10)), **kw))
    if district != "All districts":
        code = CODE_OF.get(district)
        sel = [f for f in gj["features"] if f["id"] == code]
        if sel:
            fig.add_trace(go.Choropleth(geojson={"type": "FeatureCollection", "features": sel}, locations=[code],
                                        z=[1], colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]],
                                        marker_line_color=INK, marker_line_width=2.5, showscale=False,
                                        hoverinfo="skip"))
    if zoom:
        fig.update_geos(fitbounds="locations", visible=False, projection_type="mercator")
    else:
        fig.update_geos(visible=False, projection_type="mercator", lonaxis_range=[5.8, 15.1],
                        lataxis_range=[47.3, 55.0])
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=0, b=46), paper_bgcolor="rgba(0,0,0,0)",
                      geo_bgcolor="rgba(0,0,0,0)")
    return fig


def selection_trend(d: pd.DataFrame, f26: pd.DataFrame, title: str, h=320):
    """Official vs estimate vs comparison per year (mean over districts) + the 2026 forecast."""
    g = d.groupby("year")[["actual", model, ref]].mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=g.index, y=g.actual, name="Official yield", mode="lines+markers",
                             line=dict(color=INK, width=2), marker=dict(size=7),
                             hovertemplate="%{x}: %{y:.1f} dt/ha<extra>official</extra>"))
    fig.add_trace(go.Scatter(x=g.index, y=g[model], name="Forecast (backtest)", mode="lines+markers",
                             line=dict(color=BLUE, width=2), marker=dict(size=7),
                             hovertemplate="%{x}: %{y:.1f} dt/ha<extra>backtest forecast</extra>"))
    fig.add_trace(go.Scatter(x=g.index, y=g[ref], name=dn(ref), mode="lines", line=dict(color=GREY, width=2, dash="dot"),
                             hovertemplate="%{x}: %{y:.1f} dt/ha<extra>" + ref + "</extra>"))
    if not f26.empty:
        y0 = f26.y_pred.mean()
        fig.add_trace(go.Scatter(x=[2026], y=[y0], name="2026 forecast", mode="markers",
                                 marker=dict(color=BLUE, size=12, symbol="diamond"),
                                 error_y=dict(type="data", symmetric=False, color=LIGHT, thickness=3,
                                              array=[f26.hi80.mean() - y0], arrayminus=[y0 - f26.lo80.mean()]),
                                 hovertemplate="2026 forecast: %{y:.1f} dt/ha<extra></extra>"))
    layout(fig, h, title=dict(text=title, font=dict(size=14), x=0), yaxis_title="dt/ha",
           xaxis=dict(dtick=2, tickformat="d"))
    return fig


def swing_trend(d: pd.DataFrame, f26: pd.DataFrame, title: str, h=330):
    """Season by season, in % vs the district normal so that crops can be pooled: official, backtest forecast,
    comparison (plain mean over the district-crop-years) and the sealed forecast with its 80% range."""
    x = d[d["Long-term average"] > 0]
    pct = lambda c: 100 * (x[c] / x["Long-term average"] - 1)          # noqa: E731
    g = pd.DataFrame({"off": pct("actual"), "fc": pct(model), "ref": pct(ref), "year": x.year}).groupby("year").mean()
    fig = go.Figure()
    fig.add_hline(y=0, line_color=LGREY, line_width=1)
    fig.add_trace(go.Scatter(x=g.index, y=g.off, name="Official yield", mode="lines+markers",
                             line=dict(color=INK, width=2), marker=dict(size=7),
                             hovertemplate="%{x}: %{y:+.1f}% vs normal<extra>official</extra>"))
    fig.add_trace(go.Scatter(x=g.index, y=g.fc, name="Forecast (backtest)", mode="lines+markers",
                             line=dict(color=BLUE, width=2), marker=dict(size=7),
                             hovertemplate="%{x}: %{y:+.1f}% vs normal<extra>backtest forecast</extra>"))
    if ref != "Long-term average":
        fig.add_trace(go.Scatter(x=g.index, y=g.ref, name=dn(ref), mode="lines",
                                 line=dict(color=GREY, width=2, dash="dot"),
                                 hovertemplate="%{x}: %{y:+.1f}% vs normal<extra>" + dn(ref) + "</extra>"))
    if not f26.empty:
        v = f26[f26.y_hist > 0]
        y0 = float((100 * (v.y_pred / v.y_hist - 1)).mean())
        lo, hi = float((100 * (v.lo80 / v.y_hist - 1)).mean()), float((100 * (v.hi80 / v.y_hist - 1)).mean())
        fig.add_trace(go.Scatter(x=[FC_YEAR], y=[y0], name=f"{FC_YEAR} sealed forecast", mode="markers",
                                 marker=dict(color=BLUE, size=12, symbol="diamond"),
                                 error_y=dict(type="data", symmetric=False, color=LIGHT, thickness=3,
                                              array=[hi - y0], arrayminus=[y0 - lo]),
                                 hovertemplate=f"{FC_YEAR} sealed forecast: %{{y:+.1f}}% vs normal (80% range "
                                               f"{lo:+.1f} to {hi:+.1f}%)<extra></extra>"))
    layout(fig, h, title=dict(text=title, font=dict(size=14), x=0), yaxis=dict(ticksuffix="%"),
           xaxis=dict(dtick=2, tickformat="d"))
    return fig


def accuracy_lines(d: pd.DataFrame, title: str, h=330):
    """Accuracy per season: the model, the chosen comparison and (if it is not the comparison) the district normal."""
    fig = go.Figure()
    lines = [(model, MLABEL[model].split(" · ")[0], BLUE, "solid"), (ref, dn(ref), GREY, "dot")]
    if ref != "Long-term average":
        lines.append(("Long-term average", dn("Long-term average"), LGREY, "dash"))
    for col, nm, colr, dash in lines:
        a = acc_by(d, col, "year")
        fig.add_trace(go.Scatter(x=a.index, y=a, name=nm, mode="lines+markers" if col == model else "lines",
                                 line=dict(color=colr, width=2.5 if col == model else 2, dash=dash),
                                 marker=dict(size=7),
                                 hovertemplate="%{x}: %{y:.1f}%<extra>" + html.escape(nm) + "</extra>"))
    layout(fig, h, title=dict(text=title, font=dict(size=14), x=0), yaxis=dict(ticksuffix="%"),
           xaxis=dict(dtick=2, tickformat="d"))
    return fig


def date_chart(title: str, h=300):
    """Accuracy by forecast date: in-season runs with data cut 30 June / 31 July, then the harvest-time model."""
    one = crop != "All crops"
    T = load_csv("inseason_by_crop.csv" if one else "inseason.csv")
    if T.empty:
        return None, []
    T = T[T.crop == crop] if one else T[T.window.astype(str).str.startswith("all")]   # all crops: whole period
    if T.empty:
        return None, []
    cuts = {"06-30": "30 June", "07-31": "31 July"}
    xs = [cuts.get(c, c) for c in T.cut] + ["Harvest"]
    ys = list(T.acc_XH) + [float(T.acc_XE18.iloc[0])]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=ys, name="Our forecast", mode="lines+markers+text",
                             line=dict(color=BLUE, width=2.5), marker=dict(size=9), text=[f"{y:.1f}%" for y in ys],
                             textposition="top center", hovertemplate="%{x}: %{y:.1f}%<extra></extra>"))
    fig.add_trace(go.Scatter(x=xs, y=[float(T.acc_b5.iloc[0])] * len(xs), name="5-year average", mode="lines",
                             line=dict(color=GREY, width=2, dash="dot"),
                             hovertemplate="5-year average: %{y:.1f}%<extra></extra>"))
    b5 = float(T.acc_b5.iloc[0])
    lo, hi = min(ys + [b5]), max(ys + [b5])
    layout(fig, h, title=dict(text=title, font=dict(size=14), x=0),
           yaxis=dict(ticksuffix="%", range=[lo - 1.0, hi + 0.8]))
    fig.update_traces(cliponaxis=False)
    return fig, [y - b5 for y in ys]                     # and the gap to the 5-year average per date


def gains(d: pd.DataFrame, by: str, boot: int = 500) -> pd.DataFrame:
    """Per group: accuracy of the model and of the comparison, gain (pp), seasons with a gain, and a 90% range from
    resampling whole seasons (a season's weather is shared by all districts, so seasons are the unit of chance)."""
    x = d[d.actual > 0]
    x = x.assign(em=100 * (x[model] - x.actual).abs() / x.actual, er=100 * (x[ref] - x.actual).abs() / x.actual)
    t = x.groupby([by] if by == "year" else [by, "year"]).agg(em=("em", "sum"), er=("er", "sum"), n=("em", "size"))
    rng = np.random.default_rng(0)
    rows = []
    for g, u in t.groupby(level=0):
        em, er, n = u.em.to_numpy(), u.er.to_numpy(), u.n.to_numpy()
        lo = hi = np.nan
        if boot and len(n) >= 4:
            i = rng.integers(0, len(n), (boot, len(n)))
            lo, hi = np.percentile((er[i].sum(1) - em[i].sum(1)) / n[i].sum(1), [5, 95])
        rows.append({by: g, "acc_m": 100 - em.sum() / n.sum(), "acc_r": 100 - er.sum() / n.sum(),
                     "gain": (er.sum() - em.sum()) / n.sum(), "lo": lo, "hi": hi,
                     "better": int(((er - em) > 0).sum()), "seasons": len(n), "rows": int(n.sum())})
    return pd.DataFrame(rows).set_index(by) if rows else pd.DataFrame()


def hbars(val, lo, hi, ylab, text, title: str, *, opacity=None, customdata=None, hover="", unit="", h=320):
    """Horizontal bars from zero (blue >= 0, red < 0) in the given order. The range line and the value label are
    separate layers, so the label never sits on the line and a click on a bar always reaches the bar."""
    val, lo, hi = (np.asarray(x, float) for x in (val, lo, hi))
    span = float(np.nanmax(np.abs(np.r_[val, lo, hi, 1.0])))
    pad = 0.03 * span
    fig = go.Figure(go.Bar(x=val, y=ylab, orientation="h", marker=dict(color=[BLUE if v >= 0 else RED for v in val],
                                                                       opacity=opacity),
                           customdata=customdata, hovertemplate=hover))
    wx, wy = [], []
    for l_, h_, y_ in zip(lo, hi, ylab):
        if np.isfinite(l_) and np.isfinite(h_):
            wx += [l_, h_, None]
            wy += [y_, y_, None]
    if wx:
        fig.add_trace(go.Scatter(x=wx, y=wy, mode="lines", line=dict(color="rgba(93,100,112,0.45)", width=1.3),
                             hoverinfo="skip"))
    pos = val >= 0
    edge = np.where(pos, np.fmax(val, np.nan_to_num(hi, nan=-np.inf)), np.fmin(val, np.nan_to_num(lo, nan=np.inf)))
    fig.add_trace(go.Scatter(x=np.where(pos, edge + pad, edge - pad), y=ylab, mode="text", text=text,
                             textposition=["middle right" if q else "middle left" for q in pos],
                             textfont=dict(size=11, color=INK), hoverinfo="skip", cliponaxis=False))
    fig.add_vline(x=0, line_color=GREY, line_width=1)
    left = min(0.0, float(np.nanmin(np.r_[val, lo, 0.0])))
    right = max(0.0, float(np.nanmax(np.r_[val, hi, 0.0])))
    layout(fig, h, title=dict(text=title, font=dict(size=14), x=0), yaxis=dict(autorange="reversed"),
           xaxis=dict(ticksuffix=unit, range=[left - (0.25 * span if left < 0 else 0.05 * span), right + 0.3 * span]),
           showlegend=False, bargap=0.35)
    return fig


def gain_bars(t: pd.DataFrame, order: list, name, title: str, sel=None, h=320):
    """Gain of the model over the comparison (pp) per group, fixed order (never sorted by value); thin line = 90%
    range from resampling seasons."""
    t = t.reindex([o for o in order if o in t.index])
    return hbars(t.gain, t.lo, t.hi, [name(o) for o in t.index], [f"{fs(g)} pp" for g in t.gain], title,
                 opacity=[1.0 if sel in (None, o) else 0.3 for o in t.index],
                 customdata=np.c_[t.acc_m, t.acc_r, t.better, t.seasons],
                 hover="%{y}: %{x:+.1f} pp<br>accuracy %{customdata[0]:.1f}% vs %{customdata[1]:.1f}%<br>"
                       "better in %{customdata[2]:.0f} of %{customdata[3]:.0f} seasons<extra></extra>",
                 unit=" pp", h=h)


def season_columns(d: pd.DataFrame, title: str, h=300):
    """One column per season: gain of the model over the comparison (pp)."""
    t = gains(d, "year", boot=0)
    fig = go.Figure(go.Bar(
        x=t.index, y=t.gain, marker_color=[BLUE if g >= 0 else RED for g in t.gain], text=[fs(g) for g in t.gain],
        textposition="outside", cliponaxis=False, textfont=dict(size=11), customdata=np.c_[t.acc_m, t.acc_r],
        hovertemplate="%{x}: %{y:+.1f} pp<br>accuracy %{customdata[0]:.1f}% vs %{customdata[1]:.1f}%<extra></extra>"))
    fig.add_hline(y=0, line_color=GREY, line_width=1)
    layout(fig, h, title=dict(text=title, font=dict(size=14), x=0), yaxis=dict(ticksuffix=" pp"),
           xaxis=dict(dtick=2, tickformat="d"), showlegend=False, bargap=0.3)
    return fig


def region_headline(R: pd.DataFrame) -> str:
    """Both answers to "which region does better": the most accurate one and the one where the model adds most."""
    a, g = R.acc_m.idxmax(), R.gain.idxmax()
    return (f"Most accurate: <b>{a}</b> ({R.acc_m[a]:.1f}%) · Gains most over the {rshort(ref)}: <b>{g}</b> "
            f"({fs(R.gain[g])} pp)")


def region_figure(R: pd.DataFrame, sel=None, h=250):
    """Two answers side by side, regions in fixed order N, E, S, W: accuracy (dots: values are close, so a zero-based
    bar would hide the differences) and gain over the comparison (bars from zero, thin line = 90% range)."""
    R = R.reindex([r for r in REGIONS if r in R.index])
    y = list(R.index)
    op = [1.0 if sel in (None, r) else 0.3 for r in y]
    top_a, top_g = R.acc_m.idxmax(), R.gain.idxmax()
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, column_widths=[0.42, 0.58], horizontal_spacing=0.07,
                        subplot_titles=("Accuracy", f"Gain over the {rshort(ref)}"))
    fig.add_trace(go.Scatter(
        x=R.acc_m, y=y, mode="markers+text", marker=dict(color=BLUE, size=13, opacity=op),
        text=[f"<b>{a:.1f}%</b>" if r == top_a else f"{a:.1f}%" for r, a in zip(y, R.acc_m)],
        textposition="middle right", textfont=dict(size=11, color=INK), cliponaxis=False, customdata=R.acc_r,
        hovertemplate="%{y}: %{x:.1f}%<br>comparison %{customdata:.1f}%<extra></extra>"), 1, 1)
    fig.add_trace(go.Bar(
        x=R.gain, y=y, orientation="h", marker=dict(color=[BLUE if g >= 0 else RED for g in R.gain], opacity=op),
        customdata=np.c_[R.better, R.seasons],
        hovertemplate="%{y}: %{x:+.1f} pp<br>better in %{customdata[0]:.0f} of %{customdata[1]:.0f} seasons"
                      "<extra></extra>"), 1, 2)
    wx, wy = [], []
    for l_, h_, y_ in zip(R.lo, R.hi, y):
        if np.isfinite(l_) and np.isfinite(h_):
            wx += [l_, h_, None]
            wy += [y_, y_, None]
    if wx:
        fig.add_trace(go.Scatter(x=wx, y=wy, mode="lines", line=dict(color="#5d6470", width=1.3), hoverinfo="skip"),
                      1, 2)
    span = float(np.nanmax(np.abs(np.r_[R.gain, R.lo, R.hi, 1.0])))
    edge = np.fmax(R.gain.to_numpy(), np.nan_to_num(R.hi.to_numpy(), nan=-np.inf))
    fig.add_trace(go.Scatter(x=edge + 0.04 * span, y=y, mode="text", textposition="middle right", cliponaxis=False,
                             text=[f"<b>{fs(g)} pp</b>" if r == top_g else f"{fs(g)} pp" for r, g in zip(y, R.gain)],
                             textfont=dict(size=11, color=INK), hoverinfo="skip"), 1, 2)
    fig.add_vline(x=0, line_color=GREY, line_width=1, row=1, col=2)
    layout(fig, h, yaxis=dict(autorange="reversed"), showlegend=False, bargap=0.35, margin=dict(l=8, r=8, t=28, b=8))
    lo_a, hi_a = float(R.acc_m.min()), float(R.acc_m.max())
    fig.update_xaxes(ticksuffix="%", range=[lo_a - 1.0, hi_a + 1.6], row=1, col=1)
    fig.update_xaxes(ticksuffix=" pp", range=[min(0.0, float(np.nanmin(np.r_[R.gain, R.lo]))) - 0.05 * span,
                                              span * 1.45], row=1, col=2)
    starts = [fig.layout.xaxis.domain[0], fig.layout.xaxis2.domain[0]]
    for an, x0 in zip(fig.layout.annotations, starts):     # subplot titles left-aligned like other chart titles
        an.update(xanchor="left", x=x0, font=dict(size=12, color=MUTED))
    return fig


def _region_click():
    pts = [q for q in _points("region_bars") if q.get("y") in REGIONS]
    r = pts[0]["y"] if pts else ""
    if r in REGIONS:
        ss["f_region"] = "All regions" if ss.get("f_region") == r else r
        if ss.get("f_state") != "All states" and REGION_OF.get(ss.get("f_state")) != ss["f_region"]:
            ss["f_state"] = "All states"


def district_card(name: str, compact: bool = False):
    code = CODE_OF.get(name)
    D = E_ALL[(E_ALL.landkreis == code) & E_ALL.year.between(*years)]
    FD = F26_ALL[F26_ALL.landkreis == code]
    crops_here = [c for c in CROPS if c in set(D.crop) | set(FD.crop)]
    with (nullcontext() if compact else panel("district", "sky")):
        st.markdown(f"#### {html.escape(name)} <span style='color:{MUTED};font-size:0.9rem'>({code} · "
                    f"{html.escape(sname(STATE_OF.get(code)))})</span>", unsafe_allow_html=True)
        if not crops_here:
            st.info("No official yields for this district.")
            return
        key = f"dc_crop_{'c' if compact else 'f'}"
        want = crop if crop in crops_here else crops_here[0]
        if ss.get(key) != want:
            ss[key] = want
        widget = st.selectbox if compact else partial(st.radio, horizontal=True)       # pills wrap in the narrow panel
        c = widget("Crop (sets the Crop filter)", crops_here, format_func=cname, key=key, on_change=_set_from,
                   args=("f_crop", key), label_visibility="collapsed",
                   help="Choosing a crop here sets the Crop filter for the whole dashboard.")
        d, f = D[D.crop == c], FD[FD.crop == c]
        st.markdown(f"<div class='sub'>The four cards below show <b>{cname(c)}</b>; the cards at the top of the page "
                    "show the whole selection.</div>", unsafe_allow_html=True)
        a = AREA[(AREA.landkreis == code) & (AREA.crop == c)]
        t = st.columns(2) + st.columns(2) if compact else st.columns(4)
        if len(f):
            r = f.iloc[0]
            card(t[0], "2026 forecast", f"{r.y_pred:.1f} dt/ha", f"80% range {r.lo80:.1f}–{r.hi80:.1f}", small=True)
            card(t[1], "Normal yield", f"{r.y_hist:.1f} dt/ha", f"forecast {fs(r.vs_normal)}% vs normal", small=True)
        else:
            card(t[0], "2026 forecast", "–", "no forecast", small=True)
            card(t[1], "Normal yield", "–", "", small=True)
        if len(d):
            card(t[2], f"Accuracy {yr_txt}", f"{acc(d[model], d.actual):.1f}%",
                 f"{dn(ref)}: {acc(d[ref], d.actual):.1f}%", small=True)
        else:
            card(t[2], "Accuracy", "–", "no official yields in these years", small=True)
        if len(a):
            sh = a.share_of_farmland.iloc[0]
            card(t[3], "Crop area (census)", f"{a.area_ha.iloc[0]:,.0f} ha",
                 f"{'<0.1' if sh < 0.1 else f'{sh:.1f}'}% of farmland", small=True)
        else:
            card(t[3], "Crop area (census)", "–", "", small=True)
        if len(d):
            st.plotly_chart(selection_trend(d, f, f"{cname(c)}: official vs backtest forecast, {FC_YEAR} sealed forecast", 290),
                            use_container_width=True, config=CFG, key=f"dc_trend_{'c' if compact else 'f'}")
    if compact:
        st.caption("Open **District look-up** for every year, every crop and the download.")
        return
    left, right = st.columns([1.15, 1])
    with left, panel("yearbyyear", "rose"):
        st.markdown(f"##### Year by year · {cname(c)}")
        tab = d.sort_values("year", ascending=False)
        out = pd.DataFrame({"Year": tab.year.astype(int), "Official (dt/ha)": tab.actual.map(f1),
                            "Forecast (dt/ha)": tab[model].map(f1),
                            "Error (%)": (100 * (tab[model] / tab.actual - 1)).map(fs),
                            f"{RS} (dt/ha)": tab[ref].map(f1),
                            f"{RS} error (%)": (100 * (tab[ref] / tab.actual - 1)).map(fs)})
        paged(out, "pg_dc_years", numeric=tuple(out.columns), per=16)
    with right, panel("allcrops", "teal"):
        st.markdown("##### All crops in this district")
        rows = []
        for cc in crops_here:
            dd, ff = D[D.crop == cc], FD[FD.crop == cc]
            rows.append({"Crop": cname(cc),
                         "2026 forecast (dt/ha)": f1(ff.y_pred.iloc[0]) if len(ff) else "–",
                         "vs normal (%)": fs(ff.vs_normal.iloc[0]) if len(ff) else "–",
                         "Accuracy (%)": f1(acc(dd[model], dd.actual)) if len(dd) else "–",
                         f"{RS} accuracy (%)": f1(acc(dd[ref], dd.actual)) if len(dd) else "–",
                         "Years": len(dd)})
        R = pd.DataFrame(rows)
        table(R, numeric=tuple(R.columns[1:]))
        out = (D.assign(o=D.crop.map(ORDER)).sort_values(["o", "year"])
               .merge(FD[["crop", "y_pred", "lo80", "hi80", "y_hist"]], on="crop", how="left")
               [["landkreis", "district", "crop", "year", "actual", model, ref, "y_pred", "lo80", "hi80", "y_hist"]]
               .rename(columns={"landkreis": "nuts_code", "actual": "official_dt_ha", model: "estimate_dt_ha",
                                ref: "comparison_dt_ha", "y_pred": "forecast_2026_dt_ha", "lo80": "low80_2026",
                                "hi80": "high80_2026", "y_hist": "normal_2026"})
               .assign(estimate_model=model, comparison=ref).round(2))
        st.download_button("⤓ Download this district (CSV)", csv_bytes(out), file_name=f"yieldformer_de_{code}.csv",
                           mime="text/csv")


def file_slug(*parts) -> str:
    return "_".join(re.sub(r"[^A-Za-z0-9-]+", "-", str(p)).strip("-").lower() for p in parts if p)


# --------------------------------------------------------------------------- views
if view == "Map":
    left, right = st.columns([1.15, 1])
    with left, panel("map", "slate"):
        yc, lc = st.columns([1.9, 3], vertical_alignment="center")
        my = yc.selectbox("Map year", MAP_YEARS, key="map_year", label_visibility="collapsed",
                          format_func=lambda y: f"{y} sealed forecast" if y == FC_YEAR else f"{y} (harvest) · backtest forecast",
                          help=f"{FC_YEAR}: the sealed forecast. {FIRST_YEAR}–{LAST_YEAR}: the official harvest and the backtest forecast "
                               "from a model that never saw that season.")
        past = my != FC_YEAR
        names = {"vs_normal": "vs normal", "level": "dt/ha", "error": "Forecast error", "accuracy": "Accuracy",
                 "gain": "Gain", "area": "Crop area"}
        one = crop != "All crops"
        layers = (["vs_normal"] + (["level"] if one else []) + (["error"] if past else []) + ["accuracy", "gain"]
                  + (["area"] if one else []))
        if ss.get("map_layer") not in layers:
            ss["map_layer"] = "vs_normal"
        with lc:
            layer = st.radio("Colour the map by", layers, format_func=names.get, horizontal=True, key="map_layer",
                             label_visibility="collapsed")
        fig, note = None, ""
        what = cname(crop) if one else "all crops (area-weighted)"
        if past:                                                   # official harvest + estimate of that year
            Y = select(E_ALL, use_years=False, use_district=False, use_area=False)
            Y = Y[Y.year == my].assign(off=lambda x: 100 * (x.actual / x["Long-term average"] - 1),
                                       est=lambda x: 100 * (x[model] / x["Long-term average"] - 1),
                                       err=lambda x: 100 * (x[model] / x.actual - 1),
                                       abserr=lambda x: 100 * (x[model] / x.actual - 1).abs())
            YI = Y.set_index("landkreis")

            def head(i):
                return f"<b>{html.escape(NAME_OF.get(i, i))}</b> · {sname(STATE_OF.get(i))}<br>"

            def row(i):
                r = YI.loc[i]
                return (f"official {f1(r.actual)} dt/ha ({fs(r.off)}% vs normal)<br>forecast {f1(r[model])} dt/ha "
                        f"({fs(r.est)}%) · normal {f1(r['Long-term average'])}")
        if layer == "vs_normal" and past:
            v = YI.off if one else area_weighted(Y, "off")
            ev = None if one else area_weighted(Y, "est")
            fig = district_figure(v, f"{my}: official yield vs normal, {what}", DIV, diverging=True, unit="%",
                                  hover=lambda i, x: head(i) + (row(i) if one else
                                                                f"official {fs(x)}% vs normal · forecast "
                                                                f"{fs(ev.get(i))}%"))
            note = (f"How the {my} harvest went: blue = official yield above the district's normal, red = below. "
                    "Hover for the backtest forecast.")
        elif layer == "vs_normal":
            if one:
                v = F_MAP.set_index("landkreis").vs_normal
            else:
                v = area_weighted(F_MAP)
            ff = F_MAP.set_index("landkreis") if one else None

            def hov(i, x):
                s = f"<b>{html.escape(NAME_OF.get(i, i))}</b> · {sname(STATE_OF.get(i))}<br>{fs(x)}% vs normal"
                if ff is not None and i in ff.index:
                    r = ff.loc[i]
                    s += f"<br>forecast {f1(r.y_pred)} dt/ha (80%: {f1(r.lo80)}–{f1(r.hi80)}) · normal {f1(r.y_hist)}"
                return s
            fig = district_figure(v, f"2026 forecast vs normal, {what}", DIV, diverging=True, hover=hov, unit="%")
            note = "Blue = 2026 forecast above the district's normal yield, red = below."
        elif layer == "level" and past:
            fig = district_figure(YI.actual, f"{my}: official yield, {cname(crop)} (dt/ha)", SEQ_GREY,
                                  hover=lambda i, x: head(i) + row(i))
            note = "Grey shades: a quantity, not a judgement."
        elif layer == "level":
            ff = F_MAP.set_index("landkreis")
            fig = district_figure(ff.y_pred, f"2026 forecast, {cname(crop)} (dt/ha)", SEQ_GREY,
                                  hover=lambda i, x: f"<b>{html.escape(NAME_OF.get(i, i))}</b> · "
                                                     f"{sname(STATE_OF.get(i))}<br>forecast {f1(x)} dt/ha (80%: "
                                                     f"{f1(ff.lo80.get(i))}–{f1(ff.hi80.get(i))})<br>normal "
                                                     f"{f1(ff.y_hist.get(i))} dt/ha · {fs(ff.vs_normal.get(i))}%")
            note = "Grey shades: a quantity, not a judgement."
        elif layer == "error":
            if one:
                fig = district_figure(YI.err, f"{my}: backtest forecast vs official, {cname(crop)}", DIV, diverging=True,
                                      unit="%", hover=lambda i, x: head(i) + row(i) + f"<br>error {fs(x)}%")
                note = "Blue = forecast above the official yield, red = below."
            else:
                fig = district_figure(area_weighted(Y, "abserr"), f"{my}: size of the forecast error, {what}",
                                      SEQ_RED, clip=True, unit="%")
                note = "Darker = larger error (absolute, so over- and under-estimates of different crops do not cancel)."
        elif layer == "accuracy":
            n = E_MAP.groupby("landkreis").size()
            v = acc_by(E_MAP, model)[n >= 5]
            fig = district_figure(v, f"accuracy {yr_txt}", SEQ_BLUE, clip=True, unit="%")
            note = f"Accuracy over {yr_txt} (Years filter); districts with at least 5 backtest forecasts."
        elif layer == "gain":
            n = E_MAP.groupby("landkreis").size()
            v = (acc_by(E_MAP, model) - acc_by(E_MAP, ref))[n >= 5]
            fig = district_figure(v, f"gain vs {rshort(ref)}, {yr_txt}", DIV, diverging=True, unit=" pp")
            note = (f"Over {yr_txt} (Years filter): blue = more accurate than the {rshort(ref)}, red = less; "
                    "districts with at least 5 backtest forecasts.")
        else:
            a = AREA[AREA.crop == crop].set_index("landkreis")
            fig = district_figure(a.share_of_farmland, f"{cname(crop)}, share of farmland (census)", SEQ_GREY,
                                  hover=lambda i, x: f"<b>{html.escape(NAME_OF.get(i, i))}</b> · "
                                                     f"{sname(STATE_OF.get(i))}<br>{f1(a.area_ha.get(i), 0)} ha · "
                                                     f"{f1(x)}% of farmland", unit="%")
            note = "Grey shades: a quantity, not a judgement."
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True, config=CFG, on_select=_map_click,
                            selection_mode="points", key="map_main")
            st.caption(note + " Blank = no data. Click a district to open its card.")
    with left, panel("regions", "sky"):
        R = gains(E_MAP, "region")
        st.markdown(f"##### Which region does better? · {yr_txt}")
        st.markdown(f"<div class='sub'>{region_headline(R)}</div>", unsafe_allow_html=True)
        st.plotly_chart(region_figure(R, sel=None if region == "All regions" else region),
                        use_container_width=True, config=CFG, on_select=_region_click, selection_mode="points",
                        key="region_bars")
        st.caption("Accuracy: how predictable the region is. Gain: what the model adds there (thin line: 90% range "
                   "over seasons). Click a region to filter.")
    with right:
        if district != "All districts":
            with panel("district", "sky"):
                district_card(district, compact=True)
        elif crop != "All crops":
            with panel("trend", "sky"):
                st.plotly_chart(selection_trend(E, F26, f"{cname(crop)} · {scope}: official vs backtest forecast, {FC_YEAR} sealed forecast"),
                                use_container_width=True, config=CFG, on_select=lambda: _year_click("trend_main"),
                                selection_mode="points", key="trend_main")
                st.caption("Click a year to show only that year.")
            with panel("crops", "violet"):
                st.plotly_chart(gain_bars(gains(select(E_ALL, use_crop=False), "crop"), CROPS, cname,
                                          f"Gain by crop · {scope} · {yr_txt}", sel=crop), use_container_width=True,
                                config=CFG)
        else:
            with panel("crops", "violet"):
                st.plotly_chart(gain_bars(gains(E, "crop"), CROPS, cname, f"Gain by crop · {scope} · {yr_txt}"),
                                use_container_width=True, config=CFG)
                st.caption(f"Bar: accuracy of the model minus the {rshort(ref)}, in percentage points. Thin line: 90% "
                           "range over seasons; a line crossing zero means the gain is not clear.")
            with panel("years", "teal"):
                st.plotly_chart(season_columns(E, f"Gain by season · {scope}", h=420), use_container_width=True, config=CFG,
                                on_select=lambda: _year_click("year_main"), selection_mode="points", key="year_main")
                st.caption("Click a season to show only that season.")

    # Season by season as lines (user, 2026-10-31): does the forecast follow good and poor years, and how accurate
    # was it each season? Both follow every filter.
    l2, r2 = st.columns([1.15, 1])
    with l2, panel("swing", "slate"):
        st.plotly_chart(swing_trend(E, F26, f"Forecast vs official, season by season · {scope}"),
                        use_container_width=True, config=CFG, on_select=lambda: _year_click("swing_main"),
                        selection_mode="points", key="swing_main")
        st.caption("% above or below the district normal, averaged over the district-crop-years in the selection, so "
                   f"all crops can be shown together. Diamond: the sealed {FC_YEAR} forecast with its 80% range, not "
                   "yet scored. Click a season to show only that season.")
    with r2, panel("accseason", "teal"):
        st.plotly_chart(accuracy_lines(E, f"Accuracy by season · {scope}"), use_container_width=True, config=CFG,
                        on_select=lambda: _year_click("acc_main"), selection_mode="points", key="acc_main")
        st.caption("Accuracy (100 minus the mean % error) of the backtest forecasts in each season, next to the "
                   "comparison. Poor seasons such as 2018 are where the simple averages fall furthest.")
    l3, r3 = st.columns([1.15, 1])
    with l3, panel("bydate", "violet"):
        fig, g_ = date_chart(f"Accuracy by forecast date · {cname(crop) if crop != 'All crops' else 'all crops'} · "
                         f"Germany · {FIRST_YEAR}–{LAST_YEAR}")
        if fig is None:
            st.info("No forecast-date results for this crop.")
        else:
            st.plotly_chart(fig, use_container_width=True, config=CFG)
            st.caption("30 June and 31 July: the same networks run with weather and satellite data cut at that date "
                       "(a test version, not used for the final model). Harvest: the final model. Only the Crop filter applies. "
                       f"Gap to the 5-year average: {fs(g_[0])} pp at 30 June, {fs(g_[1])} at 31 July, "
                       f"{fs(g_[-1])} at harvest.")
    with r3, panel("about", "sky"):
        st.markdown("##### Data & method")
        st.markdown(
            f"- **Question:** the official yield of ten field crops in the {E_ALL.landkreis.nunique()} German districts "
            f"with official yields, estimated at the end of the season, months before Destatis publishes it.\n"
            f"- **Data:** official district yields (Destatis, table 41241-01-03-4); daily weather (DWD stations, "
            f"ERA5-Land soil water); MODIS satellite vegetation indices over arable land; DWD crop development "
            f"stages; SoilGrids soil; census crop areas. Seasons 2000–{LAST_YEAR}. District boundaries: "
            f"© EuroGeographics for the administrative boundaries (GISCO), non-commercial use.\n"
            f"- **Model:** {MODEL}, the average of three transformer networks (Temporal Fusion Transformer, "
            f"iTransformer, PatchTST) plus a district error memory.\n"
            f"- **Test:** each season {FIRST_YEAR}–{LAST_YEAR} forecast by a model trained only on earlier seasons "
            f"(backtest); the {FC_YEAR} forecast sealed with a SHA-256 fingerprint before the official figures.\n"
            "- **Links:** [dashboard](https://yieldformer-de.streamlit.app) · "
            "[code](https://github.com/roshangupta00750/transformer-based-crop-yield-prediction) · "
            "[model](https://huggingface.co/Roshangupta007/YieldFormer-DE) · "
            "[data](https://huggingface.co/datasets/Roshangupta007/YieldFormer-DE-data) · report: the written project "
            "report (JMU Würzburg).")

elif view == "District look-up":
    names = sorted(pool.district.dropna().unique())
    opts = ["– choose a district –"] + names
    ss["lk_pick"] = district if district in opts else opts[0]
    st.selectbox("District (type to search)", opts, key="lk_pick", on_change=_set_from, args=("f_district", "lk_pick"),
                 label_visibility="collapsed")
    if district == "All districts":
        st.info("Choose a district above, in the District filter or on the Map to open its card: 2026 forecast, "
                "normal yield, accuracy, crop area, every year and every crop, with a download.")
    else:
        district_card(district)

elif view == "Table & download":
    T = E.assign(o=E.crop.map(ORDER)).sort_values(["year", "state", "district", "o"],
                                                   ascending=[False, True, True, True])
    with panel("estimates", "rose"):
        lc_, rc_ = st.columns([1.4, 1])
        level = lc_.radio("Level", LEVELS, horizontal=True, key="tbl_level",
                          help="District: one row per district, crop and year. State, Region, Germany: the district "
                               "yields of the selection averaged with each district's crop area as weight, as official "
                               "state and national yields are.")
        with rc_:
            q = search_box("tbl_search")
        L = by_level(T, ["actual", model, ref], level)
        if level != "District":
            L = L.assign(o=L.crop.map(ORDER)).sort_values(["year", "unit", "o"], ascending=[False, True, True])
        else:
            L = L.assign(state_name=L.state.map(sname))
        if q:
            hay = (L.unit.astype(str) + " " + (L.state_name if "state_name" in L else "")).str.lower()
            L = L[hay.str.contains(q, regex=False)]
        e_m, e_r = 100 * (L[model] / L.actual - 1), 100 * (L[ref] / L.actual - 1)
        unit_col = {"District": "District", "State": "State", "Region": "Region", "Germany": "Area"}[level]
        show = pd.DataFrame({"Season": L.year.astype(int).astype(str) + " (harvest)"})
        if level == "District":
            show["State"] = L.state_name
        show[unit_col] = L.unit
        show["Crop"] = L.crop.map(cname)
        if level != "District":
            show["Districts"] = L.n.astype(int)
        show = show.assign(**{"Official (dt/ha)": L.actual.map(f1), "Forecast (dt/ha)": L[model].map(f1),
                              "Error (%)": e_m.map(fs), f"{RS} (dt/ha)": L[ref].map(f1), f"{RS} error (%)": e_r.map(fs),
                              "Gain (pp)": (e_r.abs() - e_m.abs()).map(fs)})
        st.markdown(f"##### Backtest forecasts in the selection: official vs forecast by {level.lower()} "
                    f"({plural(len(show), 'row')})")
        st.caption("Backtest forecast = the forecast for a past season (harvest year in brackets) from a model trained only "
                   "on earlier seasons, so it never saw that year. The sealed forecasts for future seasons are under "
                   "Forecasts. "
                   "Gain = how many percentage points closer to the official yield than the comparison.")
        paged(show, "pg_tbl", numeric=("Districts", "Official (dt/ha)", "Forecast (dt/ha)", "Error (%)",
                                       f"{RS} (dt/ha)", f"{RS} error (%)", "Gain (pp)"))
    level_csv = (L[["year", "unit", "crop", "n", "actual", model, ref]]
                 .rename(columns={"unit": level.lower(), "n": "districts", "actual": "official_dt_ha",
                                  model: "estimate_dt_ha", ref: "comparison_dt_ha"})
                 .assign(estimate_model=model, comparison=ref, level=level.lower()).round(2))
    with panel("downloads", "green"):
        st.markdown("##### Downloads")
        st.checkbox("German Excel format (semicolon, decimal comma)", key="csv_de")
        slug = file_slug(crop if crop != "All crops" else "all-crops", sname(state) if state != "All states" else region if region != "All regions" else "germany",
                         CODE_OF.get(district) if district != "All districts" else "", yr_txt)
        d0, d1, d2, d3 = st.columns(4)
        d0.download_button(f"⤓ {level} table as shown (CSV)", csv_bytes(level_csv),
                           file_name=f"yieldformer_de_{level.lower()}_table_{slug}.csv", mime="text/csv",
                           width="stretch")
        raw = (T[["year", "landkreis", "state", "district", "crop", "actual", model, ref]]
               .assign(state=lambda x: x.state.map(sname))
               .rename(columns={"landkreis": "nuts_code", "actual": "official_dt_ha", model: "estimate_dt_ha",
                                ref: "comparison_dt_ha"})
               .assign(estimate_model=model, comparison=ref).round(2))
        d1.download_button("⤓ Backtest forecasts in this selection (CSV)", csv_bytes(raw),
                           file_name=f"yieldformer_de_backtest_forecasts_{slug}.csv", mime="text/csv", width="stretch")
        fc = (F26.assign(state=F26.landkreis.map(STATE_OF).map(sname), o=F26.crop.map(ORDER))
              .sort_values(["state", "district", "o"])
              [["landkreis", "state", "district", "crop", "y_pred", "lo80", "hi80", "y_hist", "vs_normal", "model",
                "issued", "data_end"]]
              .rename(columns={"landkreis": "nuts_code", "y_pred": "forecast_dt_ha", "lo80": "low80_dt_ha",
                               "hi80": "high80_dt_ha", "y_hist": "normal_dt_ha", "vs_normal": "vs_normal_pct"}).round(2))
        d2.download_button("⤓ 2026 forecasts in this selection (CSV)", csv_bytes(fc),
                           file_name=f"yieldformer_de_forecast_2026_{file_slug(crop if crop != 'All crops' else 'all-crops', sname(state) if state != 'All states' else region if region != 'All regions' else 'germany', CODE_OF.get(district) if district != 'All districts' else '')}.csv",
                           mime="text/csv", width="stretch")
        dictionary = pd.DataFrame([
            ("nuts_code", "district code (NUTS 3, 2021)"),
            ("official_dt_ha", "yield reported by Destatis (Regionalstatistik table 41241-01-03-4), dt/ha"),
            ("estimate_dt_ha", "backtest forecast of estimate_model for a season (harvest year) that model never saw: trained "
                               "only on earlier seasons"),
            ("comparison_dt_ha", "the comparison named in 'comparison' (5-year average, long-term average or XGBoost "
                                 "reference)"),
            ("level / districts", "table level (district, state, region, germany) and the number of districts averaged; "
                                  "above district level the yields are weighted by census crop area"),
            ("forecast_year / official_dt_ha (sealed)", "season of a sealed forecast; its official yield once "
                                                        "Destatis has published it, empty before"),
            ("forecast_dt_ha", f"sealed 2026 forecast of {MODEL}"),
            ("low80_dt_ha / high80_dt_ha", "80% range from past errors"),
            ("normal_dt_ha", "district normal for 2026: long-term mean plus a trend if clear"),
            ("vs_normal_pct", "forecast / normal − 1, in %"),
            ("issued / data_end", "date the forecast file was sealed / last day of weather and satellite data used")],
            columns=["column", "meaning"])
        d3.download_button("⤓ Data dictionary (CSV)", csv_bytes(dictionary), file_name="yieldformer_de_data_dictionary.csv",
                           mime="text/csv", width="stretch")

elif view == "Forecasts":
    if META is not None:
        st.markdown(f"<div class='sub'>Forecast shown: <b>{MODEL}</b>, sealed {dfmt(META.issued)} with data to "
                    f"{dfmt(META.data_end)} · {plural(len(F26), 'district-crop forecast')} in the selection · not yet "
                    f"scored: Destatis has not published 2026 district yields."
                    + (" The Model filter applies to the 2010–2025 backtest forecasts only." if model != MODEL else "")
                    + "</div>", unsafe_allow_html=True)
    SF = (sealed_files(max(f.stat().st_mtime for f in SEALED.iterdir()))
          if (SEALED / "FROZEN.sha256").exists() else pd.DataFrame())
    left, right = st.columns([1, 1])
    with left, panel("fcchart", "violet"):
        if F26.empty:
            st.info("No 2026 forecast for this selection.")
        elif crop == "All crops":
            fc = F26.merge(AREA[["landkreis", "crop", "area_ha"]], on=["landkreis", "crop"], how="left")
            fc["w"] = fc.area_ha.fillna(0.0)
            rows = []
            for c in CROPS:
                g = fc[fc.crop == c]
                if g.empty:
                    continue
                rows.append({"crop": c, "vs": np.average(g.vs_normal, weights=g.w) if g.w.sum() > 0 else g.vs_normal.mean(),
                             "p10": g.vs_normal.quantile(0.1), "p90": g.vs_normal.quantile(0.9),
                             "above": 100 * (g.vs_normal > 0).mean(), "pred": g.y_pred.mean(), "norm": g.y_hist.mean()})
            t = pd.DataFrame(rows)
            fig = hbars(t.vs, t.p10, t.p90, [cname(c) for c in t.crop], [f"{fs(x)}%" for x in t.vs],
                        f"2026 forecast vs normal · {scope}", customdata=np.c_[t.pred, t.norm, t.above, t.p10, t.p90],
                        hover="%{y}: %{x:+.1f}% vs normal<br>forecast %{customdata[0]:.1f} dt/ha, normal "
                              "%{customdata[1]:.1f}<br>%{customdata[2]:.0f}% of districts above normal<br>districts "
                              "range %{customdata[3]:+.1f} to %{customdata[4]:+.1f}%<extra></extra>", unit="%", h=470)
            st.plotly_chart(fig, use_container_width=True, config=CFG)
            st.caption("Bar: forecast vs normal, weighted by crop area (as the card). Thin line: spread across "
                       "districts (10th to 90th percentile).")
        elif len(F26) == 1:
            r = F26.iloc[0]
            fig = go.Figure(go.Scatter(x=[r.vs_normal], y=[cname(crop)], mode="markers",
                                       marker=dict(color=BLUE if r.vs_normal >= 0 else RED, size=14),
                                       error_x=dict(type="data", symmetric=False, color=LGREY, thickness=3,
                                                    array=[100 * (r.hi80 / r.y_hist - 1) - r.vs_normal],
                                                    arrayminus=[r.vs_normal - 100 * (r.lo80 / r.y_hist - 1)]),
                                       hovertemplate="%{x:+.1f}% vs normal<extra></extra>"))
            fig.add_vline(x=0, line_color=GREY, line_width=1)
            layout(fig, 200, title=dict(text=f"{cname(crop)} · {scope}: 2026 forecast vs normal (80% range)",
                                        font=dict(size=14), x=0), xaxis=dict(ticksuffix="%"), showlegend=False)
            st.plotly_chart(fig, use_container_width=True, config=CFG)
        else:
            v = F26.vs_normal
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=v[v >= 0], name="above normal", marker_color=BLUE, xbins=dict(size=2),
                                       hovertemplate="%{x}%: %{y} districts<extra></extra>"))
            fig.add_trace(go.Histogram(x=v[v < 0], name="below normal", marker_color=RED, xbins=dict(size=2),
                                       hovertemplate="%{x}%: %{y} districts<extra></extra>"))
            fig.add_vline(x=0, line_color=GREY, line_width=1)
            layout(fig, 380, barmode="overlay",
                   title=dict(text=f"{cname(crop)} · {scope}: districts by 2026 forecast vs normal", font=dict(size=14),
                              x=0), xaxis=dict(ticksuffix="%", title="forecast vs normal"), yaxis_title="districts")
            st.plotly_chart(fig, use_container_width=True, config=CFG)
            st.caption(f"{int((v >= 0).sum())} of {plural(len(v), 'district')} above their normal yield.")
    with right, panel("sealed", "green"):
        st.markdown("##### Sealed forecast files")
        st.caption("Each file was saved before its season's official yields existed and fingerprinted (SHA-256) the "
                   "same day. ✓ = the file is unchanged since then.")
        with st.expander("Details: how the check works"):
            st.caption("The fingerprints were recorded on the issue date in FROZEN.sha256 (kept in the project's git "
                       "history) and are recomputed on every page load. Download a file and check it yourself with "
                       "sha256sum -c FROZEN.sha256. The 2027 and 2028 files carry 'yieldformer' in their names but "
                       "contain district normals, the model's rule for a season that has not started. A public copy is "
                       "on Hugging Face (Roshangupta007/YieldFormer-DE), uploaded 30 Oct 2026, after the issue dates.")
        if SF.empty:
            st.info("No sealed files found in results/forecasts.")
        else:
            kind = F_ALL.drop_duplicates("file").set_index("file").model.to_dict()

            def label(f: str) -> str:                 # short, so the table fits beside the chart
                if f == META_FILE:
                    return "Final model (v18)"
                if "baseline" in f:
                    return "Normal · pre-registered"
                if kind.get(f) == "District normal":
                    return "District normal"
                if kind.get(f):
                    return "Earlier model (v15)"
                return "Pre-registered test (v1)"
            table(pd.DataFrame({"Season": SF.year.astype(int), "Forecast": SF.file.map(label),
                                "Issued": SF.issued.map(lambda d: pd.Timestamp(d).strftime("%-d %b")), "Fingerprint": SF.recorded.str[:8] + "…",
                                "Unchanged": SF.check.map({"ok": "✓", "changed": "✗ changed",
                                                           "missing": "file missing"})}),
                  numeric=("Season",))
            pick = st.selectbox("Sealed file", SF.file, key="sealed_pick", label_visibility="collapsed",
                                format_func=lambda f: f"{f}")
            if (SEALED / pick).exists():
                st.download_button("⤓ Download this sealed file, unchanged", (SEALED / pick).read_bytes(),
                                   file_name=pick, mime="text/csv", width="stretch")
            st.download_button("⤓ FROZEN.sha256 (the recorded fingerprints)", (SEALED / "FROZEN.sha256").read_bytes(),
                               file_name="FROZEN.sha256", mime="text/plain", width="stretch")

    FYEARS = sorted(int(y) for y in F_ALL.year.unique())
    with panel("fctable", "rose"):
        a_, b_ = st.columns([1.4, 1])
        fy = a_.radio("Season", FYEARS, horizontal=True, key="fc_year",
                      format_func=lambda y: f"{y} · {'final model' if y == FC_YEAR else 'district normal'}")
        with b_:
            q = search_box("fc_search")
        src = F_ALL[(F_ALL.year == fy) & (F_ALL.model == (MODEL if fy == FC_YEAR else "District normal"))]
        T = select(src, use_years=False)
        last = E_ALL[E_ALL.year == LAST_YEAR][["landkreis", "crop", "actual"]]
        T = T.merge(last, on=["landkreis", "crop"], how="left")
        off = OFFICIAL[OFFICIAL.year == fy][["landkreis", "crop", "official"]] if not OFFICIAL.empty else None
        if off is not None and len(off):
            T = T.merge(off, on=["landkreis", "crop"], how="left")
        T = T.assign(state_name=T.landkreis.map(STATE_OF).map(sname), o=T.crop.map(ORDER)).sort_values(["district", "o"])
        if q:
            T = T[(T.district.astype(str) + " " + T.state_name).str.lower().str.contains(q, regex=False)]
        out = pd.DataFrame({"District": T.district, "State": T.state_name, "Crop": T.crop.map(cname),
                            "Forecast": T.y_pred.map(f1), "Low 80%": T.lo80.map(f1), "High 80%": T.hi80.map(f1),
                            "Normal": T.y_hist.map(f1), "vs normal (%)": T.vs_normal.map(fs),
                            f"Official {LAST_YEAR}": T.actual.map(f1)})
        num = ["Forecast", "Low 80%", "High 80%", "Normal", "vs normal (%)", f"Official {LAST_YEAR}"]
        if "official" in T:
            out[f"Official {fy}"] = T.official.map(f1)
            out["Error (%)"] = (100 * (T.y_pred / T.official - 1)).map(fs)
            num += [f"Official {fy}", "Error (%)"]
        st.markdown(f"##### District forecasts for {fy}, dt/ha ({plural(len(out), 'row')})")
        if fy != FC_YEAR:
            st.caption(f"{fy} was sealed before its season began, so there is no weather or satellite data yet: the "
                       "forecast is the district normal (mean of earlier years plus a trend if clear). It is the no-skill "
                       "reference the final model has to beat in that season.")
        st.caption("– = no official yield published for that district and crop.")
        paged(out, "pg_fc", numeric=tuple(num), per=12)

    with panel("reveal", "sky"):
        st.markdown(f"##### Revealed: sealed {FC_YEAR} forecast vs official yield")
        V1 = load_csv("forecast_2026.csv")            # the pre-registered test file, as written by scripts/63
        R = select(F26_ALL, use_years=False)
        if not OFFICIAL.empty:
            R = R.merge(OFFICIAL[OFFICIAL.year == FC_YEAR][["landkreis", "crop", "official"]],
                        on=["landkreis", "crop"], how="inner")
        if OFFICIAL.empty or R.empty:
            st.markdown(f"<div class='sub'>Not scored yet: fills in by itself when Destatis publishes the {FC_YEAR} "
                        f"district yields, for the pre-registered file of {dfmt(V1.issued.iloc[0]) if len(V1) else '–'} "
                        f"and the final model's file of {dfmt(META.issued) if META is not None else '–'}.</div>",
                        unsafe_allow_html=True)
        else:
            a_m, a_n = acc(R.y_pred, R.official), acc(R.y_hist, R.official)
            k_ = st.columns(3)
            card(k_[0], f"Accuracy {FC_YEAR}", f"{a_m:.1f}%", f"district normal {a_n:.1f}%", hero=True,
                 chip=(f"{fs(a_m - a_n)} pp", a_m >= a_n), small=True, accent=ACCENT["forecast"])
            R_ = R.assign(actual=R.official)
            per = acc_by(R_, "y_pred") > acc_by(R_, "y_hist")
            card(k_[1], "Districts closer", f"{int(per.sum())} of {len(per)}", "than the district normal",
                 small=True)
            card(k_[2], "Within ±10%", f"{100 * np.mean(np.abs(R.y_pred / R.official - 1) <= 0.10):.0f}%",
                 f"normal {100 * np.mean(np.abs(R.y_hist / R.official - 1) <= 0.10):.0f}%", small=True)
            rows = [(cname(c), len(g), acc(g.y_pred, g.official), acc(g.y_hist, g.official))
                    for c in CROPS if len(g := R[R.crop == c])]
            t = pd.DataFrame(rows, columns=["Crop", "Districts", "Forecast accuracy (%)", "District normal accuracy (%)"])
            t["Gain (pp)"] = (t["Forecast accuracy (%)"] - t["District normal accuracy (%)"]).map(fs)
            for c in ["Forecast accuracy (%)", "District normal accuracy (%)"]:
                t[c] = t[c].map(f1)
            table(t, numeric=("Districts", "Forecast accuracy (%)", "District normal accuracy (%)", "Gain (pp)"))
            off_ = OFFICIAL[OFFICIAL.year == FC_YEAR][["landkreis", "crop", "official"]]
            both = []
            for nm, F_ in ((f"Pre-registered test (v1), sealed {dfmt(V1.issued.iloc[0])}" if len(V1) else None, V1),
                           (f"{MLABEL[MODEL]}, sealed {dfmt(META.issued)}", F26_ALL)):
                if nm is None:
                    continue
                Q = select(F_, use_years=False).merge(off_, on=["landkreis", "crop"], how="inner")
                if len(Q):
                    both.append((nm, len(Q), acc(Q.y_pred, Q.official), acc(Q.y_hist, Q.official)))
            if both:
                st.markdown("###### Both sealed files, each against the district normal in its own file")
                B = pd.DataFrame(both, columns=["Sealed file", "Forecasts", "Accuracy (%)", "District normal (%)"])
                B["Gain (pp)"] = (B["Accuracy (%)"] - B["District normal (%)"]).map(fs)
                for c in ["Accuracy (%)", "District normal (%)"]:
                    B[c] = B[c].map(f1)
                table(B, numeric=("Forecasts", "Accuracy (%)", "District normal (%)", "Gain (pp)"))
    PR = load_csv("prelim_2026.csv")
    if not PR.empty:
        with panel("prelim", "teal"):
            st.markdown(f"##### First check against the preliminary official national yields ({plural(len(PR), 'crop')}, "
                        "all of Germany)")
            off = PR.set_index("crop").official_preliminary
            v15 = F_ALL[(F_ALL.year == FC_YEAR) & (F_ALL.model == "YieldFormer-DE-i (v15)")]

            def plain(f, col):
                return float((100 * (f.groupby("crop")[col].mean().reindex(off.index) / off - 1)).abs().mean())
            rows = [(MLABEL[MODEL], PR[MODEL].abs().mean(), plain(F26_ALL, "y_pred")),
                    (MLABEL["YieldFormer-DE-i (v15)"], PR["YieldFormer-DE-i (v15)"].abs().mean(), plain(v15, "y_pred")),
                    ("District normal", PR["Normal year"].abs().mean(), plain(F26_ALL, "y_hist"))]
            R = pd.DataFrame(rows, columns=["Forecast", "Area-weighted, as official (%)", "Plain mean of districts (%)"])
            lc, rc = st.columns([1.3, 1])
            with lc:
                table(R.assign(**{c: R[c].map(f1) for c in R.columns[1:]}), numeric=tuple(R.columns[1:]))
            ahead_w = R.iloc[0, 1] < R.iloc[2, 1]
            ahead_p = R.iloc[0, 2] < R.iloc[2, 2]
            rc.caption(f"Average error of the national yield. Weighted by crop area, as official national yields are, "
                       f"YieldFormer-DE is {'closer' if ahead_w else 'further away'} than the district normal; as a plain "
                       f"mean of districts it is {'closer' if ahead_p else 'further away'}. One season, preliminary "
                       "figures: not the registered test.")
            with rc.expander("Details"):
                st.caption("The preliminary figures were published in August 2026 (potatoes 24 Sep), before every 2026 "
                           "file was sealed. The final model was chosen by the written rule on earlier seasons, not by "
                           "this check. The registered test uses the district yields once Destatis publishes them.")
else:  # Compare
    part = st.radio("Benchmark", ["Simple averages and members", "Seasons & regions", "Published models",
                                  "JRC MARS 2025", "Poor years", "Computing cost"],
                    horizontal=True, key="cmp_part")
    if part == "Simple averages and members":
        early, late = E[E.year <= EARLY_END], E[E.year > EARLY_END]
        rows = [{"Model": MLABEL.get(m, dn(m)),
                 f"{FIRST_YEAR}–{EARLY_END} accuracy (%)": f1(acc(early[m], early.actual)) if len(early) else "–",
                 f"{EARLY_END + 1}–{LAST_YEAR} accuracy (%)": f1(acc(late[m], late.actual)) if len(late) else "–",
                 f"{yr_txt} accuracy (%)": f1(acc(E[m], E.actual)),
                 "Within ±10% (%)": f1(100 * np.mean(np.abs(E[m] / E.actual - 1) <= 0.1))} for m in MODELS + COMPARE]
        with panel("simple", "rose"):
            st.markdown(f"##### Accuracy on the same {plural(len(E), 'backtest forecast')} · "
                        f"{cname(crop) if crop != 'All crops' else 'all crops'} · {html.escape(scope)}")
            R = pd.DataFrame(rows)
            table(R, numeric=tuple(R.columns[1:]))
            st.caption("All models are scored on exactly the same rows. A single TFT is about as accurate as the ensemble; "
                       f"XGBoost on the same inputs comes close in {EARLY_END + 1}–{LAST_YEAR}. The ensemble was chosen "
                       f"by the rule written in advance and tested on {FIRST_YEAR}–{EARLY_END}; Seasons & regions shows "
                       "how each model ranks season by season.")
    elif part == "Seasons & regions":
        short = {MODEL: "YieldFormer-DE", "YieldFormer-DE-T (v17)": "TFT", "YieldFormer-DE-i (v15)": "iTransformer",
                 "YieldFormer-DE-P (v17)": "PatchTST", "XGBoost + memory (reference)": "XGBoost ref.",
                 "5-year average": "5-yr avg", "Long-term average": "Normal"}
        cols = list(dict.fromkeys(MODELS + ["XGBoost + memory (reference)", ref]))
        A = pd.DataFrame({m: acc_by(E, m, "year") for m in cols})
        best = A[MODELS].max(axis=1)
        within = {m: int((A[m] >= best - 0.5).sum()) for m in MODELS}
        gap = best - A[MODEL]
        # hindsight check: each season, use the network that was most accurate in the season before
        pick = A[MODELS].idxmax(axis=1).shift(1).dropna()
        R2 = E[E.year.isin(pick.index)]
        sw = pd.Series(np.nan, index=R2.index)
        for y_, m_ in pick.items():
            sw[R2.year == y_] = R2.loc[R2.year == y_, m_]
        acc_sw, acc_v, acc_t = (acc(x, R2.actual) for x in (sw, R2[MODEL], R2["YieldFormer-DE-T (v17)"]))
        span = f"{int(pick.index.min())}–{int(pick.index.max())}" if len(pick) else "–"
        with panel("seasonsummary", "sky"):
            st.markdown(f"##### Which network suits which season? · {html.escape(scope)}")
            c = st.columns(3)
            card(c[0], "Within 0.5 pp of the season's best", f"{within[MODEL]} of {len(A)}",
                 f"TFT {within['YieldFormer-DE-T (v17)']} · iTransformer {within['YieldFormer-DE-i (v15)']} · "
                 f"PatchTST {within['YieldFormer-DE-P (v17)']}", small=True)
            card(c[1], "Largest shortfall to the season's best", f"{gap.max():.1f} pp" if len(gap) else "–",
                 f"in {int(gap.idxmax())}" if len(gap) else "", small=True)
            card(c[2], "Switch to last season's most accurate", f"{acc_sw:.1f}%" if len(pick) else "–",
                 f"YieldFormer-DE {acc_v:.1f}% · TFT {acc_t:.1f}% ({span})" if len(pick) else "", small=True)
            n_last = int((A[MODELS].idxmin(axis=1) == MODEL).sum())
            st.caption("The most accurate network changes from season to season and is only known after harvest. "
                       f"YieldFormer-DE was the least accurate of the four networks in {n_last} of "
                       f"{plural(len(A), 'season')}.")
        with panel("memberstrip", "violet"):
            fig = go.Figure()
            fig.add_hline(y=0, line_color=BLUE, line_width=2, annotation_text="YieldFormer-DE",
                          annotation_position="top right", annotation_font_color=BLUE)
            for m_, letter in (("YieldFormer-DE-T (v17)", "T"), ("YieldFormer-DE-i (v15)", "i"),
                               ("YieldFormer-DE-P (v17)", "P")):
                fig.add_trace(go.Scatter(x=A.index, y=A[m_] - A[MODEL], mode="markers+text", text=[letter] * len(A),
                                         name=f"{short[m_]} member ({letter})", textfont=dict(color=BLUE, size=10),
                                         marker=dict(size=17, color="white", line=dict(color=BLUE, width=1.5)),
                                         hovertemplate="%{x}: %{y:+.1f} pp vs YieldFormer-DE<extra>" + short[m_]
                                                       + "</extra>"))
            fig.add_trace(go.Scatter(x=A.index, y=A["XGBoost + memory (reference)"] - A[MODEL], mode="markers",
                                     name="XGBoost + memory (reference)",
                                     marker=dict(size=10, color=GREY, symbol="x-thin", line=dict(color=GREY, width=2)),
                                     hovertemplate="%{x}: %{y:+.1f} pp vs YieldFormer-DE<extra>reference</extra>"))
            layout(fig, 330, title=dict(text="Each network against the ensemble, season by season", font=dict(size=14),
                                        x=0), yaxis=dict(ticksuffix=" pp", title="accuracy vs YieldFormer-DE"),
                   xaxis=dict(dtick=1, tickformat="d"))
            st.plotly_chart(fig, use_container_width=True, config=CFG)
            st.caption("Above the blue line: more accurate than the ensemble in that season; below: less.")
        with panel("seasontable", "rose"):
            st.markdown("##### Accuracy by season (%)")
            T = pd.DataFrame({"Season": A.index.astype(int)})
            for m_ in cols:
                T[short.get(m_, m_)] = A[m_].map(f1).to_numpy()
            T[f"Gain vs {RS} (pp)"] = (A[MODEL] - A[ref]).map(fs).to_numpy()
            top = A[MODELS].idxmax(axis=1)
            margin = A[MODELS].apply(lambda r: r.nlargest(2).diff().abs().iloc[-1], axis=1)
            T["Most accurate network"] = [short[m_] + (" (tie)" if g_ < 0.5 else "") for m_, g_ in zip(top, margin)]
            table(T, numeric=tuple(T.columns[1:-1]))
            st.caption("Tie: the lead over the next network is under 0.5 pp. XGBoost is a reference, never a "
                       "candidate.")
        with panel("regiontable", "teal"):
            ER = select(E_ALL, use_district=False, use_area=False)
            G = gains(ER, "region")
            no18 = (gains(ER[ER.year != DROUGHT_YEAR], "region", boot=0).gain
                    if years[0] <= DROUGHT_YEAR <= years[1] else None)
            per_crop = pd.DataFrame({c_: gains(ER[ER.crop == c_], "region", boot=0).gain for c_ in CROPS
                                     if (ER.crop == c_).any()})
            tft = acc_by(ER, "YieldFormer-DE-T (v17)", "region") - acc_by(ER, MODEL, "region")
            dist = ER.groupby("region").landkreis.nunique()
            rows = []
            for r_ in REGIONS:
                if r_ not in G.index:
                    continue
                g_ = G.loc[r_]
                row_ = {"Region": r_, "Districts": int(dist.get(r_, 0)), "Accuracy (%)": f1(g_.acc_m),
                        f"{RS} (%)": f1(g_.acc_r), "Gain (pp)": fs(g_.gain),
                        "90% range (pp)": f"{fs(g_.lo)} to {fs(g_.hi)}", "Better in": f"{int(g_.better)} of {int(g_.seasons)}"}
                if no18 is not None:
                    row_[f"Gain without {DROUGHT_YEAR} (pp)"] = fs(no18.get(r_))
                row_["Gain, every crop equal (pp)"] = fs(per_crop.loc[r_].mean()) if r_ in per_crop.index else "–"
                row_["TFT minus ensemble (pp)"] = fs(tft.get(r_))
                rows.append(row_)
            st.markdown(f"##### By region · {cname(crop) if crop != 'All crops' else 'all crops'} · {yr_txt}")
            st.markdown(f"<div class='sub'>{region_headline(G)}</div>", unsafe_allow_html=True)
            RT = pd.DataFrame(rows)
            table(RT, numeric=tuple(RT.columns[1:]))
            st.caption("Accuracy shows how predictable a region is; gain shows what the model adds there. Range: 90% "
                       "when whole seasons are resampled. The Region, State and District filters do not apply to this "
                       "table; the others do.")
    elif part == "Computing cost":
        J, S = load_csv("compute_jobs.csv"), load_csv("compute_models.csv").set_index("network")
        if J.empty:
            st.info("No job records yet: run scripts/79_compute_cost.py --logs <folder with the SLURM job logs>.")
        else:
            J = J[J.purpose.isin(["rolling 2010-2025", "2026 forecast"])]
            NETS = ["iTransformer", "TFT", "PatchTST"]
            REF_ = "XGBoost + memory (reference)"
            BOARD_W = {"NVIDIA L40": 300, "NVIDIA L40S": 350}             # maximum board power (NVIDIA data sheets)
            fin = J[J.network.isin(NETS)]
            gpu_h = fin.wall_s.sum() / 3600
            kwh = (fin.wall_s / 3600 * fin.gpu.map(BOARD_W).fillna(350)).sum() / 1000
            cap = J[(J.stage == "train") & (J.train_s >= 7150)]
            with panel("computesummary", "sky"):
                st.markdown("##### What it took to build YieldFormer-DE (v18)")
                c = st.columns(4)
                card(c[0], "GPU time, final model", f"{gpu_h:,.0f} h",
                     f"{fin.job.nunique()} job arrays · {len(fin)} runs", small=True)
                card(c[1], "Size of the three networks", f"{S.parameters.reindex(NETS).sum() / 1e6:.1f} M",
                     " + ".join(f"{S.parameters[n] / 1e6:.2f}" for n in NETS if n in S.index) + " M parameters",
                     small=True)
                card(c[2], "GPUs", " / ".join(g.replace("NVIDIA ", "") for g in sorted(fin.gpu.unique())),
                     "NVIDIA, 48 GB · GPU cluster", small=True)
                card(c[3], "Energy, upper bound", f"≤ {kwh:,.0f} kWh", "GPU hours × maximum board power", small=True)
                st.caption("One run = one season and one seed on one GPU: 16 seasons × 3 seeds for 2010–2025 plus 3 "
                           "runs for the 2026 forecast, each followed by fine-tuning on the 10 crops. A forecast "
                           "averages 9 fine-tuned networks (3 types × 3 seeds). The error memory and the average run "
                           "on a laptop CPU in seconds.")
            rows = []
            for n in NETS + [REF_]:
                g = J[J.network == n]
                if g.empty:
                    continue
                tr, ft = g[g.stage == "train"], g[g.stage == "finetune"]
                gp = g.gpu.str.replace("NVIDIA ", "").value_counts()
                rows.append({
                    "Network": n if n == REF_ else f"{n} (YieldFormer-DE-{ {'iTransformer': 'i', 'TFT': 'T', 'PatchTST': 'P'}[n]})",
                    "Parameters": f"{S.parameters[n]:,}" if n in S.index else "–",
                    "File per network": f"{S.checkpoint_bytes[n] / 1e6:.1f} MB" if n in S.index else "–",
                    "Runs": f"{len(tr)} + {len(ft)} fine-tuning" if len(ft) else f"{len(tr)}",
                    "GPU (runs)": " · ".join(f"{k} {v}" for k, v in gp.items()),
                    "Training run, median (min)": f1(tr.wall_s.median() / 60),
                    "Fine-tuning run, median (min)": f1(ft.wall_s.median() / 60) if len(ft) else "–",
                    "Stopped by the 120-min limit": f"{int((tr.train_s >= 7150).sum())} of {len(tr)}" if len(ft) else "–",
                    "GPU hours": f1(g.wall_s.sum() / 3600)})
            with panel("computetable", "rose"):
                st.markdown("##### Per network")
                CT = pd.DataFrame(rows)
                table(CT, numeric=("Parameters", "File per network", "Training run, median (min)",
                                   "Fine-tuning run, median (min)", "GPU hours"))
                st.caption("XGBoost + memory is a reference, not part of the final model. Training stops at 120 epochs, "
                           "after 15 epochs without improvement, or after 120 minutes; runs that ended earlier stopped "
                           "for one of the first two reasons (which one was not logged).")
            with panel("computechart", "violet"):
                H = (J.assign(h=J.wall_s / 3600, part=J.stage.map({"train": "Training", "finetune": "Fine-tuning"}))
                     .groupby(["network", "part"]).h.sum().unstack(fill_value=0).reindex(NETS + [REF_]).dropna(how="all"))
                fig = go.Figure()
                for part_, colr in (("Training", BLUE), ("Fine-tuning", LIGHT)):
                    if part_ in H:
                        fig.add_trace(go.Bar(y=H.index, x=H[part_], orientation="h", name=part_,
                                             marker_color=[GREY if i == REF_ else colr for i in H.index],
                                             hovertemplate="%{y}: %{x:.1f} GPU hours<extra>" + part_ + "</extra>"))
                tot = H.sum(axis=1)
                fig.add_trace(go.Scatter(x=tot + tot.max() * 0.02, y=H.index, mode="text", text=[f"{v:.0f} h" for v in tot],
                                         textposition="middle right", hoverinfo="skip", showlegend=False,
                                         textfont=dict(size=11, color=INK), cliponaxis=False))
                layout(fig, 260, barmode="stack", title=dict(text="GPU hours per network (2010–2025 runs and the 2026 "
                                                                  "forecast)", font=dict(size=14), x=0),
                       yaxis=dict(autorange="reversed"), xaxis=dict(ticksuffix=" h", range=[0, tot.max() * 1.15]))
                fig.update_layout(legend_traceorder="normal")
                st.plotly_chart(fig, use_container_width=True, config=CFG)
                st.caption("From the SLURM job logs of the training runs (scripts/79_compute_cost.py). Times are "
                           "wall-clock per job including data loading. The energy bound ignores CPU, memory and "
                           "cooling, and real GPU draw was lower. The filters do not apply here.")
    elif part == "Published models":
        P = load_csv("papers.csv")
        P = P[P.version == "v18"].copy()
        P["study"] = P.paper.str.replace(" (same districts)", "", regex=False).str.replace(" paper", "", regex=False)
        P["ours_better"] = P.verdict.eq("ours better")
        P["pub_below_avg"] = np.where(P.metric.eq("r2"), P.theirs < P.hist_avg, P.theirs > P.hist_avg)
        P["case"] = P.study + " · " + P.crop.map(cname) + np.where(P.study.eq("CropFusionNet"), "",
                                                                    " · " + P.years.astype(str))
        res = P.groupby("case").ours_better.agg(lambda s: "won" if s.all() else "lost" if not s.any() else "tie")
        won, lost, tied = (int((res == x).sum()) for x in ("won", "lost", "tie"))
        n_rows, n_below, n_won = len(P), int(P.pub_below_avg.sum()), int(P.ours_better.sum())
        if crop != "All crops":
            P = P[P.crop == crop]
        with panel("pubsummary", "sky"):
            st.markdown("##### Our three-network ensemble (without the error memory) retrained on each study's own split")
            st.caption("Each study has its own test years and districts: the Years, Region, State and District filters do not "
                       "apply here; the Crop filter does. Not a blind test: the studies' test years were development years "
                       "of this project.")
            c = st.columns(3)
            card(c[0], "Cases won (all crops)", f"{won} of {len(res)}",
                 [f"{lost} lost, {tied} tied", "case = study × crop (× test year)"], small=True)
            card(c[1], "Measures won (all crops)", f"{n_won} of {n_rows}",
                 "R², RMSE, MAE or MAPE, as each study reports", small=True)
            card(c[2], "Published scores worse than the plain average", f"{n_below} of {n_rows}",
                 "long-term district average, same years and districts", small=True)
        with panel("pubtable", "rose"):
            if P.empty:
                st.info(f"None of the studies reports {cname(crop)}.")
            else:
                names = {"r2": "R² (higher is better)", "rmse": "RMSE, dt/ha (lower is better)",
                         "mae": "MAE, dt/ha (lower is better)", "mape": "MAPE, % (lower is better)"}
                P = P.assign(o=P.crop.map(ORDER), mo=P.metric.map({"r2": 0, "rmse": 1, "mae": 2, "mape": 3}))
                P = P.sort_values(["study", "o", "years", "mo"])
                dig = np.where(P.metric.eq("r2"), 2, 1)
                yrs = (P.years.astype(str).str.replace("-", "–", regex=False)
                       .str.replace(r" pooled incl\..*", " pooled (as their code)", regex=True))
                out = pd.DataFrame({"Study": P.study, "Crop": P.crop.map(cname), "Test years": yrs,
                                    "Measure": P.metric.map(names),
                                    "Published": [f"{x:.{d}f}" for x, d in zip(P.theirs, dig)],
                                    "Ours": [f"{x:.{d}f}" for x, d in zip(P.ours_equal_protocol, dig)],
                                    "Plain average": [f"{x:.{d}f}" for x, d in zip(P.hist_avg, dig)],
                                    "Better": np.where(P.ours_better, "ours", "published"),
                                    "Case": P.case.map(res)})
                table(out, numeric=("Published", "Ours", "Plain average"))
                st.caption("CropFusionNet is scored in two ways (their test years, and pooled 2019–24 as their code does); "
                           "both rows form one case per crop. CropFusionNet remains clearly better for silage maize.")
    elif part == "JRC MARS 2025":
        MA = load_csv("mars_2025.csv")
        six = ["winter_wheat", "winter_barley", "summer_barley", "rye", "triticale", "winter_rapeseed"]
        sel = MA if crop == "All crops" else MA[MA.crop == crop]
        with panel("marssummary", "sky"):
            st.markdown("##### Not like-for-like: our 2025 backtest uses weather up to harvest; the JRC MARS bulletin "
                        "is issued before harvest")
            st.caption("One season (2025), national yields, census-area weighted. The Years, Region, State and District filters "
                       "do not apply here; the Crop filter does. One season cannot rank the two systems.")
            if sel.empty:
                st.info(f"JRC MARS published no 2025 forecast for {cname(crop)}.")
            else:
                c = st.columns(3)
                groups = [(f"{plural(len(MA), 'crop')}" if crop == "All crops" else cname(crop), sel),
                          ("Cereals and rapeseed", MA[MA.crop.isin(six)]),
                          ("Maize, sugar beet, potatoes", MA[~MA.crop.isin(six)])]
                for i, (lab, g) in enumerate(groups):
                    ours, mars = g.ens_err_official_pct.abs().mean(), g.mars_err_pct.abs().mean()
                    card(c[i], f"{lab} · mean error", f"{ours:.1f}% ours",
                         f"JRC MARS {mars:.1f}% · {'ours closer' if ours < mars else 'MARS closer'}", small=True)
        if not sel.empty:
            order = [c for c in CROPS if c in set(MA.crop)]
            D = MA.set_index("crop").reindex(order)
            fig = go.Figure()
            yl = [cname(c) for c in order]
            for c_, y_ in zip(order, yl):                          # connector: how far apart the two systems are
                fig.add_trace(go.Scatter(x=[D.mars_err_pct[c_], D.ens_err_official_pct[c_]], y=[y_, y_], mode="lines",
                                         line=dict(color="#c9ced6", width=3), hoverinfo="skip", showlegend=False,
                                         opacity=1.0 if crop in ("All crops", c_) else 0.25))
            op = [1.0 if crop in ("All crops", c) else 0.25 for c in order]
            fig.add_trace(go.Scatter(x=D.mars_err_pct, y=yl, mode="markers", name="JRC MARS (last bulletin)",
                                     marker=dict(color=GREY, size=12, symbol="diamond", opacity=op),
                                     hovertemplate="%{y}: %{x:+.1f}%<extra>JRC MARS</extra>"))
            fig.add_trace(go.Scatter(x=D.ens_err_official_pct, y=yl, mode="markers", name=MODEL,
                                     marker=dict(color=BLUE, size=12, opacity=op), customdata=D.b5_err_official_pct,
                                     hovertemplate="%{y}: %{x:+.1f}% (5-year average %{customdata:+.1f}%)"
                                                   "<extra>YieldFormer-DE</extra>"))
            fig.add_vline(x=0, line_color=GREY, line_width=1)
            layout(fig, 360, title=dict(text="Error of the national 2025 yield vs the official figure",
                                        font=dict(size=14), x=0), yaxis=dict(autorange="reversed"),
                   xaxis=dict(ticksuffix="%"))
            with panel("marschart", "violet"):
                st.plotly_chart(fig, use_container_width=True, config=CFG)
                st.caption("MARS: last bulletin before harvest (August 2025 for winter crops, September for the others; "
                           "no oats forecast).")
    else:
        g = E.groupby(["crop", "year"])[["actual", model, ref, "Long-term average"]].mean()
        real = 100 * (g.actual / g["Long-term average"] - 1)
        poor = real <= -10
        rules = []
        for lab, thr in (("below normal", 0), ("≥5% below normal (card rule)", -5), ("≥10% below normal", -10)):
            for m in (model, ref):
                e_ = 100 * (g[m] / g["Long-term average"] - 1)
                sig = e_ < thr if thr == 0 else e_ <= thr
                rules.append({"Signal when the forecast is": lab, "Forecast by": MLABEL.get(m, m),
                              "Poor crop-years detected": f"{int((poor & sig).sum())} of {int(poor.sum())}",
                              "False alarms": f"{int((~poor & sig).sum())} of {int((~poor).sum())}",
                              "… of them in years 5–10% below normal": f"{int((~poor & sig & (real <= -5)).sum())}"})
        with panel("poortable", "rose"):
            st.markdown(f"##### Poor crop-years (mean official yield ≥10% below normal) · {html.escape(scope)} · {yr_txt}")
            table(pd.DataFrame(rules))
            st.caption("A crop-year is one crop in one year, averaged over the districts in the selection. The backtest forecast is "
                       "made at harvest from the season's data: detection before the statistics appear, not an early "
                       "warning. Many false alarms fall in years 5–10% below normal, i.e. poor but not past the 10% line.")
        if poor.any():
            with panel("poorchart", "violet"):
                P = pd.DataFrame({"real": real[poor], "est": (100 * (g[model] / g["Long-term average"] - 1))[poor],
                                  "refv": (100 * (g[ref] / g["Long-term average"] - 1))[poor]}).reset_index()
                P = P.assign(o=P.crop.map(ORDER)).sort_values(["o", "year"])
                lab_ = [f"{cname(c)} {y}" for c, y in zip(P.crop, P.year)]
                fig = go.Figure()
                for col, nm, colr in (("real", "Official", INK), ("refv", dn(ref), GREY), ("est", "Forecast (backtest)", BLUE)):
                    fig.add_trace(go.Scatter(x=P[col], y=lab_, mode="markers", name=nm, marker=dict(color=colr, size=10),
                                             hovertemplate="%{y}: %{x:+.1f}% vs normal<extra>" + nm + "</extra>"))
                fig.add_vline(x=-10, line_color=RED, line_dash="dot", annotation_text="poor: −10%",
                              annotation_position="top")
                fig.add_vline(x=-5, line_color=GREY, line_dash="dot", annotation_text="detected: −5%",
                              annotation_position="top")
                layout(fig, 110 + 24 * len(P), yaxis=dict(autorange="reversed"), xaxis=dict(ticksuffix="%"),
                       title=dict(text="Each poor crop-year: yield vs normal", font=dict(size=14), x=0),
                       margin=dict(l=8, r=8, t=60, b=8))
                st.plotly_chart(fig, use_container_width=True, config=CFG)

# --------------------------------------------------------------------------- footer
with st.expander("About this dashboard"):
    st.markdown(
        f"- **What it shows.** Official district yields from Destatis, the backtest forecasts of {MODEL} for 2010–2025 "
        "(each season forecast by a model trained only on earlier seasons) and the sealed 2026 forecast.\n"
        f"- **{MODEL}** is the average of three transformer networks (TFT, iTransformer, PatchTST) plus a district "
        "error memory. XGBoost + memory is a reference only, not a candidate model.\n"
        "- **Accuracy** = 100 minus the mean absolute percentage error. **Normal** = the district's long-term mean "
        "yield plus a trend if one is clear. **pp** = percentage points.\n"
        "- Methods, all versions and every comparison are described in the written project report.")

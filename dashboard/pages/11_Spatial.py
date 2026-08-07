"""Spatial view — where in Germany is the prediction good or bad?

One question, one map. The overlay selector chooses what the single NUTS-3
choropleth encodes; a four-metric KPI row summarises it. Ranked districts,
the pooled variance-explained figure, the per-district-R² diagnostic, the
leave-one-Landkreis-out study and the per-district yield history are all
still here, one click away inside collapsed expanders.

Merges the former `11_Spatial.py` (per-district skill + LOO XGBoost) and
`18_Geographic.py` (actual / predicted / residual choropleth). Both pages
carried a byte-identical `_load_polygons()` and rendered the same shapefile,
so they are unified here behind one overlay selector.

On the deliberate absence of per-district R²
--------------------------------------------
`results/metrics/spatial_per_district.csv` scores every district on the two
test years, so every one of its 8,948 rows has n=2. At n=2 the R² denominator
collapses to (y1 - y2)^2 / 2 — the squared gap between two harvests, which is
a noise term, not a variance estimate. When two consecutive years happen to
yield alike the denominator approaches zero and R² diverges: 69.9% of rows sit
below -1 and the floor is -1.4e7.

The ranking that produces is not merely noisy, it is misleading. R² and RMSE
come out of the same folds, so a district genuinely predicted badly should rank
badly on both — but across all 60 model-crop pairs the ten districts with the
worst R² sit at the 58th RMSE percentile on median, and for pairs such as
dlinear x oats at the 26th. Concretely, xgboost x winter_wheat puts DEG0P third
worst in the country at R² = -1521 while its RMSE sits at the 35th percentile:
a district predicted better than 65% of its peers. So per-district R² is not
mapped, not ranked, and not shown as a KPI here.

What replaces it:
  * Per-district overlays and tables use RMSE / MAE, which are well defined
    at any n and stay in dt/ha.
  * Variance-explained is reported once, pooled across districts (N ~ 430
    district-years), where the denominator is a real yield variance.
  * The raw per-district R² survives only inside a collapsed diagnostic whose
    purpose is to show why it cannot be used.

Data sources:
  data/processed/dataset.parquet          actual yields
  results/metrics/spatial_per_district.csv  per-district RMSE/MAE, 6 models
  results/metrics/loo_xgboost.csv           leave-one-Landkreis-out folds
  data/raw/geo/NUTS_RG_01M_2021_4326_LEVL_3.shp
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from branca.colormap import LinearColormap
from streamlit_folium import st_folium

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _model_template import _predict_test_set
from utils import (CHECKPOINTS_DIR, METRICS_DIR, MODELS, PROCESSED,
                   PROJECT_ROOT, kpi, landkreis_name_map,
                   load_climatology_baseline, page_header)

st.set_page_config(page_title="Spatial — Crop Yield", layout="wide")
page_header(
    "Spatial view",
    "Where in Germany is the prediction good or bad?",
)

# Colour used for districts that carry no yield series at all, and for those
# that do but have no value under the current selection. Both must be
# unmistakably different from every colormap value below.
C_OUTSIDE = "#f2f2f2"   # not in the district panel at all
C_NOVALUE = "#8a8a8a"   # in the panel, nothing to show for this selection


# --------------------------------------------------------------------------- #
# Cached loaders
# --------------------------------------------------------------------------- #
@st.cache_data(ttl=3600)
def _load_polygons() -> gpd.GeoDataFrame:
    """401 German NUTS-3 polygons, lightly simplified (halves the GeoJSON)."""
    shp = PROJECT_ROOT / "data" / "raw" / "geo" / "NUTS_RG_01M_2021_4326_LEVL_3.shp"
    if not shp.exists():
        return gpd.GeoDataFrame(columns=["landkreis", "name", "geometry"],
                                geometry="geometry", crs="EPSG:4326")
    g = gpd.read_file(shp)
    g = g[g["CNTR_CODE"] == "DE"][["NUTS_ID", "NUTS_NAME", "geometry"]].copy()
    g["geometry"] = g.geometry.simplify(0.003, preserve_topology=True)
    return g.rename(columns={"NUTS_ID": "landkreis", "NUTS_NAME": "name"})


@st.cache_data(ttl=3600)
def _yield_panel() -> pd.DataFrame:
    """One row per (crop, landkreis, year).

    The processed parquet is daily (57 MB, ~16 M rows); the map only ever needs
    the annual yield, so collapse once and cache rather than re-grouping the
    daily frame on every widget change.
    """
    if not PROCESSED.exists():
        return pd.DataFrame(columns=["crop", "landkreis", "year", "yield_dt_per_ha"])
    df = pd.read_parquet(
        PROCESSED, columns=["landkreis", "year", "crop", "yield_dt_per_ha"])
    return (df.groupby(["crop", "landkreis", "year"], observed=True)
              ["yield_dt_per_ha"].first().reset_index())


@st.cache_data(ttl=600)
def _per_district() -> pd.DataFrame:
    p = METRICS_DIR / "spatial_per_district.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


@st.cache_data(ttl=600)
def _loo() -> pd.DataFrame:
    p = METRICS_DIR / "loo_xgboost.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


@st.cache_data(ttl=900, show_spinner="Running model forward pass…")
def _predictions(model: str, crop: str) -> tuple[pd.DataFrame, str | None]:
    """Cached wrapper around the shared predictor.

    `_model_template._predict_test_set` carries no cache of its own, so an
    uncached call re-ran a full forward pass (~35 s for xgboost) on every
    widget interaction.
    """
    preds, err = _predict_test_set(model, crop)
    return preds, (err["error"] if err else None)


@st.cache_data(ttl=3600)
def _test_years() -> list[int]:
    """The test split, always read from configs/data.yaml.

    Deliberately has no literal fallback. A hardcoded {2022, 2023} is what made
    the prediction and residual overlays unreachable once the config moved to
    [2024, 2025]; a fallback would silently recreate that. If the config cannot
    be read the page says so instead of guessing.
    """
    from src.utils import load_data_config
    return [int(y) for y in load_data_config()["splits"]["temporal"]["test_years"]]


@st.cache_data(ttl=600)
def _pooled_r2(model: str, crop: str, years: tuple[int, ...]) -> dict | None:
    """Variance explained over the pooled district-year sample.

    Per-district R² is undefined at n=2; pooling restores a denominator that is
    an actual yield variance. SSE is reconstructed exactly from the per-district
    RMSE (SSEᵢ = nᵢ·RMSEᵢ²), and districts are kept only where the panel truth
    count matches the metric's n, so numerator and denominator cover identical
    observations.
    """
    metrics, pnl = _per_district(), _yield_panel()
    if metrics.empty or pnl.empty:
        return None
    s = metrics[(metrics["model"] == model) & (metrics["crop"] == crop)]
    if s.empty:
        return None
    truth = (pnl[(pnl["crop"] == crop) & (pnl["year"].isin(years))]
             .dropna(subset=["yield_dt_per_ha"]))
    cnt = truth.groupby("landkreis").size().rename("n_truth")
    s = s.merge(cnt, on="landkreis", how="left")
    s = s[s["n_truth"] == s["n"]]
    if s.empty:
        return None
    t = truth[truth["landkreis"].isin(set(s["landkreis"]))]["yield_dt_per_ha"]
    sse = float((s["n"] * s["rmse"] ** 2).sum())
    sst = float(((t - t.mean()) ** 2).sum())
    n_obs = int(s["n"].sum())
    if sst <= 0 or n_obs != len(t):
        return None
    return {"r2": 1 - sse / sst, "rmse": float(np.sqrt(sse / n_obs)),
            "n_obs": n_obs, "n_districts": int(len(s))}


polys = _load_polygons()
if polys.empty:
    st.error("NUTS-3 shapefile missing. Run `scripts/00_build_geo.py`.")
    st.stop()

panel = _yield_panel()
pd_df = _per_district()
loo_df = _loo()
name_map = landkreis_name_map()
try:
    test_years = _test_years()
except Exception as e:  # noqa: BLE001
    st.error(f"Cannot read the test split from `configs/data.yaml`: {e}")
    st.stop()

panel_districts = set(panel["landkreis"].unique()) if not panel.empty else set()

# --------------------------------------------------------------------------- #
# Controls
# --------------------------------------------------------------------------- #
OVERLAYS = [
    "Actual yield",
    "Model prediction",
    "Residual (pred − actual)",
    "Test RMSE per district",
    "Test MAE per district",
    "LOO RMSE (XGBoost)",
    "LOO RMSE penalty (LOO − in-training)",
]
NEEDS_PRED = {"Model prediction", "Residual (pred − actual)"}
NEEDS_YEAR = {"Actual yield"} | NEEDS_PRED
NEEDS_METRICS = {"Test RMSE per district", "Test MAE per district"}
NEEDS_LOO = {"LOO RMSE (XGBoost)", "LOO RMSE penalty (LOO − in-training)"}
NEEDS_MODEL = NEEDS_PRED | NEEDS_METRICS

crops = sorted(panel["crop"].unique().tolist()) if not panel.empty else ["winter_wheat"]

c1, c2, c3, c4 = st.columns([1.5, 1, 1, 1])
with c1:
    overlay = st.selectbox("Overlay", OVERLAYS, key="sp_overlay")
with c2:
    crop = st.selectbox(
        "Crop", crops,
        index=crops.index("winter_wheat") if "winter_wheat" in crops else 0,
        key="sp_crop")
with c3:
    if overlay in NEEDS_MODEL:
        avail = [m for m in MODELS
                 if (CHECKPOINTS_DIR / f"{m}.pipeline.json").exists()] or list(MODELS)
        if overlay in NEEDS_METRICS and not pd_df.empty:
            avail = [m for m in avail if m in set(pd_df["model"].unique())] or avail
        model = st.selectbox("Model", avail,
                             index=avail.index("xgboost") if "xgboost" in avail else 0,
                             key="sp_model")
    else:
        model = "xgboost"
        st.selectbox("Model", ["— not used by this overlay —"], disabled=True,
                     key="sp_model_off")
with c4:
    if overlay in NEEDS_YEAR:
        # Prediction and residual exist only on the test split. Offering just
        # those years makes the old unreachable-overlay failure structurally
        # impossible rather than warning about it after the fact.
        yrs = (sorted(y for y in panel["year"].unique()) if not panel.empty else [])
        opts = [y for y in yrs if y in set(test_years)] if overlay in NEEDS_PRED else yrs
        opts = opts or list(test_years)
        year = st.select_slider("Year", opts, value=opts[-1], key="sp_year")
    else:
        year = None
        _tw = f"test window {'–'.join(map(str, test_years))}"
        st.select_slider("Year", [_tw], value=_tw, disabled=True,
                         key="sp_year_off")

# --------------------------------------------------------------------------- #
# Staleness — fires only for the overlays that actually read a metric CSV, so
# the default "Actual yield" view (which reads the parquet, not the metrics)
# stays clean. A firing warning is never collapsed.
# --------------------------------------------------------------------------- #
PER_DISTRICT_CSV = METRICS_DIR / "spatial_per_district.csv"
LOO_CSV = METRICS_DIR / "loo_xgboost.csv"


def _stale_against_checkpoints(*paths: Path) -> list[tuple[Path, float]]:
    """Metric files older than the newest checkpoint on disk."""
    ckpts = [*CHECKPOINTS_DIR.glob("*.pt"), *CHECKPOINTS_DIR.glob("*.ubj")]
    if not ckpts:
        return []
    newest_mt = max(p.stat().st_mtime for p in ckpts)
    return [(p, p.stat().st_mtime) for p in paths
            if p.exists() and p.stat().st_mtime < newest_mt]


_sources_used: list[Path] = []
if overlay in NEEDS_METRICS:
    _sources_used.append(PER_DISTRICT_CSV)
if overlay in NEEDS_LOO:
    _sources_used += [LOO_CSV, PER_DISTRICT_CSV]
_stale_now = _stale_against_checkpoints(*dict.fromkeys(_sources_used))
if _stale_now:
    _ck = [*CHECKPOINTS_DIR.glob("*.pt"), *CHECKPOINTS_DIR.glob("*.ubj")]
    _newest = max(_ck, key=lambda p: p.stat().st_mtime)
    _nmt = _newest.stat().st_mtime
    st.warning(
        "**Stale — this overlay is not from the models currently on disk.** "
        f"Newest checkpoint `{_newest.name}` is "
        f"{datetime.fromtimestamp(_nmt):%Y-%m-%d %H:%M}:\n\n"
        + "\n".join(
            f"- `{p.name}` — {datetime.fromtimestamp(mt):%Y-%m-%d %H:%M}, "
            f"{(_nmt - mt) / 86400:.1f} days older"
            for p, mt in _stale_now)
        + "\n\nRegenerate with `python scripts/06_spatial_eval.py` / "
          "`python scripts/06b_loo_xgboost.py`."
    )

# --------------------------------------------------------------------------- #
# Resolve the overlay to a per-district value column
# --------------------------------------------------------------------------- #
values = pd.DataFrame(columns=["landkreis", "value"])
extra: pd.DataFrame | None = None      # optional tooltip / table columns
label = ""
unit = "dt/ha"
diverging = False
higher_is_worse = False
err_msg: str | None = None

if overlay == "Actual yield":
    values = (panel[(panel["crop"] == crop) & (panel["year"] == year)]
              [["landkreis", "yield_dt_per_ha"]]
              .rename(columns={"yield_dt_per_ha": "value"}))
    label = f"Actual yield — {crop}, {year}"

elif overlay in NEEDS_PRED:
    preds, err_msg = _predictions(model, crop)
    if not err_msg and not preds.empty:
        sub = preds[preds["year"] == year]
        col = "y_pred" if overlay == "Model prediction" else "residual"
        values = (sub.groupby("landkreis")[col].first().reset_index()
                  .rename(columns={col: "value"}))
        extra = (sub.groupby("landkreis")[["y_true", "y_pred"]].first()
                 .reset_index())
        diverging = overlay.startswith("Residual")
        kind = "prediction" if overlay == "Model prediction" else "residual"
        label = f"{model.upper()} {kind} — {crop}, {year}"

elif overlay in NEEDS_METRICS:
    col = "rmse" if "RMSE" in overlay else "mae"
    sub = pd_df[(pd_df["model"] == model) & (pd_df["crop"] == crop)] \
        if not pd_df.empty else pd.DataFrame()
    if not sub.empty:
        values = sub[["landkreis", col]].rename(columns={col: "value"})
        extra = sub[["landkreis", "rmse", "mae", "n"]]
        higher_is_worse = True
        label = f"{model.upper()} test {col.upper()} — {crop}, {'/'.join(map(str, test_years))}"

elif overlay in NEEDS_LOO:
    lsub = loo_df[loo_df["crop"] == crop] if not loo_df.empty else pd.DataFrame()
    if not lsub.empty:
        base = pd_df[(pd_df["model"] == "xgboost") & (pd_df["crop"] == crop)] \
            if not pd_df.empty else pd.DataFrame()
        merged = lsub[["landkreis", "rmse", "mae", "n"]].rename(
            columns={"rmse": "rmse_loo", "mae": "mae_loo"})
        if not base.empty:
            merged = merged.merge(base[["landkreis", "rmse"]].rename(
                columns={"rmse": "rmse_in_train"}), on="landkreis", how="left")
            merged["penalty"] = merged["rmse_loo"] - merged["rmse_in_train"]
        else:
            merged["rmse_in_train"] = np.nan
            merged["penalty"] = np.nan
        col = "rmse_loo" if overlay == "LOO RMSE (XGBoost)" else "penalty"
        values = merged[["landkreis", col]].rename(columns={col: "value"})
        extra = merged
        diverging = col == "penalty"
        higher_is_worse = True
        label = ("XGBoost LOO RMSE" if col == "rmse_loo"
                 else "XGBoost LOO RMSE penalty (positive = worse held out)")
        label += f" — {crop}"

values = values.dropna(subset=["value"]) if not values.empty else values

# --------------------------------------------------------------------------- #
# Map
# --------------------------------------------------------------------------- #
if err_msg:
    st.error(f"{model.upper()}: {err_msg}")
elif values.empty:
    st.info(f"No data for this overlay under {crop}"
            + (f" x {model}" if overlay in NEEDS_MODEL else "") + ".")

joined = polys.merge(values, on="landkreis", how="left")
if extra is not None and not extra.empty:
    joined = joined.merge(extra.drop(columns=["value"], errors="ignore"),
                          on="landkreis", how="left")

# Three-way status so an absent district never reads as a low value.
joined["status"] = np.where(
    joined["value"].notna(), "scored",
    np.where(joined["landkreis"].isin(panel_districts),
             "panel_no_value", "outside_panel"))

vals = joined["value"].dropna()
if not vals.empty:
    if diverging:
        span = float(np.nanpercentile(np.abs(vals), 95)) or 1.0
        vmin, vmax = -span, span
        cmap = LinearColormap(["#2166ac", "#f7f7f7", "#b2182b"],
                              vmin=vmin, vmax=vmax)
    else:
        vmin = float(np.nanpercentile(vals, 5))
        vmax = float(np.nanpercentile(vals, 95))
        if vmax <= vmin:
            vmin, vmax = float(vals.min()), float(vals.min()) + 1e-6
        palette = (["#ffffcc", "#fd8d3c", "#bd0026"] if higher_is_worse
                   else ["#f7fcb9", "#78c679", "#005a32"])
        cmap = LinearColormap(palette, vmin=vmin, vmax=vmax)
    cmap.caption = f"{label} [{unit}]"

    def _hex(v: float) -> str:
        return cmap.rgb_hex_str(float(np.clip(v, vmin, vmax)))

    joined["_color"] = [
        _hex(v) if pd.notna(v) else (C_NOVALUE if s == "panel_no_value" else C_OUTSIDE)
        for v, s in zip(joined["value"], joined["status"])
    ]
else:
    cmap = None
    joined["_color"] = np.where(joined["status"] == "panel_no_value",
                                C_NOVALUE, C_OUTSIDE)

# Map frame is string-formatted for display; `joined` stays numeric for tables.
geo = joined.copy()
geo["value_txt"] = [f"{v:,.2f} {unit}" if pd.notna(v) else
                    ("no value for this selection" if s == "panel_no_value"
                     else f"outside the {len(panel_districts)}-district panel")
                    for v, s in zip(geo["value"], geo["status"])]

tip_fields = ["name", "landkreis", "value_txt"]
tip_alias = ["District", "NUTS-3", label.split("—")[0].strip() or "Value"]
for fld, alias in [("y_true", "Actual"), ("y_pred", "Predicted"),
                   ("rmse", "RMSE"), ("mae", "MAE"),
                   ("rmse_loo", "LOO RMSE"), ("rmse_in_train", "RMSE in training"),
                   ("n", "n (test years)")]:
    if fld in geo.columns and fld not in tip_fields:
        # Render as text: a raw NaN reaches the tooltip as "null", which reads
        # like a value of zero.
        geo[fld] = [f"{v:,.2f}" if pd.notna(v) else "—" for v in geo[fld]]
        tip_fields.append(fld)
        tip_alias.append(alias)

m = folium.Map(location=[51.1657, 10.4515], zoom_start=6, tiles="cartodbpositron")
folium.GeoJson(
    geo[tip_fields + ["_color", "status", "geometry"]].__geo_interface__,
    style_function=lambda f: {
        "fillColor": f["properties"]["_color"],
        "color": "#4d4d4d" if f["properties"]["status"] == "outside_panel" else "white",
        "weight": 0.8 if f["properties"]["status"] == "outside_panel" else 0.4,
        "dashArray": "3,3" if f["properties"]["status"] == "outside_panel" else "0",
        "fillOpacity": 0.35 if f["properties"]["status"] == "outside_panel" else 0.8,
    },
    highlight_function=lambda f: {"weight": 2.0, "color": "#222222"},
    tooltip=folium.features.GeoJsonTooltip(
        fields=tip_fields, aliases=tip_alias, localize=True, max_width=320),
).add_to(m)
if cmap is not None:
    cmap.add_to(m)

n_out = int((geo["status"] == "outside_panel").sum())
n_nov = int((geo["status"] == "panel_no_value").sum())
m.get_root().html.add_child(folium.Element(f"""
<div style="position:fixed;bottom:18px;left:18px;z-index:9999;background:rgba(255,255,255,.92);
 border:1px solid #bbb;border-radius:4px;padding:7px 10px;font:12px sans-serif;line-height:1.6">
  <span style="display:inline-block;width:13px;height:13px;background:{C_NOVALUE};
   border:1px solid #666;vertical-align:-2px"></span> No value for this selection ({n_nov})<br>
  <span style="display:inline-block;width:13px;height:13px;background:{C_OUTSIDE};
   border:1px dashed #4d4d4d;vertical-align:-2px"></span> Outside the panel ({n_out})
</div>"""))
st_folium(m, width=None, height=600, returned_objects=[])

# --------------------------------------------------------------------------- #
# PRIMARY 2 — headline numbers for whatever the map is showing
# --------------------------------------------------------------------------- #
if not values.empty:
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        kpi("Districts shown", f"{len(values)}")
    with k2:
        kpi("Median", f"{values['value'].median():,.2f} {unit}")
    with k3:
        kpi("IQR", f"{values['value'].quantile(.25):,.1f} – "
                   f"{values['value'].quantile(.75):,.1f}")
    with k4:
        kpi("Panel coverage", f"{len(values)} / {len(panel_districts)}")

# --------------------------------------------------------------------------- #
# SECONDARY — ranked districts
# --------------------------------------------------------------------------- #
if not values.empty:
    with st.expander("Ranked districts — best and worst 10", expanded=False):
        tbl = joined.dropna(subset=["value"])[["name", "landkreis", "value"]].copy()
        for c in ("rmse", "mae", "n", "y_true", "y_pred", "rmse_loo",
                  "rmse_in_train"):
            # Skip an extra that just restates the mapped value in another column.
            if c in joined.columns and not joined[c].equals(joined["value"]):
                tbl[c] = joined.loc[tbl.index, c]
        vcol = label.split("—")[0].strip() or "Value"
        tbl = tbl.rename(columns={"name": "District", "landkreis": "NUTS-3",
                                  "value": vcol})
        hi_label = "Largest" if not higher_is_worse else "Worst (largest error)"
        lo_label = "Smallest" if not higher_is_worse else "Best (smallest error)"
        t1, t2 = st.columns(2)
        with t1:
            st.markdown(f"##### {hi_label}")
            st.dataframe(tbl.nlargest(10, vcol), hide_index=True, width="stretch")
        with t2:
            st.markdown(f"##### {lo_label}")
            st.dataframe(tbl.nsmallest(10, vcol), hide_index=True, width="stretch")

# --------------------------------------------------------------------------- #
# SECONDARY — pooled variance explained, and why per-district R² is withheld
#
# Both live in one expander because Streamlit forbids nesting expanders and
# the diagnostic is the justification for the pooled figure, not a separate
# topic.
# --------------------------------------------------------------------------- #
if overlay in NEEDS_METRICS and not pd_df.empty and not panel.empty:
    with st.expander("Skill diagnostics — pooled variance explained, and why "
                     "per-district R² is withheld", expanded=False):
        pooled = _pooled_r2(model, crop, tuple(test_years))
        if pooled is None:
            st.info("Not enough aligned district-years to pool a "
                    "variance-explained figure.")
        else:
            _clim = load_climatology_baseline()
            _clim_r2 = None
            if not _clim.empty and "crop" in _clim.columns:
                _row = _clim[_clim["crop"] == crop]
                if not _row.empty:
                    _clim_r2 = float(_row["r2"].iloc[0])
            p1, p2, p3, p4 = st.columns(4)
            with p1:
                kpi("Pooled R²", f"{pooled['r2']:.3f}",
                    delta=(f"{pooled['r2'] - _clim_r2:+.3f} vs climatology"
                           if _clim_r2 is not None else None))
            with p2:
                kpi("Climatology baseline R²",
                    f"{_clim_r2:.3f}" if _clim_r2 is not None else "—")
            with p3:
                kpi("Pooled RMSE", f"{pooled['rmse']:,.2f} {unit}")
            with p4:
                kpi("Sample", f"n = {pooled['n_obs']} district-years")
            st.caption(
                f"1 − Σ nᵢ·RMSEᵢ² / Σ(y − ȳ)² over {pooled['n_districts']} "
                f"districts × {len(test_years)} test years. Pooling is what "
                "makes the denominator a real yield variance; the same "
                "quantity computed district-by-district is undefined at n=2 "
                "(see below). The climatology column is the learn-nothing "
                "reference from `results/metrics/climatology_baseline.csv` — "
                "it emits each district's train-fitted baseline and no "
                "anomaly."
            )

        st.markdown("---")
        st.markdown("##### Why per-district R² is withheld")
        s = pd_df[(pd_df["model"] == model) & (pd_df["crop"] == crop)].copy()
        if s.empty or "r2" not in s.columns:
            st.write("No per-district rows for this selection.")
        else:
            s["rmse_pct"] = s["rmse"].rank(pct=True)
            worst = s.nsmallest(10, "r2")
            d1, d2, d3 = st.columns(3)
            with d1:
                kpi("Rows at n=2", f"{int((s['n'] == 2).sum())} / {len(s)}")
            with d2:
                kpi("Share with R² < −1", f"{(s['r2'] < -1).mean():.1%}")
            with d3:
                kpi("10 worst by R²: median RMSE percentile",
                    f"{worst['rmse_pct'].median():.0%}")
            s["flag"] = np.where(s.index.isin(worst.index),
                                 "10 worst by R²", "other districts")
            fig = px.scatter(
                s, x="rmse", y="r2", color="flag", hover_data=["landkreis", "n"],
                color_discrete_map={"10 worst by R²": "#b2182b",
                                    "other districts": "#9ecae1"},
                labels={"rmse": f"RMSE ({unit}) — actual error",
                        "r2": "Reported per-district R²", "flag": ""},
                title="The two rankings disagree")
            fig.update_yaxes(range=[max(-60.0, float(s["r2"].min())), 1.2])
            fig.add_hline(y=0, line_dash="dot", line_color="grey")
            fig.update_layout(height=340)
            st.plotly_chart(fig, width="stretch")
            st.caption(
                "R² and RMSE are two outputs of the same folds, so a district "
                "genuinely predicted badly should rank badly on both. It does "
                "not: across all 60 model-crop pairs the ten districts with the "
                "worst R² sit at the 58th RMSE percentile on median — the middle "
                "of the error distribution, and for several pairs the best "
                "quartile. At n=2 the R² denominator is (y₁−y₂)²/2, so a district "
                "whose two harvests came in alike divides by ~0 and posts a "
                "catastrophic R² while predicting fine. That ranking is not noisy, "
                "it is misleading, which is why this page maps RMSE."
            )

# --------------------------------------------------------------------------- #
# SECONDARY — leave-one-Landkreis-out detail
# --------------------------------------------------------------------------- #
if overlay in NEEDS_LOO:
    with st.expander("Leave-one-Landkreis-out — XGBoost", expanded=False):
        if loo_df.empty:
            st.info("`results/metrics/loo_xgboost.csv` not generated. Run "
                    "`python scripts/06b_loo_xgboost.py`.")
        else:
            lsub = loo_df[loo_df["crop"] == crop]
            n_below = int((lsub["n"] < 3).sum())
            if n_below:
                st.warning(
                    f"{n_below} of {len(lsub)} folds on disk hold fewer than 3 "
                    "test observations — below the n>=3 guard now enforced in "
                    "`scripts/06b_loo_xgboost.py`. This CSV predates that guard, "
                    "so fold-level figures are indicative only. Regenerate before "
                    "quoting them."
                )
            st.caption(
                "Each fold is a separate XGBoost fit with the named district "
                "removed from the 2003–2022 training set, so any RMSE increase is "
                "attributable to the model never having seen that district. The "
                "other five models would need GPU compute beyond the Praktikum "
                "budget."
            )
            base = pd_df[(pd_df["model"] == "xgboost") & (pd_df["crop"] == crop)] \
                if not pd_df.empty else pd.DataFrame()
            mg = lsub[["landkreis", "rmse"]].rename(columns={"rmse": "rmse_loo"})
            if not base.empty:
                mg = mg.merge(base[["landkreis", "rmse"]].rename(
                    columns={"rmse": "rmse_in_train"}), on="landkreis",
                    how="inner")

            g1, g2, g3 = st.columns(3)
            with g1:
                kpi("Folds", f"{len(lsub)}")
            with g2:
                kpi("Median LOO RMSE", f"{lsub['rmse'].median():,.2f} {unit}")
            with g3:
                if "rmse_in_train" in mg.columns and not mg.empty:
                    delta = float((mg["rmse_loo"] - mg["rmse_in_train"]).median())
                    kpi("Median RMSE penalty", f"{delta:+,.2f} {unit}",
                        delta="positive = spatial generalization gap"
                        if delta > 0 else None)
                else:
                    kpi("Median RMSE penalty", "—")

            fig = px.histogram(lsub, x="rmse", nbins=20,
                               labels={"rmse": f"LOO RMSE ({unit})"},
                               title=f"LOO RMSE across {len(lsub)} folds — {crop}")
            fig.update_layout(height=300)
            st.plotly_chart(fig, width="stretch")

            if "rmse_in_train" in mg.columns and not mg.empty:
                fig = px.scatter(
                    mg, x="rmse_in_train", y="rmse_loo", hover_data=["landkreis"],
                    labels={"rmse_in_train": f"RMSE, district in training ({unit})",
                            "rmse_loo": f"RMSE, district held out ({unit})"},
                    title="Same district, in training vs held out")
                hi = float(max(mg["rmse_in_train"].max(), mg["rmse_loo"].max()))
                fig.add_shape(type="line", x0=0, y0=0, x1=hi, y1=hi,
                              line=dict(color="grey", dash="dash"))
                fig.update_layout(height=360)
                st.plotly_chart(fig, width="stretch")
                st.caption("Points above the diagonal cost accuracy when the "
                           "district was withheld.")

# --------------------------------------------------------------------------- #
# SECONDARY — per-district yield history (its selector drives nothing else)
# --------------------------------------------------------------------------- #
with st.expander("Per-district yield history", expanded=False):
    codes = sorted(panel_districts)
    if not codes:
        st.info("No districts in the panel.")
    else:
        dist = st.selectbox("Landkreis", codes,
                            format_func=lambda c: f"{name_map.get(c, c)} ({c})",
                            key="sp_dist")
        trend = (panel[(panel["crop"] == crop) & (panel["landkreis"] == dist)]
                 .sort_values("year"))
        if trend.empty:
            st.info(f"No {crop} series for {name_map.get(dist, dist)}.")
        else:
            fig = px.line(trend, x="year", y="yield_dt_per_ha", markers=True,
                          labels={"yield_dt_per_ha": f"Yield ({unit})",
                                  "year": "Year"},
                          title=f"{name_map.get(dist, dist)} ({dist}) — {crop}")
            fig.add_hline(y=float(trend["yield_dt_per_ha"].mean()),
                          line_dash="dash", line_color="grey",
                          annotation_text=f"mean {trend['yield_dt_per_ha'].mean():.1f}")
            for ty in test_years:
                if ty in set(trend["year"]):
                    fig.add_vrect(x0=ty - 0.5, x1=ty + 0.5, fillcolor="#888",
                                  opacity=0.12, line_width=0)
            fig.update_layout(height=340)
            st.plotly_chart(fig, width="stretch")
            st.caption(f"Shaded band: test years "
                       f"{', '.join(map(str, test_years))}.")

# --------------------------------------------------------------------------- #
# SECONDARY — provenance
# --------------------------------------------------------------------------- #
with st.expander("Provenance — panel, sources and file dates", expanded=False):
    _prov = []
    for _p in (PROCESSED, PER_DISTRICT_CSV, LOO_CSV):
        _prov.append({
            "Artefact": str(_p.relative_to(PROJECT_ROOT))
            if _p.is_relative_to(PROJECT_ROOT) else _p.name,
            "Last written": (datetime.fromtimestamp(_p.stat().st_mtime)
                             .strftime("%Y-%m-%d %H:%M")
                             if _p.exists() else "missing"),
        })
    st.dataframe(pd.DataFrame(_prov), hide_index=True, width="stretch")
    st.caption(
        f"Panel: {len(panel_districts)} NUTS-3 districts with yield series "
        f"({len(polys)} German polygons exist; the remainder are historical "
        "Kreis boundaries that do not map to 2021 NUTS geography). "
        "Generated by `scripts/06_spatial_eval.py` and "
        "`scripts/06b_loo_xgboost.py`; the test split is read from "
        f"`configs/data.yaml` ({', '.join(map(str, test_years))})."
    )

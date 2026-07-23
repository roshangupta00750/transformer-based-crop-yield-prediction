"""Spatial holdout analysis — per-district performance + LOO XGBoost.

The PPT (Slide 08) promises a leave-one-Landkreis-out evaluation. Doing
that with full retraining would cost ~150+ hours on CPU for all six
models. So the page surfaces two complementary views:

1. **Per-district test R² for all 6 models** — no retraining, just slicing
   the 2024-2025 test predictions by district. Measures which districts
   each model finds hard.
2. **True LOO for XGBoost** — the methodology-anchor: 50 folds with each
   district fully removed from training. The remaining models would need
   GPU compute beyond the Praktikum budget; documented as such.

Data sources:
  - ``results/metrics/spatial_per_district.csv``  (View 1)
  - ``results/metrics/loo_xgboost.csv``            (View 2)
"""
from __future__ import annotations

import sys
from pathlib import Path

import folium
import geopandas as gpd
import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_folium import st_folium

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import (METRICS_DIR, MODELS, PROJECT_ROOT, divider, kpi,
                   landkreis_name_map, page_header)

st.set_page_config(page_title="Spatial — Crop Yield",
                   layout="wide")
page_header("Spatial holdout analysis",
            "Per-district model performance + leave-one-Landkreis-out "
            "methodology anchor.")
st.caption("_Per-district R² on a NUTS-3 choropleth for all six models plus a true leave-one-Landkreis-out XGBoost anchor — the spatial-generalization dimension the leaderboard collapses._")

# --------------------------------------------------------------------------- #
# Load
# --------------------------------------------------------------------------- #
per_district_path = METRICS_DIR / "spatial_per_district.csv"
loo_path = METRICS_DIR / "loo_xgboost.csv"

if not per_district_path.exists():
    st.warning(
        "`results/metrics/spatial_per_district.csv` not found. "
        "Run `python scripts/06_spatial_eval.py` to generate it."
    )
    st.stop()

pd_df = pd.read_csv(per_district_path)
loo_df = pd.read_csv(loo_path) if loo_path.exists() else pd.DataFrame()
name_map = landkreis_name_map()


@st.cache_data(ttl=3600)
def _load_polygons() -> gpd.GeoDataFrame:
    shp = PROJECT_ROOT / "data" / "raw" / "geo" / "NUTS_RG_01M_2021_4326_LEVL_3.shp"
    g = gpd.read_file(shp)
    g = g[g["CNTR_CODE"] == "DE"][["NUTS_ID", "NUTS_NAME", "geometry"]]
    return g.rename(columns={"NUTS_ID": "landkreis", "NUTS_NAME": "name"})


polys = _load_polygons()

# --------------------------------------------------------------------------- #
# View 1 — Per-district R² map (no retraining)
# --------------------------------------------------------------------------- #
divider("Per-district test R² (no retraining)")

f1, f2 = st.columns([1, 1])
with f1:
    sel_model = st.selectbox(
        "Model",
        sorted(pd_df["model"].unique()),
        index=0,
        key="spatial_model",
    )
with f2:
    crops = sorted(pd_df["crop"].unique())
    sel_crop = st.selectbox(
        "Crop", crops,
        index=crops.index("winter_wheat") if "winter_wheat" in crops else 0,
        key="spatial_crop",
    )

sub = pd_df[(pd_df["model"] == sel_model) & (pd_df["crop"] == sel_crop)]
if sub.empty:
    st.info("No per-district scores for this (model, crop) combo.")
else:
    # Choropleth color scale only carries meaning in [-1, 1] — R² below -1
    # is "much worse than predicting the mean," visually indistinguishable
    # from R² = -1 at the user level. Clip *for the map only* so a broken
    # checkpoint (e.g. TFT with R² ≈ -100) doesn't crash folium's binning.
    sub = sub.copy()
    sub["r2_display"] = sub["r2"].clip(-1.0, 1.0)
    joined = polys.merge(sub, on="landkreis", how="left")

    k1, k2, k3 = st.columns(3)
    with k1: kpi("Districts scored", f"{int(sub['n'].count())}")
    with k2: kpi("Median R²", f"{sub['r2'].median():.3f}")
    with k3: kpi("Mean RMSE (dt/ha)", f"{sub['rmse'].mean():.2f}")

    n_clipped = int((sub["r2"] < -1.0).sum())
    if n_clipped:
        st.warning(
            f"{n_clipped} district(s) had R² < −1 for {sel_model} × {sel_crop} "
            "and were clipped to −1 on the map (color scale only). "
            "The KPI / tooltip values above show the raw R²."
        )

    center = [51.1657, 10.4515]
    m = folium.Map(location=center, zoom_start=6, tiles="cartodbpositron")
    if joined["r2_display"].notna().any():
        folium.Choropleth(
            geo_data=joined.__geo_interface__,
            data=joined,
            columns=["landkreis", "r2_display"],
            key_on="feature.properties.landkreis",
            fill_color="RdYlGn",
            fill_opacity=0.78,
            line_opacity=0.2,
            legend_name=f"R² — {sel_model} × {sel_crop} (clipped to [−1, 1])",
            nan_fill_color="lightgrey",
            nan_fill_opacity=0.4,
            bins=[-1.0, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        ).add_to(m)
    folium.GeoJson(
        joined.__geo_interface__,
        style_function=lambda f: {"color": "white", "weight": 0.4, "fillOpacity": 0},
        tooltip=folium.features.GeoJsonTooltip(
            fields=["name", "landkreis", "r2", "rmse", "n"],
            aliases=["District", "NUTS-3", "R²", "RMSE", "n"],
            localize=True, max_width=300,
        ),
    ).add_to(m)
    st_folium(m, width=None, height=560, returned_objects=[])

    # Worst + best districts table
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Worst 10 districts (lowest R²)")
        worst = sub.nsmallest(10, "r2").copy()
        worst["name"] = worst["landkreis"].map(name_map).fillna(worst["landkreis"])
        st.dataframe(
            worst[["name", "landkreis", "r2", "rmse", "mae", "n"]].rename(
                columns={"name": "District", "landkreis": "NUTS-3",
                         "rmse": "RMSE", "mae": "MAE"}
            ).style.format({"r2": "{:.3f}", "RMSE": "{:.2f}", "MAE": "{:.2f}"}),
            width="stretch", hide_index=True,
        )
    with c2:
        st.markdown("##### Best 10 districts (highest R²)")
        best = sub.nlargest(10, "r2").copy()
        best["name"] = best["landkreis"].map(name_map).fillna(best["landkreis"])
        st.dataframe(
            best[["name", "landkreis", "r2", "rmse", "mae", "n"]].rename(
                columns={"name": "District", "landkreis": "NUTS-3",
                         "rmse": "RMSE", "mae": "MAE"}
            ).style.format({"r2": "{:.3f}", "RMSE": "{:.2f}", "MAE": "{:.2f}"}),
            width="stretch", hide_index=True,
        )

# --------------------------------------------------------------------------- #
# View 2 — LOO XGBoost
# --------------------------------------------------------------------------- #
divider("Leave-one-Landkreis-out — XGBoost (proper methodology anchor)")
if loo_df.empty:
    st.info(
        "`results/metrics/loo_xgboost.csv` not yet generated. "
        "Run `python scripts/06b_loo_xgboost.py` (≈ 30-50 min for 50 districts × 10 crops)."
    )
else:
    st.caption(
        "Each row here is a separate XGBoost training run with the named "
        "district fully removed from the 2003-2020 training set. Compared "
        "to the per-district view above, any drop in R² is attributable to "
        "the model genuinely never having seen that district. The remaining "
        "five models would need GPU compute beyond the Praktikum budget for "
        "this same methodology."
    )

    loo_crop = st.selectbox(
        "Crop", sorted(loo_df["crop"].unique()),
        index=0, key="loo_crop",
    )
    lsub = loo_df[loo_df["crop"] == loo_crop]
    has_r2 = lsub["r2"].notna()
    k1, k2, k3, k4 = st.columns(4)
    with k1: kpi("Folds run", f"{len(lsub)}")
    with k2: kpi("Median R² (where defined)",
                 f"{lsub.loc[has_r2, 'r2'].median():.3f}" if has_r2.any() else "—")
    with k3: kpi("Mean RMSE", f"{lsub['rmse'].mean():.2f} dt/ha")
    with k4:
        # Compare: same crop, XGBoost from the per-district view (no LOO)
        same = pd_df[(pd_df["model"] == "xgboost") & (pd_df["crop"] == loo_crop)]
        delta = (lsub.loc[has_r2, "r2"].median() - same["r2"].median()) \
            if has_r2.any() and not same.empty else None
        kpi("Median R² Δ vs no-LOO",
            f"{delta:+.3f}" if delta is not None else "—",
            delta="more negative = bigger generalization gap"
            if delta is not None and delta < 0 else None)

    # Histogram of LOO R²
    if has_r2.any():
        fig = px.histogram(
            lsub.loc[has_r2], x="r2", nbins=20,
            title=f"Distribution of LOO R² across {has_r2.sum()} folds — {loo_crop}",
            labels={"r2": "R² when district was held out"},
        )
        fig.add_vline(x=0, line_dash="dot", line_color="grey")
        fig.update_layout(height=320)
        st.plotly_chart(fig, width="stretch")

    # Compare side-by-side: per-district (no LOO) vs LOO for the same districts.
    if not same.empty and not lsub.empty:
        merged = same[["landkreis", "r2"]].merge(
            lsub[["landkreis", "r2"]], on="landkreis", how="inner",
            suffixes=("_no_loo", "_loo"),
        )
        if not merged.empty:
            st.markdown("##### Same district: no-LOO R² vs LOO R²")
            fig = px.scatter(
                merged, x="r2_no_loo", y="r2_loo",
                hover_data=["landkreis"],
                title="Per-district R²: no-LOO vs LOO (XGBoost)",
                labels={"r2_no_loo": "R² (no LOO, district was in training)",
                        "r2_loo": "R² (LOO, district held out)"},
            )
            lo = min(merged["r2_no_loo"].min(), merged["r2_loo"].min(), -0.5)
            hi = max(merged["r2_no_loo"].max(), merged["r2_loo"].max(), 1.0)
            fig.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi,
                          line=dict(color="grey", dash="dash"))
            fig.update_layout(height=380)
            st.plotly_chart(fig, width="stretch")
            st.caption(
                "Points on the diagonal mean LOO didn't hurt; points below "
                "the line mean the district benefited from being in training "
                "(spatial generalization gap)."
            )

st.markdown("---")
st.caption(
    "Data files: `results/metrics/spatial_per_district.csv` (View 1, "
    "regenerate with `python scripts/06_spatial_eval.py`); "
    "`results/metrics/loo_xgboost.csv` (View 2, regenerate with "
    "`python scripts/06b_loo_xgboost.py`)."
)

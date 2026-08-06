"""Data sources page — provenance + live previews."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import (DATA_SOURCES, INTERIM, RAW, divider, kpi, load_unified,
                   page_header)

st.set_page_config(page_title="Data — Crop Yield Benchmark",
                   layout="wide")

page_header("Data sources & previews",
            "Origin, license, and a live row preview for each modality.")

# --------------------------------------------------------------------------- #
# Hero KPI row — the four numbers a reader needs at a glance
# --------------------------------------------------------------------------- #
df = load_unified()
if df.empty:
    st.warning("`data/processed/dataset.parquet` not found. Run "
               "`python scripts/02_preprocess.py` first.")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
with c1: kpi("Rows", f"{len(df):,}")
with c2: kpi("Districts (NUTS-3)", f"{df['landkreis'].nunique()}")
with c3: kpi("Crops", f"{df['crop'].nunique()}")
with c4: kpi("Year span",
             f"{int(df['year'].min())}–{int(df['year'].max())}")

# --------------------------------------------------------------------------- #
# Provenance table (reference material — collapsed by default)
# --------------------------------------------------------------------------- #
with st.expander("Source provenance table", expanded=False):
    src_df = pd.DataFrame(DATA_SOURCES)
    src_df["source"] = src_df["source"] + " — " + src_df["url"]
    st.dataframe(
        src_df.drop(columns=["url"]).rename(columns={
            "modality": "Modality", "source": "Source",
            "resolution": "Resolution", "variables": "Variables",
            "years": "Years", "license": "License",
        }),
        width="stretch", hide_index=True,
    )

# --------------------------------------------------------------------------- #
# Unified dataset preview (collapsed)
# --------------------------------------------------------------------------- #
with st.expander("Unified processed dataset — row preview", expanded=False):
    st.dataframe(df.head(50), width="stretch", hide_index=True)

with st.expander("Column dtypes + null counts"):
    info = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "non_null": df.notna().sum(),
        "null %": (100 * df.isna().mean()).round(2),
        "n_unique_sample": df.head(50_000).nunique(),
    })
    st.dataframe(info, width="stretch")

# --------------------------------------------------------------------------- #
# Per-modality interim previews (cheap reads) — collapsed by default
# --------------------------------------------------------------------------- #
with st.expander("Per-source interim previews", expanded=False):
    tabs = st.tabs(["Weather", "Satellite", "Soil", "Yield"])

    with tabs[0]:
        p = INTERIM / "weather.parquet"
        if p.exists():
            w = pd.read_parquet(p)
            st.caption(f"`{p.name}` — {len(w):,} rows, "
                       f"{w['landkreis'].nunique()} districts, "
                       f"{w['date'].min().date()} – {w['date'].max().date()}")
            st.dataframe(w.head(30), width="stretch", hide_index=True)
            # Pick a district and crop year, plot weather
            district = st.selectbox("District",
                                    sorted(w["landkreis"].unique())[:50],
                                    key="w_dist")
            sub = w[w["landkreis"] == district].copy()
            sub["year"] = pd.to_datetime(sub["date"]).dt.year
            year = st.select_slider("Year", sorted(sub["year"].unique()),
                                    value=int(sub["year"].max()), key="w_year")
            sy = sub[sub["year"] == year].sort_values("date")
            feats = [c for c in sy.columns
                     if c.startswith(("temperature_", "precip",
                                      "radiation", "humid"))]
            fig = px.line(sy, x="date", y=feats,
                          title=f"{district} — {year}")
            fig.update_layout(height=380, legend_title_text="")
            st.plotly_chart(fig, width="stretch")
        else:
            st.warning("No weather.parquet yet. Run `02_preprocess.py`.")

    with tabs[1]:
        p = INTERIM / "satellite.parquet"
        if p.exists():
            s = pd.read_parquet(p)
            st.caption(f"`{p.name}` — {len(s):,} rows, "
                       f"{s['landkreis'].nunique()} districts")
            st.dataframe(s.head(30), width="stretch", hide_index=True)
            district = st.selectbox("District",
                                    sorted(s["landkreis"].unique())[:50],
                                    key="s_dist")
            sub = s[s["landkreis"] == district].copy()
            sub["year"] = pd.to_datetime(sub["date"]).dt.year
            sub["doy"] = pd.to_datetime(sub["date"]).dt.dayofyear
            years = sorted(sub["year"].unique())
            sel_years = st.multiselect("Years", years,
                                       default=years[-3:], key="s_years")
            sf = sub[sub["year"].isin(sel_years)]
            if not sf.empty:
                fig = px.line(sf, x="doy", y="ndvi", color="year",
                              title=f"NDVI seasonal curve — {district}")
                fig.update_layout(height=380)
                st.plotly_chart(fig, width="stretch")
        else:
            st.info("Satellite data missing. Run "
                    "`01_download_data.py --source gee` then "
                    "`02_preprocess.py`.")

    with tabs[2]:
        p = INTERIM / "soil.parquet"
        if p.exists():
            so = pd.read_parquet(p)
            st.caption(f"`{p.name}` — {len(so):,} districts")
            st.dataframe(so.head(30), width="stretch", hide_index=True)
            feat = st.selectbox("Property to plot",
                                [c for c in so.columns if c != "landkreis"],
                                key="so_feat")
            fig = px.histogram(so, x=feat, nbins=40,
                               title=f"Distribution of {feat} "
                                     "across Landkreise")
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("Soil data missing. Run "
                    "`01_download_data.py --source soilgrids`.")

    with tabs[3]:
        p = INTERIM / "yield.parquet"
        if p.exists():
            y = pd.read_parquet(p)
            st.caption(f"`{p.name}` — {len(y):,} rows, "
                       f"{y['landkreis'].nunique()} districts, "
                       f"{y['crop'].nunique()} crops")
            st.dataframe(y.head(30), width="stretch", hide_index=True)
            crop = st.selectbox("Crop", sorted(y["crop"].unique()),
                                key="y_crop")
            ysub = y[y["crop"] == crop]
            yearly = ysub.groupby("year")["yield_dt_per_ha"].agg(
                ["mean", "std", "min", "max"])
            fig = px.line(yearly.reset_index(), x="year", y="mean",
                          error_y=yearly["std"].values,
                          title=f"Annual mean yield (± std across "
                                f"districts) — {crop}")
            fig.update_layout(height=380, yaxis_title="Yield (dt/ha)")
            st.plotly_chart(fig, width="stretch")
        else:
            st.warning("Yield data missing. Set REGIO_USER/REGIO_PASS "
                       "and run `01_download_data.py --source destatis`.")

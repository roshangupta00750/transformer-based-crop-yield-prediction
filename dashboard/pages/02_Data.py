"""Data — the multivariate panel, audited live.

The page answers one question: what is in this dataset? Three elements are
visible by default — the panel in five numbers, the district x year coverage
grid, and the multivariate series viewer. Everything else built for this page
survives untouched inside three collapsed expanders (coverage detail, feature
inventory and column audit, sources and provenance). Where a collapsed section
contains a live defect, the count rides in the expander label so that folding
it away does not conceal it.

Every number rendered here is computed at request time from
``data/processed/dataset.parquet``, ``configs/data.yaml``,
``src/data/dataset.py`` or the per-source interim parquets. Nothing is
hardcoded: this dataset has been rebuilt several times and every literal
previously written into this page went stale on the next rebuild.

Cheap paths are used deliberately. Column null counts and min/max come from
parquet row-group statistics (metadata only, no data read); a single
time-series slice is a predicate-pushdown read of a handful of row groups.
The two genuinely expensive passes (column moments, exact-duplicate audit)
are cached against the parquet's mtime+size so a rebuild invalidates them.
"""
from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pyarrow.parquet as pq
import streamlit as st
import yaml
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import (DATA_SOURCES, INTERIM, PROCESSED, PROJECT_ROOT, kpi,
                   landkreis_name_map, page_header)

st.set_page_config(page_title="Data — Crop Yield Benchmark", layout="wide")

CONFIG_PATH = PROJECT_ROOT / "configs" / "data.yaml"
DATASET_PY = PROJECT_ROOT / "src" / "data" / "dataset.py"
POLICY_SIDECAR = PROCESSED.with_suffix(".policy.json")
CENTROIDS = INTERIM / "landkreis_centroids.parquet"

# Categorical hues, fixed order, never cycled (validated adjacent-pair palette).
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
                 "#008300", "#4a3aa7", "#e34948"]

KEY_COLUMNS = {"landkreis", "date", "crop", "year", "doy"}
TARGET_COLUMN = "yield_dt_per_ha"


# --------------------------------------------------------------------------- #
# Cache keys — (path, mtime, size) so a rebuild busts every derived cache
# --------------------------------------------------------------------------- #
def sig(path: Path) -> tuple:
    try:
        s = path.stat()
        return (str(path), int(s.st_mtime), int(s.st_size))
    except OSError:
        return (str(path), 0, 0)


# --------------------------------------------------------------------------- #
# Config / policy readers
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def read_config(_sig: tuple) -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text()) if CONFIG_PATH.exists() else {}


@st.cache_data(show_spinner=False)
def read_list_comments(_sig: tuple, key: str) -> dict[str, str]:
    """Attach each YAML comment block to the list items that follow it.

    The exclusion rationale lives in comments in ``configs/data.yaml``; this
    lifts it out so the page can render the policy as a table with reasons
    instead of restating it as prose that drifts from the config.
    """
    if not CONFIG_PATH.exists():
        return {}
    lines = CONFIG_PATH.read_text().splitlines()
    start = next((i for i, ln in enumerate(lines)
                  if re.match(rf"^(\s*){re.escape(key)}:\s*$", ln)), None)
    if start is None:
        return {}
    base_indent = len(lines[start]) - len(lines[start].lstrip())
    out: dict[str, str] = {}
    buf: list[str] = []
    current = ""
    consumed = False
    for ln in lines[start + 1:]:
        if not ln.strip():
            continue
        indent = len(ln) - len(ln.lstrip())
        if indent <= base_indent:
            break
        s = ln.strip()
        if s.startswith("#"):
            if consumed:
                buf, consumed = [], False
            buf.append(s.lstrip("#").strip())
        elif s.startswith("- "):
            if buf:
                current = " ".join(buf)
                consumed = True
            out[s[2:].strip()] = current
    return out


@st.cache_data(show_spinner=False)
def code_policy_version(_sig: tuple) -> int | None:
    """FEATURE_POLICY_VERSION from src/data/dataset.py, without importing torch."""
    if not DATASET_PY.exists():
        return None
    try:
        tree = ast.parse(DATASET_PY.read_text())
    except SyntaxError:
        return None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "FEATURE_POLICY_VERSION":
                    try:
                        return int(ast.literal_eval(node.value))
                    except ValueError:
                        return None
    return None


@st.cache_data(show_spinner=False)
def data_policy_version(_sig: tuple) -> int | None:
    if not POLICY_SIDECAR.exists():
        return None
    try:
        return int(json.loads(POLICY_SIDECAR.read_text())["feature_policy_version"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


# --------------------------------------------------------------------------- #
# Parquet readers
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def parquet_profile(_sig: tuple) -> dict:
    """Rows, dtypes, null counts and min/max — from row-group metadata only."""
    pf = pq.ParquetFile(PROCESSED)
    md, schema = pf.metadata, pf.schema_arrow
    col_index = {md.row_group(0).column(i).path_in_schema: i
                 for i in range(md.row_group(0).num_columns)}
    rows = []
    for name, typ in zip(schema.names, schema.types):
        nulls, lo, hi = 0, None, None
        i = col_index.get(name)
        if i is not None:
            for g in range(md.num_row_groups):
                stats = md.row_group(g).column(i).statistics
                if stats is None:
                    nulls, lo, hi = None, None, None
                    break
                nulls += stats.null_count
                if stats.has_min_max:
                    lo = stats.min if lo is None else min(lo, stats.min)
                    hi = stats.max if hi is None else max(hi, stats.max)
        rows.append({"column": name, "dtype": str(typ),
                     "nulls": nulls, "min": lo, "max": hi})
    return {"rows": md.num_rows, "n_columns": md.num_columns,
            "row_groups": md.num_row_groups,
            "size_mb": PROCESSED.stat().st_size / 1e6,
            "columns": rows,
            "names": list(schema.names),
            "numeric": [n for n, t in zip(schema.names, schema.types)
                        if str(t) not in ("string",) and n != "date"]}


@st.cache_data(show_spinner="Computing per-column moments over 16.5M rows…")
def column_moments(_sig: tuple, columns: tuple[str, ...]) -> pd.DataFrame:
    """Streaming mean/std — one column in memory at a time, never the frame."""
    import pyarrow.compute as pc
    out = []
    for c in columns:
        arr = pq.read_table(PROCESSED, columns=[c])[c]
        out.append({"column": c,
                    "mean": pc.mean(arr).as_py(),
                    "std": pc.stddev(arr).as_py(),
                    "n_distinct": pc.count_distinct(arr).as_py()})
    return pd.DataFrame(out)


@st.cache_data(show_spinner="Auditing columns for exact duplication…")
def duplicate_classes(_sig: tuple, columns: tuple[str, ...]) -> list[list[str]]:
    """Exact byte-equality classes over all 16.5M rows.

    Two stages so this stays cheap: row-group (null_count, min, max) vectors
    are a free metadata fingerprint that no distinct pair of columns can
    share by accident often; only columns sharing a fingerprint are then read
    and compared element-wise. No sampling — the equality claim is exact.
    """
    md = pq.ParquetFile(PROCESSED).metadata
    col_index = {md.row_group(0).column(i).path_in_schema: i
                 for i in range(md.row_group(0).num_columns)}
    fingerprints: dict[tuple, list[str]] = {}
    for c in columns:
        i = col_index.get(c)
        if i is None:
            continue
        sigv = []
        for g in range(md.num_row_groups):
            s = md.row_group(g).column(i).statistics
            sigv.append((s.null_count, s.min, s.max) if s is not None else (None,))
        fingerprints.setdefault(tuple(sigv), []).append(c)

    classes: list[list[str]] = []
    for group in fingerprints.values():
        if len(group) < 2:
            continue
        table = pq.read_table(PROCESSED, columns=group)
        arrays = {c: table[c].combine_chunks().to_numpy(zero_copy_only=False)
                  for c in group}
        buckets: list[list[str]] = []
        for c in group:
            for b in buckets:
                a0, a1 = arrays[b[0]], arrays[c]
                if a0.dtype != a1.dtype:
                    continue
                same = np.array_equal(
                    a0, a1, equal_nan=np.issubdtype(a0.dtype, np.floating))
                if same:
                    b.append(c)
                    break
            else:
                buckets.append([c])
        classes += [b for b in buckets if len(b) > 1]
    return classes


@st.cache_data(show_spinner="Indexing the (district, crop, year) panel…")
def panel_index(_sig: tuple, _cfg_sig: tuple) -> pd.DataFrame:
    """One row per crop-year sequence, with its growing-season coverage.

    Reproduces `src.data.dataset.apply_phenology_window` +
    `filter_incomplete_sequences` so the page shows the sequences the models
    actually see, not the raw calendar-year rows in the parquet.
    """
    cfg = read_config(sig(CONFIG_PATH))
    phen = cfg.get("phenology", {})
    df = pd.read_parquet(
        PROCESSED, columns=["landkreis", "crop", "year", "doy", TARGET_COLUMN])
    sow = df["crop"].map({c: p["sowing_doy"] for c, p in phen.items()})
    harv = df["crop"].map({c: p["harvest_doy"] for c, p in phen.items()})
    in_window = np.where(sow < harv,
                         (df["doy"] >= sow) & (df["doy"] <= harv),
                         df["doy"] <= harv)
    expected = {c: (p["harvest_doy"] if p["sowing_doy"] > p["harvest_doy"]
                    else p["harvest_doy"] - p["sowing_doy"] + 1)
                for c, p in phen.items()}
    keys = ["landkreis", "crop", "year"]
    raw = df.groupby(keys, observed=True).agg(
        calendar_days=("doy", "size"), yield_dt_per_ha=(TARGET_COLUMN, "max"))
    win = (df[in_window].groupby(keys, observed=True)["doy"]
           .nunique().rename("season_days"))
    out = raw.join(win, how="left").reset_index()
    out["season_days"] = out["season_days"].fillna(0).astype(int)
    out["expected_days"] = out["crop"].map(expected)
    out["coverage"] = out["season_days"] / out["expected_days"]
    return out


@st.cache_data(show_spinner=False)
def sequence_slice(_sig: tuple, landkreis: str, crop: str,
                   years: tuple[int, ...], columns: tuple[str, ...]) -> pd.DataFrame:
    """Predicate-pushdown read of a few crop-year sequences (sub-second)."""
    need = sorted({"year", "doy", "date", *columns})
    frames = []
    for y in years:
        frames.append(pd.read_parquet(
            PROCESSED, columns=need,
            filters=[("landkreis", "==", landkreis), ("crop", "==", crop),
                     ("year", "==", int(y))]))
    if not frames:
        return pd.DataFrame(columns=need)
    return pd.concat(frames, ignore_index=True).sort_values(["year", "doy"])


@st.cache_data(show_spinner="Sampling channels for the correlation matrix…")
def channel_correlation(_sig: tuple, columns: tuple[str, ...],
                        stride: int) -> tuple[pd.DataFrame, int]:
    """Pearson correlation on a systematic every-Nth-row subsample.

    Streamed one column at a time so the full 16.5M x N frame is never
    materialised. The subsample size is returned and displayed — this is not
    a full-population correlation and the page says so.
    """
    cols = {}
    for c in columns:
        arr = (pq.read_table(PROCESSED, columns=[c])[c]
               .combine_chunks().to_numpy(zero_copy_only=False))
        cols[c] = np.asarray(arr, dtype="float64")[::stride]
    frame = pd.DataFrame(cols)
    return frame.corr(), len(frame)


@st.cache_data(show_spinner="Reading per-source district coverage…")
def source_districts(_sigs: tuple) -> dict[str, set]:
    """Which NUTS-3 codes each pipeline stage actually carries."""
    out: dict[str, set] = {}
    files = {"NUTS-3 geography": CENTROIDS,
             "Yield labels": INTERIM / "yield.parquet",
             "DWD weather": INTERIM / "weather.parquet",
             "Satellite": INTERIM / "satellite.parquet",
             "Soil": INTERIM / "soil.parquet"}
    for label, path in files.items():
        if path.exists():
            out[label] = set(pd.read_parquet(path, columns=["landkreis"])
                             ["landkreis"].astype(str).unique())
    out["Unified panel"] = set(pd.read_parquet(PROCESSED, columns=["landkreis"])
                               ["landkreis"].astype(str).unique())
    return out


@st.cache_data(show_spinner=False)
def source_extent(_sigs: tuple) -> pd.DataFrame:
    """Rows / districts / observed span per interim file, read live."""
    rows = []
    for label, name in [("Weather", "weather.parquet"),
                        ("Satellite", "satellite.parquet"),
                        ("Soil", "soil.parquet"),
                        ("Yield labels", "yield.parquet")]:
        path = INTERIM / name
        if not path.exists():
            rows.append({"Modality": label, "File": name, "Status": "missing"})
            continue
        pf = pq.ParquetFile(path)
        names = pf.schema_arrow.names
        span = "static"
        if "date" in names:
            d = pd.read_parquet(path, columns=["date"])["date"]
            span = f"{d.min().date()} – {d.max().date()}"
        elif "year" in names:
            y = pd.read_parquet(path, columns=["year"])["year"]
            span = f"{int(y.min())} – {int(y.max())}"
        districts = pd.read_parquet(path, columns=["landkreis"])["landkreis"].nunique()
        rows.append({
            "Modality": label, "File": name, "Status": "present",
            "Rows": f"{pf.metadata.num_rows:,}", "Districts": districts,
            "Observed span": span,
            "Variables in file": ", ".join(
                c for c in names if c not in ("landkreis", "date", "year")),
        })
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def satellite_cadence(_sig: tuple) -> tuple[int, int] | None:
    """Median gap between genuinely observed satellite dates, and their count."""
    path = INTERIM / "satellite.parquet"
    if not path.exists():
        return None
    s = pd.read_parquet(path, columns=["date", "ndvi_observed"])
    dates = np.sort(s.loc[s["ndvi_observed"] == 1, "date"].unique())
    if len(dates) < 2:
        return None
    gaps = np.diff(dates).astype("timedelta64[D]").astype(int)
    return int(np.median(gaps)), int(len(dates))




# --------------------------------------------------------------------------- #
# Page — one question: what is in this dataset?
#
# Three primary elements are visible by default: the panel in five numbers,
# one coverage picture, and the multivariate series viewer. Every other audit
# built for this page is preserved verbatim, one click away, inside three
# collapsed expanders whose labels carry the live counts so that nothing
# alarming is hidden behind a neutral word.
# --------------------------------------------------------------------------- #
page_header(
    "Data",
    "What is in this dataset. Every number is profiled from the parquet, the "
    "feature policy and the interim sources at page load.")

if not PROCESSED.exists():
    st.warning("`data/processed/dataset.parquet` not found. Run "
               "`python scripts/02_preprocess.py` first.")
    st.stop()

PQ_SIG = sig(PROCESSED)
CFG_SIG = sig(CONFIG_PATH)
profile = parquet_profile(PQ_SIG)
cfg = read_config(CFG_SIG)
features_cfg = cfg.get("features", {}) or {}
panel = panel_index(PQ_SIG, CFG_SIG)
min_cov = float((cfg.get("quality") or {}).get("min_sequence_coverage", 0.90))

kept = panel[panel["coverage"] >= min_cov]
n_districts = panel["landkreis"].nunique()
n_years = panel["year"].nunique()
NAME_MAP = landkreis_name_map()

SOURCE_SIGS = tuple(sig(p) for p in
                    [CENTROIDS, INTERIM / "yield.parquet",
                     INTERIM / "weather.parquet", INTERIM / "satellite.parquet",
                     INTERIM / "soil.parquet", PROCESSED])

v_code = code_policy_version(sig(DATASET_PY))
v_data = data_policy_version(sig(POLICY_SIDECAR))
policy_mismatch = (v_code is not None and v_data is not None and v_code != v_data)


# --------------------------------------------------------------------------- #
# PRIMARY 1 — the panel in five numbers
# --------------------------------------------------------------------------- #
c1, c2, c3, c4, c5 = st.columns(5)
with c1: kpi("Rows", f"{profile['rows']:,}")
with c2: kpi("Districts (NUTS-3)", n_districts)
with c3: kpi("Crops", panel["crop"].nunique())
with c4: kpi("Years", f"{int(panel['year'].min())}–{int(panel['year'].max())}")
with c5: kpi("Crop-year sequences", f"{len(kept):,}",
             delta=f"-{len(panel) - len(kept):,} below coverage floor")

# A firing policy mismatch is never collapsed: it invalidates everything below
# it. When code and parquet agree, the version detail lives in Provenance.
if policy_mismatch:
    st.error(f"Feature-policy mismatch: `src/data/dataset.py` declares v{v_code} "
             f"but `dataset.policy.json` says the parquet was built under "
             f"v{v_data}. Rebuild before trusting anything below.")


# --------------------------------------------------------------------------- #
# PRIMARY 2 — one coverage picture: which district-years exist at all
# --------------------------------------------------------------------------- #
st.markdown("#### District x year coverage")
grid_col1, grid_col2, grid_col3 = st.columns([2, 2, 2])
crop_opts = ["All crops (count)"] + sorted(panel["crop"].unique())
with grid_col1:
    grid_crop = st.selectbox("Cell value", crop_opts, key="grid_crop")
with grid_col2:
    apply_floor = st.checkbox(
        f"Only sequences passing the {min_cov:.0%} coverage floor",
        value=True, key="grid_floor")
with grid_col3:
    show_all_nuts = st.checkbox(
        "Include NUTS-3 districts absent from the panel",
        value=False, key="grid_allnuts")

src = kept if apply_floor else panel
if grid_crop == "All crops (count)":
    cell = (src.groupby(["landkreis", "year"], observed=True)["crop"]
            .nunique().rename("value").reset_index())
    cbar = "Crops present"
else:
    cs = src[src["crop"] == grid_crop]
    cell = (cs.groupby(["landkreis", "year"], observed=True)["crop"]
            .size().rename("value").reset_index())
    cbar = f"{grid_crop} present"

grid = cell.pivot(index="landkreis", columns="year", values="value")
if show_all_nuts:
    all_nuts = source_districts(SOURCE_SIGS).get("NUTS-3 geography")
    if all_nuts:
        grid = grid.reindex(sorted(all_nuts))
grid = grid.sort_index()
labels = [f"{NAME_MAP.get(k, '')} ({k})".strip() for k in grid.index]

total_cells = grid.shape[0] * grid.shape[1]
filled = int(grid.notna().sum().sum())

heat = go.Figure(go.Heatmap(
    z=grid.to_numpy(), x=[str(y) for y in grid.columns], y=labels,
    colorscale="Blues", colorbar=dict(title=cbar),
    hovertemplate="%{y}<br>%{x}<br>" + cbar + ": %{z}<extra></extra>",
    hoverongaps=False))
heat.update_layout(height=min(760, max(420, 13 * len(grid))),
                   xaxis_title="", yaxis_title="",
                   margin=dict(l=0, r=0, t=10, b=0))
# Below ~60 rows the district names are legible; above it they collide, so the
# axis is dropped and the name is served on hover instead of drawn illegibly.
heat.update_yaxes(showticklabels=len(grid) <= 60, tickfont=dict(size=8))
st.plotly_chart(heat, width="stretch")
st.caption(
    f"{filled:,} of {total_cells:,} district-year cells are populated "
    f"({100 * filled / total_cells if total_cells else 0:.1f}%). "
    f"Blank cells are real gaps — no "
    f"sequence exists for that district-year; the panel is unbalanced. Values "
    f"are counted from the panel index, not assumed. Hover a cell for the "
    f"district name.")


# --------------------------------------------------------------------------- #
# PRIMARY 3 — the multivariate time-series viewer
# --------------------------------------------------------------------------- #
st.markdown("#### Inspect one crop-year sequence")
districts = sorted(panel["landkreis"].unique())
labelled = {landkreis: f"{NAME_MAP.get(landkreis, '')} ({landkreis})".strip()
            for landkreis in districts}

s1, s2, s3 = st.columns([3, 2, 3])
with s1:
    district = st.selectbox("District", districts,
                            format_func=lambda k: labelled[k],
                            key="ts_district")
d_panel = panel[panel["landkreis"] == district]
with s2:
    crop = st.selectbox("Crop", sorted(d_panel["crop"].unique()), key="ts_crop")
dc_panel = d_panel[d_panel["crop"] == crop].sort_values("year")
avail_years = [int(y) for y in dc_panel["year"]]
with s3:
    years = st.multiselect(
        "Year(s)", avail_years,
        default=avail_years[-1:] if avail_years else [],
        max_selections=len(SERIES_COLORS), key="ts_years")

numeric_channels = [c for c in profile["numeric"]
                    if c not in KEY_COLUMNS and c != "landkreis_id"]
default_channels = [c for c in ["temperature_air_max_2m",
                                "precipitation_height", "ndvi", "lai"]
                    if c in numeric_channels]
ch1, ch2, ch3 = st.columns([4, 2, 2])
with ch1:
    channels = st.multiselect("Channels", numeric_channels,
                              default=default_channels, max_selections=8,
                              key="ts_channels")
with ch2:
    mode = st.radio("Layout", ["Stacked (raw units)", "Overlaid (z-scored)"],
                    key="ts_mode", horizontal=False)
with ch3:
    show_rows = st.checkbox("Show the rows behind this plot", value=False,
                            key="ts_rows")

seq = pd.DataFrame()
if not years:
    st.info("Pick at least one year. This district-crop pair exists in "
            f"{len(avail_years)} of {n_years} years.")
elif not channels:
    st.info("Pick at least one channel.")
else:
    seq = sequence_slice(PQ_SIG, district, crop, tuple(sorted(years)),
                         tuple(channels + [c for c in
                                           ("ndvi_observed",) if c in
                                           numeric_channels]))
    if seq.empty:
        st.warning("No rows returned for that combination.")
    else:
        phen = (cfg.get("phenology") or {}).get(crop, {})
        sow, harv = phen.get("sowing_doy"), phen.get("harvest_doy")
        win_lo = 1 if (sow is not None and harv is not None and sow > harv) else sow
        meta = dc_panel[dc_panel["year"].isin(years)]

        if mode.startswith("Stacked"):
            fig = make_subplots(rows=len(channels), cols=1,
                                shared_xaxes=True, vertical_spacing=0.03,
                                subplot_titles=channels)
            for r, ch in enumerate(channels, start=1):
                for i, y in enumerate(sorted(years)):
                    sub = seq[seq["year"] == y]
                    fig.add_trace(go.Scatter(
                        x=sub["doy"], y=sub[ch], mode="lines", name=str(y),
                        legendgroup=str(y), showlegend=(r == 1),
                        line=dict(width=2,
                                  color=SERIES_COLORS[i % len(SERIES_COLORS)]),
                        hovertemplate=(f"{ch}<br>DOY %{{x}}<br>"
                                       f"%{{y:.4g}}<extra>{y}</extra>")),
                        row=r, col=1)
                if harv is not None:
                    fig.add_vline(x=harv, line_dash="dot", line_width=1,
                                  line_color="#888888", row=r, col=1)
            fig.update_layout(height=max(300, 170 * len(channels)),
                              hovermode="x unified",
                              margin=dict(l=0, r=0, t=30, b=0),
                              legend_title_text="Year")
            fig.update_xaxes(title_text="day of year", row=len(channels), col=1)
            for ann in fig.layout.annotations:
                ann.font.size = 12
            st.plotly_chart(fig, width="stretch")
        else:
            z = seq.copy()
            for ch in channels:
                sd = z[ch].std()
                z[ch] = (z[ch] - z[ch].mean()) / (sd if sd else 1.0)
            fig = go.Figure()
            for i, ch in enumerate(channels):
                for y in sorted(years):
                    sub = z[z["year"] == y]
                    fig.add_trace(go.Scatter(
                        x=sub["doy"], y=sub[ch], mode="lines",
                        name=f"{ch} · {y}" if len(years) > 1 else ch,
                        line=dict(width=2,
                                  color=SERIES_COLORS[i % len(SERIES_COLORS)],
                                  dash="solid" if y == min(years) else "dot"),
                        hovertemplate="DOY %{x}<br>z %{y:.2f}<extra></extra>"))
            if harv is not None:
                fig.add_vline(x=harv, line_dash="dot", line_width=1,
                              line_color="#888888",
                              annotation_text="harvest DOY")
            fig.update_layout(height=460, hovermode="x unified",
                              xaxis_title="day of year",
                              yaxis_title="z-score (within this slice)",
                              margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, width="stretch")

        # One context line instead of a second KPI row: the facts a reader
        # needs to interpret the traces above, including the carry-forward
        # caveat that explains their step structure.
        bits = [f"{len(seq):,} rows"]
        if harv is not None:
            bits.append(f"season window DOY {win_lo}–{harv}")
        bits.append("yield " + " / ".join(
            f"{v:.1f}" for v in meta["yield_dt_per_ha"]) + " dt/ha")
        bits.append("season coverage " + " / ".join(
            f"{v:.0%}" for v in meta["coverage"]))
        note = " · ".join(bits) + "."
        if "ndvi_observed" in seq.columns:
            note += (f" {float(seq['ndvi_observed'].mean()):.1%} of days carry a "
                     f"genuine satellite observation; the rest are carried "
                     f"forward from the last composite, and the step structure "
                     f"in the NDVI/EVI/LAI traces is that carry-forward, not a "
                     f"smooth signal.")
        if not mode.startswith("Stacked"):
            note += (" Z-scoring is per channel within this slice only — it "
                     "makes shapes comparable, not magnitudes; raw units live "
                     "in the stacked layout.")
        st.caption(note)

        if show_rows:
            st.dataframe(seq, width="stretch", hide_index=True)
            st.download_button(
                "Download this slice as CSV",
                seq.to_csv(index=False).encode(),
                file_name=f"{district}_{crop}_"
                          f"{'-'.join(str(y) for y in sorted(years))}.csv",
                mime="text/csv")

st.markdown("---")


# --------------------------------------------------------------------------- #
# SECONDARY 1 — coverage detail: how the panel got to its final size
# --------------------------------------------------------------------------- #
with st.expander(
        f"Coverage detail — district funnel, season completeness, "
        f"{len(panel) - len(kept):,} sequences dropped at the "
        f"{min_cov:.0%} floor", expanded=False):

    st.markdown("**Where districts are lost**")
    stages = source_districts(SOURCE_SIGS)
    if stages:
        stage_df = pd.DataFrame(
            [{"Stage": k, "Districts": len(v)} for k, v in stages.items()])
        fig = px.bar(stage_df, x="Stage", y="Districts", text="Districts")
        fig.update_traces(marker_color=SERIES_COLORS[0],
                          marker_cornerradius=4, textposition="outside")
        fig.update_layout(height=330, yaxis_title="NUTS-3 districts present",
                          xaxis_title="", margin=dict(t=10, b=0))
        st.plotly_chart(fig, width="stretch")

        geo = stages.get("NUTS-3 geography", set())
        yld = stages.get("Yield labels", set())
        wx = stages.get("DWD weather", set())
        final = stages.get("Unified panel", set())
        loss = pd.DataFrame([
            {"Step": "NUTS-3 polygons in the geography file",
             "Districts": len(geo), "Lost here": 0},
            {"Step": "…that also have a Destatis yield label",
             "Districts": len(geo & yld), "Lost here": len(geo - yld)},
            {"Step": "…that also have DWD weather",
             "Districts": len(geo & yld & wx), "Lost here": len((geo & yld) - wx)},
            {"Step": "…present in dataset.parquet",
             "Districts": len(final), "Lost here": len((geo & yld & wx) - final)},
        ])
        st.dataframe(loss, width="stretch", hide_index=True)
        if yld and wx:
            st.caption(
                f"The binding constraint is weather, not geography: "
                f"{len(yld - wx)} districts carry a yield label but no DWD "
                f"station coverage. Satellite and soil cover all "
                f"{len(stages.get('Satellite', geo))} polygons.")

    st.markdown("---")
    st.markdown("**Growing-season completeness**")
    left, right = st.columns([3, 2])
    with left:
        hist = px.histogram(panel, x="coverage", nbins=60,
                            color_discrete_sequence=[SERIES_COLORS[0]])
        hist.add_vline(x=min_cov, line_dash="dash", line_color=SERIES_COLORS[7],
                       annotation_text=f"floor {min_cov:.0%}",
                       annotation_position="top left")
        hist.update_layout(height=330, yaxis_type="log",
                           xaxis_title="observed DOYs / configured season length",
                           yaxis_title="sequences (log)",
                           margin=dict(t=10, b=0), showlegend=False)
        st.plotly_chart(hist, width="stretch")
    with right:
        per_crop = (panel.assign(dropped=panel["coverage"] < min_cov)
                    .groupby("crop")
                    .agg(sequences=("coverage", "size"),
                         dropped=("dropped", "sum"),
                         season_days=("expected_days", "max"))
                    .reset_index())
        per_crop["dropped %"] = (100 * per_crop["dropped"]
                                 / per_crop["sequences"]).round(2)
        st.dataframe(per_crop.rename(columns={
            "crop": "Crop", "sequences": "Sequences", "dropped": "Dropped",
            "season_days": "Season length (d)"}),
            width="stretch", hide_index=True)

    st.markdown("---")
    st.markdown("**Sequences per year**")
    per_year = (kept.groupby("year")
                .agg(districts=("landkreis", "nunique"),
                     sequences=("crop", "size")).reset_index())
    fy = go.Figure()
    fy.add_bar(x=per_year["year"], y=per_year["sequences"],
               name="Crop-year sequences", marker_color=SERIES_COLORS[0],
               marker_cornerradius=4)
    fy.add_bar(x=per_year["year"], y=per_year["districts"],
               name="Distinct districts", marker_color=SERIES_COLORS[1],
               marker_cornerradius=4)
    fy.update_layout(barmode="group", height=330, bargap=0.25,
                     xaxis_title="", yaxis_title="count",
                     margin=dict(t=10, b=0))
    st.plotly_chart(fy, width="stretch")

    st.markdown("---")
    st.markdown(f"**Yield label history — "
                f"{labelled.get(district, str(district))}**")
    hist_rows = panel[panel["landkreis"] == district].sort_values(["crop", "year"])
    if not hist_rows.empty:
        yfig = px.line(hist_rows, x="year", y="yield_dt_per_ha", color="crop",
                       markers=True, color_discrete_sequence=SERIES_COLORS)
        yfig.update_traces(line=dict(width=2), marker=dict(size=6))
        yfig.update_layout(height=360, xaxis_title="",
                           yaxis_title="Yield (dt/ha)",
                           legend_title_text="", margin=dict(t=10, b=0))
        st.plotly_chart(yfig, width="stretch")
        st.caption("Follows the district picked in the viewer above. Gaps are "
                   "missing reporting years, not zeros — the panel is "
                   "unbalanced and the line is broken where no label exists.")


# --------------------------------------------------------------------------- #
# SECONDARY 2 — feature inventory, the duplicate audit, and the policy
# --------------------------------------------------------------------------- #
role_of: dict[str, str] = {}
for role in ("past_observed", "static", "known_future", "excluded"):
    for name in features_cfg.get(role) or []:
        role_of[name] = role

feature_roles = ("past_observed", "static", "known_future")
feature_order = {name: i for i, name in enumerate(
    n for role in feature_roles for n in (features_cfg.get(role) or []))}


def _rank(col: str) -> tuple:
    """Declared features lead their equivalence class, in config order.

    Which member is called the original decides which members are called
    redundant, so the representative must be the column a model actually
    reads — otherwise `soil_imputed` would be reported as a duplicate of
    `sand_imputed`, which no model ever sees.
    """
    return (col not in feature_order, feature_order.get(col, 0), col)


dup_classes = [sorted(cls, key=_rank)
               for cls in duplicate_classes(PQ_SIG, tuple(profile["numeric"]))]
dup_of: dict[str, str] = {}
for cls in dup_classes:
    for member in cls[1:]:
        dup_of[member] = cls[0]

# The duplicate count rides in the expander label so a known defect is never
# hidden behind a neutral word.
inv_label = (f"Feature inventory and column audit — {profile['n_columns']} "
             f"columns")
if dup_of:
    inv_label += f", {len(dup_of)} byte-identical duplicates"

with st.expander(inv_label, expanded=False):
    moments = column_moments(PQ_SIG, tuple(profile["numeric"]))
    moment_map = moments.set_index("column").to_dict("index")

    rows = []
    for col in profile["columns"]:
        name = col["column"]
        if name in KEY_COLUMNS:
            role = "key"
        elif name == TARGET_COLUMN:
            role = "target"
        else:
            role = role_of.get(name, "unused")
        m = moment_map.get(name, {})
        nulls = col["nulls"]
        rows.append({
            "Column": name,
            "Role": role,
            "dtype": col["dtype"],
            "Missing %": (round(100 * nulls / profile["rows"], 3)
                          if nulls is not None else None),
            "Min": col["min"], "Mean": m.get("mean"), "Max": col["max"],
            "Std": m.get("std"), "Distinct": m.get("n_distinct"),
            "Duplicate of": dup_of.get(name, ""),
        })
    inv = pd.DataFrame(rows)

    role_counts = inv["Role"].value_counts()
    m1, m2, m3, m4 = st.columns(4)
    with m1: kpi("Past-observed channels", int(role_counts.get("past_observed", 0)))
    with m2: kpi("Static covariates", int(role_counts.get("static", 0)))
    with m3: kpi("Excluded (leaky/broken)", int(role_counts.get("excluded", 0)))
    with m4: kpi("Exact-duplicate columns", len(dup_of))

    if dup_classes:
        redundant = [c for c in dup_of if c in feature_order]
        by_role = pd.Series([role_of[c] for c in redundant]).value_counts()
        breakdown = ", ".join(
            f"{n} of the {int(role_counts.get(r, 0))} {r.replace('_', '-')} "
            f"channels" for r, n in by_role.items())
        st.error(
            f"{len(dup_of)} of {len(profile['numeric'])} numeric columns are "
            f"byte-identical to another column across all "
            f"{profile['rows']:,} rows — compared element-wise, not sampled. "
            f"{len(redundant)} of them are declared model features and carry "
            f"zero incremental information ({breakdown}): "
            f"{', '.join(sorted(redundant))}.")
        dup_table = pd.DataFrame([{
            "Kept as the original": cls[0],
            "Byte-identical to it": ", ".join(cls[1:]),
            "Redundant model features in this class": ", ".join(
                c for c in cls[1:] if c in feature_order) or "—",
        } for cls in dup_classes])
        st.dataframe(dup_table, width="stretch", hide_index=True)

    st.markdown("---")
    st.markdown("**Per-column profile**")
    f1, f2 = st.columns([2, 3])
    with f1:
        role_pick = st.multiselect(
            "Roles", sorted(inv["Role"].unique()),
            default=[r for r in ["past_observed", "static", "known_future"]
                     if r in set(inv["Role"])],
            key="role_pick")
    with f2:
        needle = st.text_input("Filter column names", "", key="col_needle")
    view = inv[inv["Role"].isin(role_pick)] if role_pick else inv
    if needle:
        view = view[view["Column"].str.contains(needle, case=False, regex=False)]
    st.dataframe(
        view, width="stretch", hide_index=True,
        column_config={
            "Missing %": st.column_config.NumberColumn(format="%.3f"),
            "Mean": st.column_config.NumberColumn(format="%.4g"),
            "Std": st.column_config.NumberColumn(format="%.4g"),
        })
    st.caption("Missing %, Min and Max come from parquet row-group statistics; "
               "Mean, Std and Distinct are streamed one column at a time.")

    st.markdown("---")
    st.markdown("**Feature policy**")
    reasons = read_list_comments(CFG_SIG, "excluded")
    excluded = features_cfg.get("excluded") or []
    if excluded:
        null_by_col = {c["column"]: c["nulls"] for c in profile["columns"]}
        exc = pd.DataFrame([{
            "Excluded feature": name,
            "In parquet": name in profile["names"],
            "Missing %": (round(100 * null_by_col[name] / profile["rows"], 2)
                          if null_by_col.get(name) is not None else None),
            "Reason (from configs/data.yaml)": reasons.get(name, ""),
        } for name in excluded])
        st.dataframe(
            exc, width="stretch", hide_index=True,
            column_config={"Missing %": st.column_config.NumberColumn(format="%.2f")})
        st.caption("Reasons are lifted from the comments in `configs/data.yaml`; "
                   "missingness is measured on the parquet. Excluded columns "
                   "remain in the file — `build_feature_spec` drops them at "
                   "training time.")

    declared = {name for role in ("past_observed", "static", "known_future")
                for name in (features_cfg.get(role) or [])}
    present = set(profile["names"])
    derived_prefix = "crop_is_"
    absent = sorted(declared - present)
    silently_dropped = [a for a in absent if not a.startswith(derived_prefix)]
    derived = [a for a in absent if a.startswith(derived_prefix)]
    unused = sorted(inv.loc[inv["Role"] == "unused", "Column"])

    ic1, ic2 = st.columns(2)
    with ic1:
        st.markdown("**Declared in `data.yaml`, absent from the parquet**")
        if silently_dropped or derived:
            st.dataframe(pd.DataFrame(
                [{"Feature": f, "Fate": "silently dropped by build_feature_spec"}
                 for f in silently_dropped] +
                [{"Feature": f, "Fate": "generated at train time by "
                                        "add_crop_indicators"} for f in derived]),
                width="stretch", hide_index=True)
        else:
            st.success("Every declared feature exists in the parquet.")
        if silently_dropped:
            st.warning(
                f"{len(silently_dropped)} configured feature(s) never reach a "
                "model: `build_feature_spec` keeps only columns present in the "
                "frame, without warning. The config overstates the feature set.")
    with ic2:
        st.markdown("**In the parquet, assigned no role**")
        if unused:
            st.dataframe(pd.DataFrame({"Column": unused}),
                         width="stretch", hide_index=True)
            st.caption("Carried in the file but never read as a feature, key, "
                       "or target.")
        else:
            st.success("Every parquet column has a role.")

    st.markdown("---")
    st.markdown("**Channel correlation**")
    past = [c for c in (features_cfg.get("past_observed") or []) if c in present]
    if len(past) >= 2:
        stride = st.select_slider(
            "Subsample stride (every Nth row)", options=[5, 11, 17, 41, 101],
            value=17, key="corr_stride")
        corr, n_sample = channel_correlation(PQ_SIG, tuple(past), stride)
        cfig = px.imshow(corr, color_continuous_scale="RdBu_r",
                         zmin=-1, zmax=1, aspect="auto",
                         labels=dict(color="Pearson r"))
        cfig.update_layout(height=max(420, 22 * len(past)),
                           margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(cfig, width="stretch")
        st.caption(f"Pearson r on a systematic subsample of {n_sample:,} of "
                   f"{profile['rows']:,} rows (every {stride}th). Solid blocks "
                   f"of r = 1 off the diagonal are the duplicate columns "
                   f"listed above.")


# --------------------------------------------------------------------------- #
# SECONDARY 3 — sources, provenance, and config drift
# --------------------------------------------------------------------------- #
cadence = satellite_cadence(sig(INTERIM / "satellite.parquet"))
cfg_cadence = cfg.get("satellite_composite_days")
cadence_drift = bool(cadence and cfg_cadence is not None
                     and int(cfg_cadence) != cadence[0])

# Same principle as the duplicate audit: the live discrepancy count is in the
# label, so collapsing the section does not conceal that something disagrees.
src_label = "Sources and provenance — per-source extent, licences, raw rows"
if cadence_drift:
    src_label += " · 1 config mismatch"

with st.expander(src_label, expanded=False):
    st.markdown("**What each source actually contributes**")
    ext_sigs = tuple(sig(INTERIM / n) for n in
                     ["weather.parquet", "satellite.parquet",
                      "soil.parquet", "yield.parquet"])
    st.dataframe(source_extent(ext_sigs), width="stretch", hide_index=True)

    if cadence:
        median_gap, n_dates = cadence
        msg = (f"Satellite: {n_dates:,} distinct observation dates, median gap "
               f"{median_gap} days.")
        if cadence_drift:
            st.warning(msg + f" `configs/data.yaml` declares "
                             f"`satellite_composite_days: {cfg_cadence}`, which "
                             f"the data does not support — the observed cadence "
                             f"is the MODIS native {median_gap}-day composite.")
        else:
            st.caption(msg)

    years_cfg = cfg.get("years") or {}
    if years_cfg:
        st.caption(f"`configs/data.yaml` requests "
                   f"{years_cfg.get('start')}–{years_cfg.get('end')}; the "
                   f"parquet holds {int(panel['year'].min())}–"
                   f"{int(panel['year'].max())}.")

    st.caption(f"Feature policy v{v_data if v_data is not None else '?'} "
               f"(code v{v_code if v_code is not None else '?'}) · "
               f"parquet {profile['size_mb']:.0f} MB in "
               f"{profile['row_groups']} row groups · "
               f"coverage floor {min_cov:.0%} of the configured growing season.")

    st.markdown("---")
    st.markdown("**Provenance strings (static, from dashboard/utils.py)**")
    st.caption("Source, licence and URL only. The counts and year ranges in "
               "this table are hand-maintained strings and are not derived "
               "from the build — trust the live table above for those.")
    src_df = pd.DataFrame(DATA_SOURCES)
    src_df["source"] = src_df["source"] + " — " + src_df["url"]
    st.dataframe(
        src_df.drop(columns=["url"]).rename(columns={
            "modality": "Modality", "source": "Source",
            "resolution": "Resolution", "variables": "Variables",
            "years": "Years", "license": "License"}),
        width="stretch", hide_index=True)

    st.markdown("---")
    st.markdown("**Unified dataset — first rows**")
    head = pq.ParquetFile(PROCESSED).read_row_group(0).slice(0, 200).to_pandas()
    st.dataframe(head, width="stretch", hide_index=True)

    st.markdown("---")
    st.markdown("**Soil property distributions across districts**")
    soil_path = INTERIM / "soil.parquet"
    if soil_path.exists():
        soil = pd.read_parquet(soil_path)
        prop = st.selectbox(
            "Property",
            [c for c in soil.columns
             if c != "landkreis" and not c.endswith("_imputed")],
            key="soil_prop")
        sfig = px.histogram(soil, x=prop, nbins=40,
                            color_discrete_sequence=[SERIES_COLORS[0]])
        sfig.update_layout(height=330, yaxis_title="districts",
                           margin=dict(t=10, b=0), bargap=0.02)
        st.plotly_chart(sfig, width="stretch")
    else:
        st.info("Soil data missing. Run `01_download_data.py --source soilgrids`.")

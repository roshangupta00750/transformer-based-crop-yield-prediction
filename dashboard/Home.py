"""Home — status board for the crop-yield benchmark.

Answers four questions from live artefacts only: what data is on disk, what
is trained, how good it scores, and whether the scores still describe the
checkpoints they claim to describe. Every number is read at render time —
nothing on this page is hardcoded, because the benchmark is retrained often
and any literal would go stale silently.

Streamlit auto-discovers this file when invoked as ``streamlit run
dashboard/Home.py``. For the grouped sidebar (7 collapsible sections
instead of 19 flat entries) run via ``streamlit run dashboard/nav.py``.
"""
from __future__ import annotations

# Pin OpenMP / MKL to one thread *before* any library that ships its own
# libomp.dylib is imported.
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import (CHECKPOINTS_DIR, METRICS_DIR, MODELS, PROCESSED, divider,
                   kpi, load_multicrop_summary, load_taxonomy,
                   results_policy_is_current)

st.set_page_config(
    page_title="Crop Yield Benchmark — JMU Würzburg",
    layout="wide",
    initial_sidebar_state="expanded",
)

DASHBOARD_DIR = Path(__file__).resolve().parent
PAGES = DASHBOARD_DIR / "pages"

CKPT_EXTS = (".pt", ".ubj")
SUMMARY_PATH = METRICS_DIR / "multicrop_summary.csv"
TAXONOMY_VALIDATION_PATH = METRICS_DIR / "taxonomy_validation.csv"
RESULTS_POLICY_PATH = METRICS_DIR / "feature_policy.json"

# Diverging ramp for R², which has a meaningful zero: 0 = no better than
# predicting the crop mean, negative = worse than that. Two hues with a
# neutral midpoint pinned to zero, so sign reads without the colourbar.
# Both ends stop short of the darkest ColorBrewer steps so that CELL_INK
# clears 4.5:1 on every cell in either Streamlit theme — the fills are
# opaque, so contrast is against the fill, not the page background.
R2_SCALE = [
    [0.00, "#d6604d"], [0.25, "#f4a582"],
    [0.50, "#f7f7f7"],
    [0.75, "#92c5de"], [1.00, "#4393c3"],
]
# Error metrics are one-directional magnitudes: single hue, light to dark,
# warm because larger always means worse.
ERROR_SCALE = [
    [0.00, "#fff5eb"], [0.33, "#fdd0a2"], [0.66, "#fdae6b"],
    [1.00, "#f16913"],
]
CELL_INK = "#1a1a1a"
METRIC_CHOICES = {
    "R² (higher is better)": "r2",
    "RMSE dt/ha (lower is better)": "rmse",
    "MAE dt/ha (lower is better)": "mae",
}


def _mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def _stamp(ts: float | None) -> str:
    if ts is None:
        return "missing"
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")


# --------------------------------------------------------------------------- #
# Cached readers — each returns only small aggregates so the 16 M-row parquet
# is never held in session memory.
# --------------------------------------------------------------------------- #
@st.cache_data(ttl=300, show_spinner="Reading dataset…")
def dataset_facts() -> dict:
    """Shape of `data/processed/dataset.parquet`, read column-subset only."""
    if not PROCESSED.exists():
        return {}
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    meta = pq.ParquetFile(PROCESSED).metadata
    table = pq.read_table(
        PROCESSED, columns=["crop", "landkreis", "year", "yield_dt_per_ha"]
    )
    panel = table.group_by(["crop", "landkreis", "year"]).aggregate(
        [("yield_dt_per_ha", "count")]
    )
    per_crop = (
        panel.group_by("crop")
        .aggregate([("landkreis", "count_distinct"),
                    ("year", "min"), ("year", "max"), ("year", "count")])
        .to_pandas()
        .rename(columns={"crop": "Crop",
                         "landkreis_count_distinct": "Districts",
                         "year_min": "First year",
                         "year_max": "Last year",
                         "year_count": "Crop-district-years"})
        .sort_values("Crop-district-years", ascending=False)
        .reset_index(drop=True)
    )
    return {
        "daily_rows": meta.num_rows,
        "columns": meta.num_columns,
        "crops": len(pc.unique(table["crop"])),
        "districts": len(pc.unique(table["landkreis"])),
        "year_min": pc.min(table["year"]).as_py(),
        "year_max": pc.max(table["year"]).as_py(),
        "panel_rows": panel.num_rows,
        "per_crop": per_crop,
        "mtime": _mtime(PROCESSED),
    }


@st.cache_data(ttl=30)
def checkpoint_inventory(crops: tuple[str, ...]) -> pd.DataFrame:
    """One row per model family: which checkpoints exist and when written.

    Matches exact filenames only — `<model>.pt|.ubj` for the pooled run and
    `<model>_<crop>.pt|.ubj` for per-crop runs — so experiment variants such
    as `xgboost_abs` or `tft.55m` are not miscounted as production artefacts.
    """
    rows = []
    for model in MODELS:
        pooled = next(
            (CHECKPOINTS_DIR / f"{model}{e}" for e in CKPT_EXTS
             if (CHECKPOINTS_DIR / f"{model}{e}").exists()), None
        )
        per_crop = []
        for crop in crops:
            for e in CKPT_EXTS:
                fp = CHECKPOINTS_DIR / f"{model}_{crop}{e}"
                if fp.exists():
                    per_crop.append(fp)
                    break
        files = ([pooled] if pooled is not None else []) + per_crop
        stamps = [t for t in (_mtime(f) for f in files) if t is not None]
        rows.append({
            "Model": model,
            "Pooled": pooled is not None,
            "Per-crop": len(per_crop),
            "_newest": max(stamps) if stamps else None,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Header + policy gate
# --------------------------------------------------------------------------- #
st.markdown(
    "# Transformer-based Crop Yield Prediction\n"
    "**Praktikum — Julius-Maximilians-Universität Würzburg · "
    "Institut für Informatik**"
)

if not results_policy_is_current():
    st.error(
        "The processed dataset is now feature-policy v3, but the bundled model "
        "metrics are legacy. They are intentionally hidden to prevent reporting "
        "leakage-contaminated results. Retrain the models and run "
        "`python scripts/05b_eval_multicrop.py`."
    )
    st.stop()

bar_left, bar_right = st.columns([4, 1])
with bar_left:
    st.caption(f"Read from disk at {datetime.now():%Y-%m-%d %H:%M:%S}. "
               "Cached readers hold values for up to 5 minutes.")
with bar_right:
    if st.button("Reload from disk", width="stretch"):
        st.cache_data.clear()
        st.rerun()

# --------------------------------------------------------------------------- #
# Data on disk
# --------------------------------------------------------------------------- #
divider("Data")

facts = dataset_facts()
if not facts:
    st.warning(
        f"`{PROCESSED}` not found. Build it with "
        "`python scripts/02_build_dataset.py` before training or scoring."
    )
else:
    d1, d2, d3, d4, d5 = st.columns(5)
    with d1:
        kpi("Crop-district-years", f"{facts['panel_rows']:,}")
    with d2:
        kpi("Daily rows", f"{facts['daily_rows']:,}")
    with d3:
        kpi("Crops", f"{facts['crops']}")
    with d4:
        kpi("NUTS-3 districts", f"{facts['districts']}")
    with d5:
        kpi("Years", f"{facts['year_min']}-{facts['year_max']}")

    with st.expander("Coverage per crop"):
        st.dataframe(
            facts["per_crop"],
            hide_index=True,
            width="stretch",
            column_config={
                "Crop-district-years": st.column_config.NumberColumn(
                    format="%d"),
                "First year": st.column_config.NumberColumn(format="%d"),
                "Last year": st.column_config.NumberColumn(format="%d"),
            },
        )
    st.caption(f"{PROCESSED.name} · {facts['columns']} columns · "
               f"written {_stamp(facts['mtime'])}")

# --------------------------------------------------------------------------- #
# What is trained
# --------------------------------------------------------------------------- #
divider("Trained models")

multicrop = load_multicrop_summary()
crop_list = tuple(sorted(facts["per_crop"]["Crop"])) if facts else (
    tuple(sorted(multicrop["crop"].unique())) if not multicrop.empty else ()
)
inventory = checkpoint_inventory(crop_list)
summary_mtime = _mtime(SUMMARY_PATH)

if multicrop.empty:
    st.warning(
        f"`{SUMMARY_PATH}` not found. Run "
        "`python scripts/05b_eval_multicrop.py` to score the checkpoints."
    )
    scored = pd.DataFrame(columns=["Model", "Crops scored", "Mean R²",
                                   "Best crop", "Best R²"])
else:
    best_idx = multicrop.groupby("model")["r2"].idxmax()
    best = multicrop.loc[best_idx].set_index("model")
    scored = pd.DataFrame({
        "Model": multicrop.groupby("model")["crop"].nunique().index,
        "Crops scored": multicrop.groupby("model")["crop"].nunique().values,
        "Mean R²": multicrop.groupby("model")["r2"].mean().values,
    })
    scored["Best crop"] = scored["Model"].map(best["crop"])
    scored["Best R²"] = scored["Model"].map(best["r2"])

status = inventory.merge(scored, on="Model", how="outer")
# A model scored from an artefact that is no longer on disk merges in with
# no inventory columns; show it as "no checkpoint" rather than as NaN.
status["Pooled"] = status["Pooled"].fillna(False).astype(bool)
status["Per-crop"] = status["Per-crop"].fillna(0).astype(int)
status["Checkpoint written"] = status["_newest"].map(_stamp)
status["Scores current"] = [
    bool(summary_mtime is not None and n is not None and summary_mtime >= n)
    for n in status["_newest"]
]
status = status.sort_values("Mean R²", ascending=False, na_position="last")

st.dataframe(
    status[["Model", "Pooled", "Per-crop", "Checkpoint written",
            "Crops scored", "Mean R²", "Best crop", "Best R²",
            "Scores current"]],
    hide_index=True,
    width="stretch",
    column_config={
        "Pooled": st.column_config.CheckboxColumn(
            help="An all-crop checkpoint <model>.pt or <model>.ubj exists"),
        "Per-crop": st.column_config.NumberColumn(
            format="%d",
            help="Count of <model>_<crop> checkpoints on disk"),
        "Crops scored": st.column_config.NumberColumn(
            format="%d",
            help="Crops this model has a row for in multicrop_summary.csv"),
        "Mean R²": st.column_config.NumberColumn(
            format="%.3f",
            help="Unweighted mean of this model's per-crop test-split R². "
                 "A row with scores but no checkpoint was scored from an "
                 "artefact that is no longer on disk."),
        "Best R²": st.column_config.NumberColumn(format="%.3f"),
        "Scores current": st.column_config.CheckboxColumn(
            help="multicrop_summary.csv is at least as new as this model's "
                 "newest checkpoint"),
    },
)

t1, _ = st.columns([1, 4])
with t1:
    try:
        st.page_link(str(PAGES / "17_Train.py"), label="Train a model",
                     width="stretch")
    except Exception:
        st.caption("Train page: pages/17_Train.py")

# --------------------------------------------------------------------------- #
# Freshness chain
# --------------------------------------------------------------------------- #
divider("Artefact freshness")

newest_ckpt = status["_newest"].dropna()
chain = [
    ("data/processed/dataset.parquet", facts.get("mtime") if facts else None),
    ("results/checkpoints (newest)",
     float(newest_ckpt.max()) if not newest_ckpt.empty else None),
    ("results/metrics/multicrop_summary.csv", summary_mtime),
    ("results/metrics/taxonomy_validation.csv",
     _mtime(TAXONOMY_VALIDATION_PATH)),
]
# Each stage must be newer than *everything* upstream of it, not merely than
# the stage directly above — otherwise one stale stage lowers the bar for the
# next and the rot stops being visible.
fresh_rows, stale = [], []
up_name, up_ts = None, None
for name, ts in chain:
    if ts is None:
        state = "Missing"
    elif up_ts is not None and ts < up_ts:
        state = f"Stale — predates {up_name}"
        stale.append(name)
    else:
        state = "Current"
    fresh_rows.append({"Artefact": name, "Last written": _stamp(ts),
                       "Status": state})
    if ts is not None and (up_ts is None or ts > up_ts):
        up_name, up_ts = name, ts

st.dataframe(pd.DataFrame(fresh_rows), hide_index=True, width="stretch")
if stale:
    st.warning(
        "Rebuild these before quoting anything downstream of them: "
        + ", ".join(f"`{s}`" for s in stale)
        + ". Each stage must be newer than every stage above it."
    )
st.caption(f"Results feature policy: {RESULTS_POLICY_PATH.name}, written "
           f"{_stamp(_mtime(RESULTS_POLICY_PATH))}.")

# --------------------------------------------------------------------------- #
# Scores
# --------------------------------------------------------------------------- #
divider("Held-out test scores by crop and model")

if multicrop.empty:
    st.info("No scores to plot until `multicrop_summary.csv` exists.")
else:
    canonical = [m for m in MODELS if m in set(multicrop["model"])]
    extra = [m for m in sorted(multicrop["model"].unique())
             if m not in canonical]
    all_models = canonical + extra

    ctrl1, ctrl2, ctrl3 = st.columns([2, 3, 2])
    with ctrl1:
        metric_label = st.selectbox("Metric", list(METRIC_CHOICES))
        metric = METRIC_CHOICES[metric_label]
    with ctrl2:
        shown = st.multiselect("Models", all_models, default=all_models)
    with ctrl3:
        overlay = st.checkbox("Mark taxonomy prediction", value=False)

    subset = multicrop[multicrop["model"].isin(shown)]
    if subset.empty or metric not in subset.columns:
        st.info("Select at least one model to draw the heatmap.")
    else:
        pivot = subset.pivot_table(index="crop", columns="model",
                                   values=metric, aggfunc="max")
        pivot = pivot[[m for m in all_models if m in pivot.columns]]

        is_r2 = metric == "r2"
        fig = go.Figure(
            data=go.Heatmap(
                z=pivot.values,
                x=list(pivot.columns),
                y=list(pivot.index),
                colorscale=R2_SCALE if is_r2 else ERROR_SCALE,
                zmid=0 if is_r2 else None,
                colorbar=dict(title=metric_label.split(" ")[0]),
                texttemplate="%{z:.2f}",
                textfont=dict(size=11, color=CELL_INK),
                xgap=2, ygap=2,
                hovertemplate=("Crop: %{y}<br>Model: %{x}<br>"
                               f"{metric}: %{{z:.3f}}<extra></extra>"),
            )
        )

        hits = 0
        if overlay:
            winners = (load_taxonomy() or {}).get("predicted_winners", {})
            crop_order, model_order = list(pivot.index), list(pivot.columns)
            shapes = []
            for yi, crop in enumerate(crop_order):
                pred = (winners.get(crop) or {}).get("primary")
                if pred not in model_order:
                    continue
                xi = model_order.index(pred)
                shapes.append(dict(
                    type="rect", x0=xi - 0.5, x1=xi + 0.5,
                    y0=yi - 0.5, y1=yi + 0.5,
                    line=dict(color="#111111", width=3),
                    fillcolor="rgba(0,0,0,0)", layer="above",
                ))
                row = pivot.loc[crop].dropna()
                if not row.empty:
                    winner = (row.idxmax() if is_r2 else row.idxmin())
                    hits += int(winner == pred)
            fig.update_layout(shapes=shapes)

        fig.update_layout(
            height=460, margin=dict(l=10, r=10, t=30, b=10),
            xaxis_title="Model", yaxis_title="Crop",
        )
        st.plotly_chart(fig, width="stretch")

        if overlay:
            o1, _ = st.columns([1, 3])
            with o1:
                # st.metric direct, not utils.kpi, so the encoding note can
                # ride along as a tooltip instead of another line of prose.
                st.metric("Outline is the best cell in its row",
                          f"{hits}/{len(pivot.index)} crops",
                          help="Outline marks the a-priori primary from "
                               "configs/taxonomy.yaml, scored against the "
                               "metric selected above.")

        if is_r2:
            st.caption("Blue above zero, red below; zero is no better than "
                       "predicting the crop mean. Source: "
                       "multicrop_summary.csv.")
        else:
            st.caption("Darker is larger error, in dt/ha. Source: "
                       "multicrop_summary.csv.")

        with st.expander("Table view"):
            st.dataframe(pivot.round(3), width="stretch")

# --------------------------------------------------------------------------- #
# Navigation
# --------------------------------------------------------------------------- #
divider("Go to")

nav_targets = [
    (PAGES / "02_Data.py", "Data"),
    (PAGES / "17_Train.py", "Train models"),
    (PAGES / "09_Leaderboard.py", "Leaderboard"),
    (PAGES / "13_Taxonomy.py", "Crop-model taxonomy"),
]
for target, col in zip(nav_targets, st.columns(len(nav_targets))):
    page, label = target
    with col:
        try:
            st.page_link(str(page), label=label, width="stretch")
        except Exception:
            st.caption(f"{label}: pages/{page.name}")

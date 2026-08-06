"""In-season horizon analysis.

How accuracy degrades the further before harvest a prediction is made, per
crop and per architecture, against a horizon-independent climatology
reference.

Data sources, both read from disk:
    results/metrics/multicrop_horizons.csv   (scripts/05c_eval_horizons.py)
    results/metrics/climatology_baseline.csv (computed here when stale)
"""
from __future__ import annotations

import datetime as _dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import (CHECKPOINTS_DIR, METRICS_DIR, MODELS, PROCESSED,
                   PROJECT_ROOT, divider, kpi, load_multicrop_summary,
                   page_header)

st.set_page_config(page_title="In-Season Horizons — Crop Yield",
                   layout="wide")

TRANSFORMERS = ("itransformer", "patchtst", "tft")
BASELINES = ("lstm", "xgboost", "dlinear")
MODEL_ORDER = list(TRANSFORMERS) + list(BASELINES)
# Categorical slots 1-6, fixed order; class is carried redundantly by dash
# style so identity never rests on colour alone.
MODEL_COLOR = {"itransformer": "#2a78d6", "patchtst": "#eb6834",
               "tft": "#1baf7a", "lstm": "#eda100", "xgboost": "#e87ba4",
               "dlinear": "#008300"}
REFERENCE_INK = "#52514e"

CLIMATOLOGY_CSV = METRICS_DIR / "climatology_baseline.csv"
HORIZONS_CSV = METRICS_DIR / "multicrop_horizons.csv"


def model_class(model: str) -> str:
    return "Transformer" if model in TRANSFORMERS else "Baseline"


# --------------------------------------------------------------------------- #
# Climatology (null) reference — horizon-independent by construction: it uses
# no in-season observations at all.
# --------------------------------------------------------------------------- #
@st.cache_data(ttl=300, show_spinner="Computing climatology reference…")
def load_climatology() -> pd.DataFrame:
    """Per-crop R² of predicting each district's train-fitted baseline yield.

    Scored on exactly the (district, crop, year) population the benchmark
    scores models on. Cached to ``results/metrics/climatology_baseline.csv``
    and recomputed when the dataset or a fitted pipeline is newer.
    """
    pipes = [CHECKPOINTS_DIR / f"{m}.pipeline.json" for m in MODELS]
    pipes = [p for p in pipes if p.exists()]
    if not pipes or not PROCESSED.exists():
        return pd.DataFrame()
    newest = max(p.stat().st_mtime for p in pipes + [PROCESSED])
    if CLIMATOLOGY_CSV.exists() and CLIMATOLOGY_CSV.stat().st_mtime >= newest:
        return pd.read_csv(CLIMATOLOGY_CSV)

    sys.path.insert(0, str(PROJECT_ROOT))
    try:
        from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                                     r2_score)
        from src.data.dataset import (FeaturePipeline, apply_phenology_window,
                                      filter_incomplete_sequences)
        from src.utils import load_data_config
    except Exception:  # noqa: BLE001
        return pd.DataFrame()

    cfg = load_data_config()
    phen = cfg["phenology"]
    df = pd.read_parquet(
        PROCESSED,
        columns=["landkreis", "crop", "year", "doy", "yield_dt_per_ha"],
    )
    df = df[df["year"].isin(cfg["splits"]["temporal"]["test_years"])]
    df = df.dropna(subset=["yield_dt_per_ha"])
    df = filter_incomplete_sequences(
        apply_phenology_window(df, phen), phen,
        float(cfg.get("quality", {}).get("min_sequence_coverage", 0.90)))
    keys = df.drop_duplicates(subset=["landkreis", "crop", "year"])
    pipe = FeaturePipeline.load(pipes[0])

    rows = []
    for crop, g in keys.groupby("crop"):
        y = g["yield_dt_per_ha"].to_numpy(float)
        p = np.array([pipe.baseline_yield(lk, crop, yr)
                      for lk, yr in zip(g["landkreis"], g["year"])])
        rows.append({"crop": crop, "r2": float(r2_score(y, p)),
                     "rmse": float(np.sqrt(mean_squared_error(y, p))),
                     "mae": float(mean_absolute_error(y, p)), "n": int(len(g))})
    out = pd.DataFrame(rows).sort_values("crop").reset_index(drop=True)
    try:
        tmp = CLIMATOLOGY_CSV.with_suffix(".csv.tmp")
        out.to_csv(tmp, index=False)
        tmp.replace(CLIMATOLOGY_CSV)
    except OSError:
        pass
    return out


def newest_checkpoint_mtime() -> float:
    if not CHECKPOINTS_DIR.exists():
        return 0.0
    times = [p.stat().st_mtime for p in CHECKPOINTS_DIR.glob("*.pt")]
    return max(times) if times else 0.0


page_header(
    "In-season horizon analysis",
    "R² as a function of days before harvest, per crop and per architecture, "
    "against the climatology reference.",
)

if not HORIZONS_CSV.exists():
    st.warning(
        "`results/metrics/multicrop_horizons.csv` not found. Generate it with "
        "`python scripts/05c_eval_horizons.py`."
    )
    st.stop()

df = pd.read_csv(HORIZONS_CSV)
if df.empty:
    st.warning("`results/metrics/multicrop_horizons.csv` is empty.")
    st.stop()

clim = load_climatology()
clim_by_crop = dict(zip(clim["crop"], clim["r2"])) if not clim.empty else {}

# --------------------------------------------------------------------------- #
# Provenance gate — this file is regenerated separately from the per-crop
# evaluation, so it can silently describe superseded checkpoints.
# --------------------------------------------------------------------------- #
_h_mtime = HORIZONS_CSV.stat().st_mtime
_ckpt = newest_checkpoint_mtime()
_fmt = "%Y-%m-%d %H:%M"
IS_STALE = _ckpt > _h_mtime

if IS_STALE:
    st.error(
        f"**These horizon numbers describe superseded checkpoints.** "
        f"`multicrop_horizons.csv` was written "
        f"{_dt.datetime.fromtimestamp(_h_mtime):{_fmt}}; the newest model "
        f"checkpoint was written {_dt.datetime.fromtimestamp(_ckpt):{_fmt}}. "
        f"Rerun `python scripts/05c_eval_horizons.py` before quoting anything "
        f"on this page. The evidence of the mismatch is expanded below."
    )

mc = load_multicrop_summary()
if IS_STALE and not mc.empty and "full" in set(df["horizon"]):
    cmp_df = (df[df["horizon"] == "full"][["model", "crop", "r2", "n"]]
              .merge(mc[["model", "crop", "r2", "n"]], on=["model", "crop"],
                     how="inner", suffixes=("_horizons", "_current")))
    if not cmp_df.empty:
        cmp_df["Δ R²"] = cmp_df["r2_horizons"] - cmp_df["r2_current"]
        with st.expander(
                f"Disagreement with the current per-crop evaluation "
                f"({len(cmp_df)} shared (model, crop) pairs, "
                f"max |Δ R²| = {cmp_df['Δ R²'].abs().max():.3f})"):
            st.caption(
                "Same models, same crops, no in-season cutoff on either side. "
                "`multicrop_summary.csv` is the current evaluation; the "
                "`n` columns differ too, so the two files do not even score "
                "the same test population."
            )
            st.dataframe(
                cmp_df.sort_values("Δ R²", key=abs, ascending=False)
                      .rename(columns={"r2_horizons": "R² (horizons file)",
                                       "r2_current": "R² (current eval)",
                                       "n_horizons": "n (horizons file)",
                                       "n_current": "n (current eval)"})
                      .style.format({"R² (horizons file)": "{:.3f}",
                                     "R² (current eval)": "{:.3f}",
                                     "Δ R²": "{:+.3f}"}),
                width="stretch", hide_index=True,
            )

# --------------------------------------------------------------------------- #
# Coverage — drop models with no in-season rows instead of rendering an empty
# column that reads as a measured zero.
# --------------------------------------------------------------------------- #
in_season = df[df["horizon"] != "full"]
covered = [m for m in MODEL_ORDER if m in set(in_season["model"])]
missing = [m for m in MODEL_ORDER
           if m in set(df["model"]) and m not in covered]
absent = [m for m in MODEL_ORDER if m not in set(df["model"])]

if missing or absent:
    gaps = []
    if missing:
        gaps.append(f"no in-season rows: {', '.join(missing)}")
    if absent:
        gaps.append(f"absent from the file: {', '.join(absent)}")
    st.warning(
        "Not every architecture is evaluated at in-season horizons — "
        + "; ".join(gaps)
        + ". Those models are excluded below rather than drawn as blanks."
    )

df = df[df["model"].isin(covered)]
if df.empty:
    st.warning("No model in this file has in-season horizon rows.")
    st.stop()

crops = sorted(df["crop"].unique().tolist())

# --------------------------------------------------------------------------- #
# Headline
# --------------------------------------------------------------------------- #
if clim_by_crop:
    divider("Skill floor")
    st.caption(
        "The climatology reference uses no in-season observation at all, so it "
        "is flat across every horizon. A curve below it means the extra "
        "in-season data cost accuracy."
    )
    tightest = int(in_season["horizon_days"].min())
    at_tight = in_season[in_season["horizon_days"] == tightest].copy()
    at_tight["skill"] = at_tight["r2"] - at_tight["crop"].map(clim_by_crop)
    per_model = at_tight.groupby("model")["skill"].mean()
    k1, k2, k3 = st.columns(3)
    with k1:
        kpi("Climatology R² (null)", f"{clim['r2'].mean():.3f}")
    with k2:
        kpi(f"Best mean skill at {tightest}d",
            f"{per_model.max():+.3f}" if len(per_model) else "—",
            per_model.idxmax() if len(per_model) else None)
    with k3:
        kpi(f"Models above null at {tightest}d",
            f"{int((per_model > 0).sum())} of {len(per_model)}")

# --------------------------------------------------------------------------- #
# Per-crop horizon curves
# --------------------------------------------------------------------------- #
divider("R² vs days before harvest")
left, right = st.columns([1, 3])

with left:
    sel_crop = st.selectbox(
        "Crop", crops,
        index=crops.index("winter_wheat") if "winter_wheat" in crops else 0,
        key="curve_crop",
    )
    show_full = st.checkbox(
        "Include the no-cutoff endpoint",
        value=False,
        help="Adds the case where the model sees the entire growing season — "
             "the easiest case, and an upper bound rather than a forecast.",
    )

sub = df[df["crop"] == sel_crop].copy()
if not show_full:
    sub = sub[sub["horizon"] != "full"]

with right:
    if sub.empty:
        st.info("No horizon rows for this crop.")
    else:
        sub["x"] = sub["horizon_days"].fillna(0).astype(int)
        fig = go.Figure()
        for m in [m for m in MODEL_ORDER if m in set(sub["model"])]:
            g = sub[sub["model"] == m].sort_values("x", ascending=False)
            is_tf = m in TRANSFORMERS
            fig.add_scatter(
                x=g["x"], y=g["r2"], name=f"{m} ({'T' if is_tf else 'B'})",
                mode="lines+markers",
                line=dict(color=MODEL_COLOR[m], width=2,
                          dash="solid" if is_tf else "dash"),
                marker=dict(size=9, symbol="circle" if is_tf else "square"),
                hovertemplate=(f"{m}<br>%{{x}} days pre-harvest"
                               "<br>R² %{y:.3f}<extra></extra>"),
            )
        if sel_crop in clim_by_crop:
            fig.add_hline(
                y=clim_by_crop[sel_crop], line_dash="dot",
                line_color=REFERENCE_INK, line_width=2,
                annotation_text=(f"climatology (null) R² = "
                                 f"{clim_by_crop[sel_crop]:.3f}"),
                annotation_position="bottom right",
            )
        fig.update_xaxes(autorange="reversed", title="Days before harvest")
        fig.update_yaxes(title="R²")
        fig.update_layout(height=460, hovermode="x unified",
                          title=f"{sel_crop} — solid = transformer, "
                                f"dashed = baseline",
                          legend_title_text="Model")
        st.plotly_chart(fig, width="stretch")

# Table view beneath the chart: the numbers behind it, and the accessible
# fallback for the low-contrast categorical slots.
if not sub.empty:
    piv = (sub.pivot(index="model", columns="horizon", values="r2")
              .reindex(index=[m for m in MODEL_ORDER if m in set(sub["model"])]))
    cols = sorted([c for c in piv.columns if c.startswith("h-")],
                  key=lambda c: int(c[2:-1]))
    if "full" in piv.columns:
        cols.append("full")
    piv = piv[cols]
    piv.insert(0, "class", [model_class(m) for m in piv.index])
    if sel_crop in clim_by_crop:
        piv.insert(1, "climatology (null)", clim_by_crop[sel_crop])
    st.dataframe(
        piv.style.format("{:.3f}", na_rep="—",
                         subset=[c for c in piv.columns if c != "class"]),
        width="stretch",
    )

# --------------------------------------------------------------------------- #
# Crop × model at one horizon
# --------------------------------------------------------------------------- #
divider("Crop × model at one horizon")
fc1, fc2 = st.columns([1, 1])
horizon_choices = (sorted([h for h in df["horizon"].unique() if h != "full"],
                          key=lambda c: int(c[2:-1]))
                   + (["full"] if "full" in set(df["horizon"]) else []))
with fc1:
    sel_horizon = st.selectbox("Horizon", horizon_choices, index=0,
                               key="matrix_horizon")
with fc2:
    _choices = (["skill vs null"] if clim_by_crop else []) + ["r2", "rmse", "mae"]
    metric = st.selectbox("Cell value", _choices, index=0, key="matrix_metric")

mh = df[df["horizon"] == sel_horizon].copy()
if mh.empty:
    st.info("No rows at this horizon.")
else:
    mh["skill vs null"] = mh["r2"] - mh["crop"].map(clim_by_crop)
    order = [m for m in MODEL_ORDER if m in set(mh["model"])]
    mat = (mh.pivot(index="crop", columns="model", values=metric)
             .reindex(columns=order))
    higher_is_better = metric in ("r2", "skill vs null")
    mat = mat.assign(__k__=mat.mean(axis=1)).sort_values(
        "__k__", ascending=not higher_is_better).drop(columns="__k__")

    if metric == "skill vs null" and clim_by_crop:
        lim = float(np.nanmax(np.abs(mat.to_numpy()))) or 1.0
        styled = (mat.style.format("{:+.3f}", na_rep="—")
                     .background_gradient(cmap="RdBu", vmin=-lim, vmax=lim))
    elif metric == "r2":
        styled = (mat.style.format("{:.3f}", na_rep="—")
                     .background_gradient(cmap="RdBu", vmin=-0.7, vmax=0.7))
    else:
        styled = (mat.style.format("{:.2f}", na_rep="—")
                     .background_gradient(cmap="RdBu_r"))
    st.dataframe(styled, width="stretch")
    st.caption(
        f"Transformers: {', '.join(m for m in order if m in TRANSFORMERS) or '—'}. "
        f"Baselines: {', '.join(m for m in order if m in BASELINES) or '—'}."
        + ("" if metric != "skill vs null" else
           " Positive = beats the climatology reference for that crop.")
    )

    # Winner per crop, with the margin that makes the win meaningful or not.
    st.markdown("##### Leading model per crop at this horizon")
    best_rows = []
    for crop, g in mh.groupby("crop"):
        g = g.sort_values("r2", ascending=False).reset_index(drop=True)
        margin = (float(g.at[0, "r2"] - g.at[1, "r2"]) if len(g) > 1
                  else float("nan"))
        best_rows.append({
            "Crop": crop, "Model": g.at[0, "model"],
            "Class": model_class(g.at[0, "model"]),
            "R²": float(g.at[0, "r2"]),
            "Climatology R²": clim_by_crop.get(crop, float("nan")),
            "Skill vs null": float(g.at[0, "skill vs null"]),
            "Margin over next": margin,
            "Runner-up": g.at[1, "model"] if len(g) > 1 else "—",
            "n": int(g.at[0, "n"]),
        })
    best = pd.DataFrame(best_rows)
    best = best.sort_values("Skill vs null" if clim_by_crop else "R²",
                            ascending=False)
    if not clim_by_crop:
        best = best.drop(columns=["Climatology R²", "Skill vs null"])

    def _tie_flag(v):
        if not isinstance(v, (int, float)) or v != v:
            return ""
        return "background-color: rgba(227,73,72,0.18)" if v < 0.01 else ""

    def _skill_ink(v):
        if not isinstance(v, (int, float)) or v != v:
            return ""
        return "color: #2a78d6" if v > 0 else "color: #e34948"

    styler = (best.style
                  .format({"R²": "{:.3f}", "Climatology R²": "{:.3f}",
                           "Skill vs null": "{:+.3f}",
                           "Margin over next": "{:.3f}"}, na_rep="—")
                  .map(_tie_flag, subset=["Margin over next"]))
    if clim_by_crop:
        styler = styler.map(_skill_ink, subset=["Skill vs null"])
    st.dataframe(styler, width="stretch", hide_index=True)

    n_ties = int((best["Margin over next"] < 0.01).sum())
    note = (f"Highlighted margins are below 0.01 R² ({n_ties} of {len(best)} "
            f"crops) — inside run-to-run noise, so the leader is a tie rather "
            f"than a winner.")
    if clim_by_crop:
        note += (f" {int((best['Skill vs null'] > 0).sum())} of {len(best)} "
                 f"leaders beat the climatology reference.")
    st.caption(note)

st.markdown("---")
st.caption(
    "Sources: `results/metrics/multicrop_horizons.csv` "
    "(`python scripts/05c_eval_horizons.py`) and "
    "`results/metrics/climatology_baseline.csv`."
)

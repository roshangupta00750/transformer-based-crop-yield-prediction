"""Head-to-head leaderboard — Chapter 4 landing page.

Three transformers (iTransformer, PatchTST, TFT) against three baselines
(LSTM, XGBoost, DLinear), all scored against a learn-nothing climatology
reference on the 2024-2025 held-out split.

Every number rendered here is read from, or computed from, a file on disk:

    results/metrics/multicrop_summary.csv      per (model, crop) R²/RMSE/MAE
    results/metrics/climatology_baseline.csv   the null reference (see below)
    configs/taxonomy.yaml                      the a-priori architecture picks
"""
from __future__ import annotations

import datetime as _dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import (CHECKPOINTS_DIR, METRICS_DIR, MODEL_INFO, MODELS, PROCESSED,
                   PROJECT_ROOT, divider, kpi, load_multicrop_summary,
                   load_taxonomy, page_header, taxonomy_verdict)

st.set_page_config(page_title="Leaderboard — Crop Yield Benchmark",
                   layout="wide")

# --------------------------------------------------------------------------- #
# Architecture classes — the paper's actual subject is the transformer column.
# --------------------------------------------------------------------------- #
TRANSFORMERS = ("itransformer", "patchtst", "tft")
BASELINES = ("lstm", "xgboost", "dlinear")
MODEL_ORDER = list(TRANSFORMERS) + list(BASELINES)


def model_class(model: str) -> str:
    return "Transformer" if model in TRANSFORMERS else "Baseline"


def ordered(models) -> list[str]:
    """Known models in transformer-then-baseline order, then anything new.

    Never drops a model the metrics file contains — an unrecognised name is
    appended rather than silently omitted.
    """
    seen = set(models)
    return ([m for m in MODEL_ORDER if m in seen]
            + sorted(m for m in seen if m not in MODEL_ORDER))


# Categorical slots 1-2 (validated adjacent-pair CVD separation, light surface).
CLASS_COLOR = {"Transformer": "#2a78d6", "Baseline": "#eb6834"}
REFERENCE_INK = "#52514e"
# Diverging blue (skill) <-> red (no skill), neutral gray midpoint.
SKILL_SCALE = [(0.0, "#e34948"), (0.5, "#f0efec"), (1.0, "#2a78d6")]

CLIMATOLOGY_CSV = METRICS_DIR / "climatology_baseline.csv"


# --------------------------------------------------------------------------- #
# Climatology (null) reference
# --------------------------------------------------------------------------- #
@st.cache_data(ttl=300, show_spinner="Computing climatology reference…")
def load_climatology() -> pd.DataFrame:
    """Per-crop skill floor: predict every test sample with its district's
    train-fitted baseline yield, i.e. anomaly = 0 — a model that learns nothing.

    Scored on exactly the (district, crop, year) population the benchmark
    scores the models on, so it is directly comparable to
    ``multicrop_summary.csv``. Result is cached to
    ``results/metrics/climatology_baseline.csv`` and recomputed whenever the
    processed dataset or a fitted pipeline is newer than that cache.
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
    except Exception:  # noqa: BLE001 — page must still render without the src tree
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
    """mtime of the most recently written model checkpoint, or 0.0."""
    if not CHECKPOINTS_DIR.exists():
        return 0.0
    times = [p.stat().st_mtime for p in CHECKPOINTS_DIR.glob("*.pt")]
    return max(times) if times else 0.0


def staleness_note(path: Path) -> str | None:
    """Return a warning string when ``path`` predates the newest checkpoint."""
    if not path.exists():
        return None
    ckpt = newest_checkpoint_mtime()
    if ckpt <= path.stat().st_mtime:
        return None
    fmt = "%Y-%m-%d %H:%M"
    return (f"`{path.name}` was written "
            f"{_dt.datetime.fromtimestamp(path.stat().st_mtime):{fmt}} but a "
            f"checkpoint was written "
            f"{_dt.datetime.fromtimestamp(ckpt):{fmt}}. The numbers below "
            f"describe superseded weights.")


page_header(
    "Head-to-head Leaderboard",
    "Three transformers against three baselines and a learn-nothing "
    "climatology reference, on the 2024-2025 held-out split.",
)

mc = load_multicrop_summary()
clim = load_climatology()
tax = load_taxonomy()

if mc.empty:
    st.warning(
        "No current per-crop results. Either the policy gate is closed "
        "(results predate the active feature policy) or "
        "`results/metrics/multicrop_summary.csv` is missing. Regenerate with "
        "`python scripts/05b_eval_multicrop.py`."
    )
    st.stop()

_stale = staleness_note(METRICS_DIR / "multicrop_summary.csv")
if _stale:
    st.warning(_stale)

if clim.empty:
    st.error(
        "Climatology reference unavailable — `data/processed/dataset.parquet` "
        "or the fitted pipelines could not be read. Model R² below is shown "
        "without a skill floor and must not be read as evidence of skill."
    )

clim_by_crop = dict(zip(clim["crop"], clim["r2"])) if not clim.empty else {}
clim_mean = float(clim["r2"].mean()) if not clim.empty else float("nan")

# Per-(model, crop) skill = R² minus the null reference for that crop.
mc = mc.copy()
mc["class"] = mc["model"].map(model_class)
mc["clim_r2"] = mc["crop"].map(clim_by_crop)
mc["skill"] = mc["r2"] - mc["clim_r2"]

# --------------------------------------------------------------------------- #
# Headline
# --------------------------------------------------------------------------- #
divider("Skill floor")
st.caption(
    "R² is measured in absolute dt/ha while the models predict a district "
    "anomaly, so every model is handed its district's long-run mean for free. "
    "The climatology reference is that free lunch alone — the district "
    "baseline carried forward on its train-only trend, with the weather-driven "
    "anomaly set to zero. A model only demonstrates skill above this line."
)

by_model = (mc.groupby("model")
              .agg(mean_r2=("r2", "mean"), median_r2=("r2", "median"),
                   min_r2=("r2", "min"), max_r2=("r2", "max"),
                   mean_skill=("skill", "mean"),
                   crops_beating=("skill", lambda s: int((s > 0).sum())),
                   n_crops=("crop", "nunique"))
              .reindex(ordered(mc["model"])))
by_model["class"] = [model_class(m) for m in by_model.index]

k1, k2, k3, k4 = st.columns(4)
with k1:
    kpi("Climatology R² (null)",
        f"{clim_mean:.3f}" if clim_mean == clim_mean else "—")
best_t = by_model[by_model["class"] == "Transformer"]["mean_r2"]
best_b = by_model[by_model["class"] == "Baseline"]["mean_r2"]
with k2:
    kpi("Best transformer",
        f"{best_t.max():.3f}" if len(best_t) else "—",
        f"{best_t.max() - clim_mean:+.3f} vs null"
        if len(best_t) and clim_mean == clim_mean else None)
with k3:
    kpi("Best baseline",
        f"{best_b.max():.3f}" if len(best_b) else "—",
        f"{best_b.max() - clim_mean:+.3f} vs null"
        if len(best_b) and clim_mean == clim_mean else None)
with k4:
    n_above = int((by_model["mean_r2"] > clim_mean).sum()) \
        if clim_mean == clim_mean else 0
    kpi("Models above the null", f"{n_above} of {len(by_model)}")

# --------------------------------------------------------------------------- #
# Leaderboard, grouped by architecture class
# --------------------------------------------------------------------------- #
divider("Mean R² per model, by architecture class")

rows = []
if clim_mean == clim_mean:
    rows.append({
        "Model": "climatology (null)", "Class": "Reference",
        "Mean R²": clim_mean, "Median R²": float(clim["r2"].median()),
        "Min R²": float(clim["r2"].min()), "Max R²": float(clim["r2"].max()),
        "Mean skill vs null": 0.0, "Crops above null": "—",
        "Crops": int(len(clim)), "Reference": "district mean + train-only trend",
    })
for cls in ("Transformer", "Baseline"):
    block = (by_model[by_model["class"] == cls]
             .sort_values("mean_r2", ascending=False))
    for m, r in block.iterrows():
        rows.append({
            "Model": m, "Class": cls,
            "Mean R²": r["mean_r2"], "Median R²": r["median_r2"],
            "Min R²": r["min_r2"], "Max R²": r["max_r2"],
            "Mean skill vs null": r["mean_skill"],
            "Crops above null": f"{int(r['crops_beating'])}/{int(r['n_crops'])}",
            "Crops": int(r["n_crops"]),
            "Reference": MODEL_INFO.get(m, {}).get("ref", "—"),
        })
board = pd.DataFrame(rows)


def _skill_ink(v):
    if not isinstance(v, (int, float)) or v != v:
        return ""
    return "color: #2a78d6" if v > 0 else "color: #e34948"


st.dataframe(
    board.style
         .format({"Mean R²": "{:.3f}", "Median R²": "{:.3f}",
                  "Min R²": "{:.3f}", "Max R²": "{:.3f}",
                  "Mean skill vs null": "{:+.3f}"}, na_rep="—")
         .map(_skill_ink, subset=["Mean skill vs null"]),
    width="stretch", hide_index=True,
)

fig = go.Figure()
bar_order: list[str] = []
for cls in ("Transformer", "Baseline"):
    block = (by_model[by_model["class"] == cls]
             .sort_values("mean_r2", ascending=False))
    bar_order += block.index.tolist()
    fig.add_bar(
        x=block.index.tolist(), y=block["mean_r2"].tolist(), name=cls,
        marker_color=CLASS_COLOR[cls],
        text=[f"{v:.3f}" for v in block["mean_r2"]], textposition="outside",
        hovertemplate="%{x}<br>mean R² %{y:.3f}<extra></extra>",
    )
if clim_mean == clim_mean:
    fig.add_hline(
        y=clim_mean, line_dash="dash", line_color=REFERENCE_INK,
        annotation_text=f"climatology (null) R² = {clim_mean:.3f}",
        annotation_position="top left",
    )
# Each model belongs to exactly one class, so overlay keeps every bar centred
# on its own category instead of offsetting it into a half-slot.
fig.update_layout(
    height=420, yaxis_title="Mean R² across crops", xaxis_title=None,
    legend_title_text="Architecture class", barmode="overlay", bargap=0.35,
    xaxis=dict(categoryorder="array", categoryarray=bar_order),
)
st.plotly_chart(fig, width="stretch")

# --------------------------------------------------------------------------- #
# Per-crop matrix, coloured by skill against the null
# --------------------------------------------------------------------------- #
divider("Per-crop R², coloured by skill against the null")
st.caption(
    "Blue = beats the climatology reference for that crop; red = worse than "
    "predicting the district baseline. [P] marks the taxonomy's primary "
    "architecture pick for that crop, [A] an alternate."
)

present = ordered(mc["model"])
crop_order = (mc.groupby("crop")["r2"].mean()
                .sort_values(ascending=False).index.tolist())
r2_pivot = (mc.pivot(index="crop", columns="model", values="r2")
              .reindex(index=crop_order, columns=present))
skill_pivot = (mc.pivot(index="crop", columns="model", values="skill")
                 .reindex(index=crop_order, columns=present))

disp = pd.DataFrame(index=r2_pivot.index, columns=present, dtype=object)
for crop in r2_pivot.index:
    for m in present:
        v = r2_pivot.at[crop, m]
        if pd.isna(v):
            disp.at[crop, m] = "—"
            continue
        verdict = taxonomy_verdict(crop, m, tax)
        tag = " [P]" if verdict == "primary" else (
            " [A]" if verdict == "alternate" else "")
        disp.at[crop, m] = f"{v:.3f}{tag}"
if clim_by_crop:
    disp.insert(0, "climatology (null)",
                [f"{clim_by_crop.get(c, float('nan')):.3f}"
                 if c in clim_by_crop else "—" for c in disp.index])


def _skill_bg(frame: pd.DataFrame) -> pd.DataFrame:
    css = pd.DataFrame("", index=frame.index, columns=frame.columns)
    for m in present:
        if m not in skill_pivot.columns:
            continue
        for crop in frame.index:
            s = skill_pivot.at[crop, m]
            if pd.isna(s):
                continue
            # Saturate at +-0.4 skill; blue above the null, red below.
            t = float(np.clip(s / 0.4, -1, 1))
            if t >= 0:
                bg = f"rgba(42,120,214,{0.10 + 0.55 * t:.2f})"
            else:
                bg = f"rgba(227,73,72,{0.10 + 0.55 * -t:.2f})"
            css.at[crop, m] = f"background-color: {bg}"
    if "climatology (null)" in frame.columns:
        css["climatology (null)"] = "background-color: rgba(82,81,78,0.10)"
    return css


st.dataframe(disp.style.apply(_skill_bg, axis=None), width="stretch")

st.caption(
    "Column order: transformers "
    f"({', '.join(m for m in present if m in TRANSFORMERS)}), then baselines "
    f"({', '.join(m for m in present if m in BASELINES)})."
)

# --------------------------------------------------------------------------- #
# Family compatibility
# --------------------------------------------------------------------------- #
if tax and clim_by_crop:
    divider("Crop phenology family × model architecture family")
    st.caption("Cell value is mean skill against the null reference, not raw R².")
    crop_fam = {c: v.get("phenology_family", "unknown")
                for c, v in tax.get("crops", {}).items()}
    model_fam = {m: v.get("architecture_family", "unknown")
                 for m, v in tax.get("models", {}).items()}
    mc2 = mc.copy()
    mc2["crop_family"] = mc2["crop"].map(crop_fam)
    mc2["model_family"] = mc2["model"].map(model_fam)
    fam_pivot = (mc2.dropna(subset=["skill"])
                    .groupby(["crop_family", "model_family"])["skill"]
                    .mean().unstack())
    lim = float(np.nanmax(np.abs(fam_pivot.to_numpy()))) or 1.0
    fig = px.imshow(
        fam_pivot, color_continuous_scale=SKILL_SCALE,
        zmin=-lim, zmax=lim, aspect="auto", text_auto=".3f",
        labels=dict(x="Model architecture family",
                    y="Crop phenology family", color="Mean skill vs null"),
    )
    fig.update_layout(height=420)
    st.plotly_chart(fig, width="stretch")
    st.dataframe(fam_pivot.style.format("{:+.3f}", na_rep="—"), width="stretch")

# --------------------------------------------------------------------------- #
# Single-crop drilldown
# --------------------------------------------------------------------------- #
divider("Single-crop ranking")
sel_crop = st.selectbox(
    "Crop", crop_order,
    index=crop_order.index("winter_wheat") if "winter_wheat" in crop_order else 0,
)

cw = (mc[mc["crop"] == sel_crop]
      .sort_values("r2", ascending=False)
      .reset_index(drop=True))
cw["Rank"] = cw.index + 1
# Margin over the next model down — a "win" smaller than this is not a result.
cw["margin"] = cw["r2"] - cw["r2"].shift(-1)
cw["Class"] = cw["model"].map(model_class)
cw["Taxonomy pick"] = cw["model"].map(
    lambda m: {"primary": "PRIMARY", "alternate": "alternate", "": "—"}[
        taxonomy_verdict(sel_crop, m, tax)])

if len(cw) >= 2:
    top_margin = float(cw.at[0, "margin"])
    top_skill = float(cw.at[0, "skill"]) if cw["skill"].notna().all() else float("nan")
    msg = (f"Winner **{cw.at[0, 'model']}** leads the runner-up "
           f"**{cw.at[1, 'model']}** by **{top_margin:.3f}** R².")
    if top_skill == top_skill:
        msg += (f" Against the climatology reference "
                f"({clim_by_crop[sel_crop]:.3f}) it is **{top_skill:+.3f}**.")
    if top_margin < 0.01:
        st.warning(msg + " A margin under 0.01 R² is inside run-to-run noise — "
                          "treat this crop as a tie, not a win.")
    elif top_skill == top_skill and top_skill <= 0:
        st.warning(msg + " No model on this crop beats the null reference.")
    else:
        st.info(msg)

show = cw[["Rank", "model", "Class", "Taxonomy pick", "r2", "skill", "margin",
           "rmse", "mae", "n"]].rename(columns={
    "model": "Model", "r2": "R²", "skill": "Skill vs null",
    "margin": "Margin over next", "rmse": "RMSE (dt/ha)", "mae": "MAE",
    "n": "n"})
st.dataframe(
    show.style.format({"R²": "{:.3f}", "Skill vs null": "{:+.3f}",
                       "Margin over next": "{:.3f}", "RMSE (dt/ha)": "{:.2f}",
                       "MAE": "{:.2f}"}, na_rep="—")
              .map(_skill_ink, subset=["Skill vs null"]),
    width="stretch", hide_index=True,
)

st.markdown("---")
st.caption(
    "Sources: `results/metrics/multicrop_summary.csv` "
    "(`python scripts/05b_eval_multicrop.py`) and "
    "`results/metrics/climatology_baseline.csv` (recomputed on this page "
    "whenever the processed dataset or a fitted pipeline is newer)."
)

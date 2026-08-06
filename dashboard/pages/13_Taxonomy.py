"""Crop-Model Compatibility Taxonomy.

``configs/taxonomy.yaml`` is an a-priori hypothesis: for each crop, which
architecture should win, given crop biology and model inductive bias. This
page tests it against the current benchmark.

The test is recomputed here, on every render, from
``configs/taxonomy.yaml`` + ``results/metrics/multicrop_summary.csv`` using
the same rule as ``scripts/14_taxonomy_validate.py``. The shipped
``results/metrics/taxonomy_validation.csv`` is treated as a build artefact:
it is compared against the live recomputation and flagged when it disagrees,
never rendered as the current answer.
"""
from __future__ import annotations

import datetime as _dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import (CHECKPOINTS_DIR, METRICS_DIR, MODELS, PROCESSED,
                   PROJECT_ROOT, divider, kpi, load_multicrop_summary,
                   load_taxonomy_validation, page_header)

st.set_page_config(page_title="Taxonomy — Crop Yield Benchmark",
                   layout="wide")

TRANSFORMERS = ("itransformer", "patchtst", "tft")
BASELINES = ("lstm", "xgboost", "dlinear")
# A margin under this is inside run-to-run variance for this benchmark, so a
# "winner" separated by less than it is a tie.
NOISE_R2 = 0.01

CLIMATOLOGY_CSV = METRICS_DIR / "climatology_baseline.csv"


def model_class(model: str) -> str:
    return "Transformer" if model in TRANSFORMERS else "Baseline"


# --------------------------------------------------------------------------- #
# Climatology (null) reference
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


def predicted_order(crop: str, tax: dict) -> list[str]:
    """Primary, then alternates in listed order, then the remaining models.

    Mirrors ``scripts/14_taxonomy_validate.py::rank_models_from_taxonomy``.
    """
    pred = tax["predicted_winners"].get(crop, {}) or {}
    ordered = []
    if pred.get("primary"):
        ordered.append(pred["primary"])
    ordered += [m for m in (pred.get("alternates") or []) if m not in ordered]
    ordered += [m for m in tax["models"] if m not in ordered]
    return ordered


page_header(
    "Crop-Model Compatibility Taxonomy",
    "An a-priori hypothesis about which architecture should win each crop, "
    "tested against the current benchmark.",
)

tax_path = PROJECT_ROOT / "configs" / "taxonomy.yaml"
if not tax_path.exists():
    st.warning("`configs/taxonomy.yaml` not found.")
    st.stop()
tax = yaml.safe_load(tax_path.read_text())

scores = load_multicrop_summary()
if scores.empty:
    st.warning(
        "No current per-crop results to test the taxonomy against. Either the "
        "policy gate is closed or `results/metrics/multicrop_summary.csv` is "
        "missing. Regenerate with `python scripts/05b_eval_multicrop.py`."
    )
    st.stop()

clim = load_climatology()
clim_by_crop = dict(zip(clim["crop"], clim["r2"])) if not clim.empty else {}

# --------------------------------------------------------------------------- #
# Recompute the test from the current metrics
# --------------------------------------------------------------------------- #
all_models = [m for m in tax["models"]]
rows, rank_rows = [], []
for crop, pred in tax["predicted_winners"].items():
    primary = (pred or {}).get("primary", "")
    alts = list((pred or {}).get("alternates") or [])
    top3 = ([primary] + alts)[:3]

    cs = (scores[scores["crop"] == crop]
          .sort_values("r2", ascending=False).reset_index(drop=True))
    if cs.empty:
        continue
    observed = cs.at[0, "model"]
    observed_r2 = float(cs.at[0, "r2"])
    margin = float(observed_r2 - cs.at[1, "r2"]) if len(cs) > 1 else float("nan")
    prim_row = cs[cs["model"] == primary]
    primary_r2 = float(prim_row["r2"].iloc[0]) if not prim_row.empty else float("nan")
    clim_r2 = clim_by_crop.get(crop, float("nan"))

    rows.append({
        "crop": crop,
        "predicted_primary": primary,
        "predicted_alternates": ", ".join(alts) or "—",
        "observed_best_model": observed,
        "observed_best_r2": observed_r2,
        "margin_over_runner_up": margin,
        "runner_up": cs.at[1, "model"] if len(cs) > 1 else "—",
        "predicted_primary_r2": primary_r2,
        "primary_gap": observed_r2 - primary_r2,
        "climatology_r2": clim_r2,
        "winner_skill_vs_null": observed_r2 - clim_r2,
        "hit_at_1": int(observed == primary),
        "hit_at_3": int(observed in top3),
        "rationale": (pred or {}).get("rationale", ""),
    })

    pmap = {m: i + 1 for i, m in enumerate(predicted_order(crop, tax))}
    omap = {m: i + 1 for i, m in enumerate(cs["model"])}
    for m in all_models:
        if m not in omap:
            continue
        rank_rows.append({"crop": crop, "model": m,
                          "predicted_rank": pmap.get(m, len(all_models)),
                          "observed_rank": omap[m]})

live = pd.DataFrame(rows)
ranks = pd.DataFrame(rank_rows)
if live.empty:
    st.warning("No crop in `multicrop_summary.csv` overlaps the taxonomy.")
    st.stop()

n = len(live)
hit1 = int(live["hit_at_1"].sum())
hit3 = int(live["hit_at_3"].sum())

per_crop_rho = {}
if not ranks.empty:
    from scipy.stats import spearmanr
    for crop, g in ranks.groupby("crop"):
        if g["predicted_rank"].nunique() > 1 and g["observed_rank"].nunique() > 1:
            per_crop_rho[crop] = float(
                spearmanr(g["predicted_rank"], g["observed_rank"]).statistic)
mean_rho = float(np.mean(list(per_crop_rho.values()))) if per_crop_rho else float("nan")

# Reference strategy: ignore the taxonomy and always name the same model.
win_counts = (live["observed_best_model"].value_counts()
              if not live.empty else pd.Series(dtype=int))
const_model = win_counts.idxmax() if len(win_counts) else "—"
const_hits = int(win_counts.max()) if len(win_counts) else 0

# --------------------------------------------------------------------------- #
# Provenance: is the shipped validation CSV still describing this benchmark?
# --------------------------------------------------------------------------- #
shipped = load_taxonomy_validation()
val_path = METRICS_DIR / "taxonomy_validation.csv"
summary_path = METRICS_DIR / "multicrop_summary.csv"
_fmt = "%Y-%m-%d %H:%M"

if not shipped.empty and "hit_at_1" in shipped.columns:
    s_hit1, s_n = int(shipped["hit_at_1"].sum()), len(shipped)
    s_hit3 = int(shipped["hit_at_3"].sum())
    older = (val_path.exists() and summary_path.exists()
             and val_path.stat().st_mtime < summary_path.stat().st_mtime)
    if (s_hit1, s_hit3, s_n) != (hit1, hit3, n) or older:
        st.error(
            f"**`taxonomy_validation.csv` is superseded — the numbers on this "
            f"page are recomputed live, not read from it.** The shipped file "
            f"(written {_dt.datetime.fromtimestamp(val_path.stat().st_mtime):{_fmt}}) "
            f"reports top-1 {s_hit1}/{s_n} and top-3 {s_hit3}/{s_n}. The same "
            f"rule applied to `multicrop_summary.csv` "
            f"(written {_dt.datetime.fromtimestamp(summary_path.stat().st_mtime):{_fmt}}) "
            f"gives top-1 {hit1}/{n} and top-3 {hit3}/{n}. "
            f"Rerun `python scripts/14_taxonomy_validate.py` to resync."
        )
elif shipped.empty:
    st.info(
        "`taxonomy_validation.csv` is unavailable or gated out; everything on "
        "this page is recomputed from `multicrop_summary.csv` regardless."
    )

# --------------------------------------------------------------------------- #
# Headline
# --------------------------------------------------------------------------- #
divider("Does the taxonomy predict the winner?")

k1, k2, k3, k4 = st.columns(4)
with k1:
    kpi("Top-1", f"{hit1}/{n}",
        f"{hit1 - const_hits:+d} vs always naming {const_model}")
with k2:
    kpi("Top-3", f"{hit3}/{n}")
with k3:
    kpi("Mean Spearman ρ",
        f"{mean_rho:+.3f}" if mean_rho == mean_rho else "—")
with k4:
    n_noise = int((live["margin_over_runner_up"] < NOISE_R2).sum())
    kpi("Crops whose winner is a tie", f"{n_noise}/{n}",
        f"margin < {NOISE_R2:.2f} R²", )

st.caption(
    f"Reference strategy: ignoring the taxonomy and always naming "
    f"`{const_model}` — the most frequent winner in the current benchmark — "
    f"scores {const_hits}/{n} top-1. The taxonomy has to beat that, not zero. "
    f"No significance test is reported: the six candidate models are not "
    f"exchangeable (their mean R² spans "
    f"{scores.groupby('model')['r2'].mean().min():.3f} to "
    f"{scores.groupby('model')['r2'].mean().max():.3f}), so a uniform-random "
    f"null does not apply."
)

if clim_by_crop:
    n_skill = int((live["winner_skill_vs_null"] > 0).sum())
    if n_skill == n:
        st.info(f"The observed winner beats the climatology reference on all "
                f"{n} crops.")
    else:
        losers = ", ".join(live.loc[live["winner_skill_vs_null"] <= 0, "crop"])
        st.warning(
            f"The observed winner beats the climatology reference on "
            f"{n_skill} of {n} crops. On the other {n - n_skill} "
            f"({losers}) the taxonomy is naming the least-bad of a set in "
            f"which nothing beats predicting the district baseline."
        )

# --------------------------------------------------------------------------- #
# Per-crop
# --------------------------------------------------------------------------- #
divider("Per crop: predicted primary, observed winner, margin")

show = live[["crop", "predicted_primary", "predicted_alternates",
             "observed_best_model", "observed_best_r2", "runner_up",
             "margin_over_runner_up", "climatology_r2",
             "winner_skill_vs_null", "predicted_primary_r2", "primary_gap",
             "hit_at_1", "hit_at_3"]].copy()
show["winner_class"] = show["observed_best_model"].map(model_class)
show = show.rename(columns={
    "crop": "Crop", "predicted_primary": "Predicted primary",
    "predicted_alternates": "Predicted alts",
    "observed_best_model": "Observed winner", "winner_class": "Winner class",
    "observed_best_r2": "Winner R²", "runner_up": "Runner-up",
    "margin_over_runner_up": "Margin", "climatology_r2": "Climatology R²",
    "winner_skill_vs_null": "Winner skill vs null",
    "predicted_primary_r2": "Primary R²", "primary_gap": "Winner − primary",
    "hit_at_1": "Top-1", "hit_at_3": "Top-3",
})
show = show[["Crop", "Predicted primary", "Predicted alts", "Observed winner",
             "Winner class", "Winner R²", "Runner-up", "Margin",
             "Climatology R²", "Winner skill vs null", "Primary R²",
             "Winner − primary", "Top-1", "Top-3"]]


def _hit_bg(v):
    if v == 1:
        return "background-color: rgba(42,120,214,0.20)"
    if v == 0:
        return "background-color: rgba(227,73,72,0.20)"
    return ""


def _margin_bg(v):
    if isinstance(v, (int, float)) and v == v and v < NOISE_R2:
        return "background-color: rgba(227,73,72,0.20)"
    return ""


def _skill_ink(v):
    if not isinstance(v, (int, float)) or v != v:
        return ""
    return "color: #2a78d6" if v > 0 else "color: #e34948"


styler = (show.style
              .format({"Winner R²": "{:.3f}", "Margin": "{:.3f}",
                       "Climatology R²": "{:.3f}",
                       "Winner skill vs null": "{:+.3f}",
                       "Primary R²": "{:.3f}",
                       "Winner − primary": "{:+.3f}"}, na_rep="—")
              .map(_hit_bg, subset=["Top-1", "Top-3"])
              .map(_margin_bg, subset=["Margin"])
              .map(_skill_ink, subset=["Winner skill vs null"]))
st.dataframe(styler, hide_index=True, width="stretch")
st.caption(
    f"A highlighted Margin is under {NOISE_R2:.2f} R², i.e. the winner and "
    f"runner-up are indistinguishable — a top-1 hit or miss on those crops "
    f"carries no information. `Winner − primary` is how much R² the "
    f"taxonomy's pick gave up; where it is near zero the miss is nominal."
)

near_miss = live[(live["hit_at_1"] == 0)
                 & (live["primary_gap"].abs() < NOISE_R2)]
if not near_miss.empty:
    st.caption(
        "Nominal misses (taxonomy primary within "
        f"{NOISE_R2:.2f} R² of the winner): "
        + ", ".join(near_miss["crop"]) + "."
    )

# --------------------------------------------------------------------------- #
# Rank correlation
# --------------------------------------------------------------------------- #
if not ranks.empty:
    divider("Predicted rank minus observed rank")
    ranks = ranks.copy()
    ranks["rank_diff"] = ranks["predicted_rank"] - ranks["observed_rank"]
    order = [m for m in list(TRANSFORMERS) + list(BASELINES)
             if m in set(ranks["model"])]
    pivot = (ranks.pivot(index="crop", columns="model", values="rank_diff")
                  .reindex(columns=order))
    fig = px.imshow(
        pivot, color_continuous_scale=[(0.0, "#e34948"), (0.5, "#f0efec"),
                                       (1.0, "#2a78d6")],
        zmin=-5, zmax=5, aspect="auto", text_auto=True,
        labels=dict(x="Model", y="Crop", color="Predicted − observed rank"),
    )
    fig.update_layout(height=420)
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "0 = the taxonomy placed the model where the benchmark did. Negative "
        "= the taxonomy ranked it better than it performed. Columns: "
        f"transformers ({', '.join(m for m in order if m in TRANSFORMERS)}), "
        f"then baselines ({', '.join(m for m in order if m in BASELINES)})."
    )
    if per_crop_rho:
        rho_df = (pd.DataFrame({"Crop": list(per_crop_rho),
                                "Spearman ρ": list(per_crop_rho.values())})
                    .sort_values("Spearman ρ", ascending=False))
        st.dataframe(rho_df.style.format({"Spearman ρ": "{:+.3f}"}),
                     hide_index=True, width="stretch")

# --------------------------------------------------------------------------- #
# Rationale per crop
# --------------------------------------------------------------------------- #
divider("The hypothesis, crop by crop")

for _, row in live.iterrows():
    verdict = ("top-1" if row["hit_at_1"] else
               ("top-3" if row["hit_at_3"] else "miss"))
    margin = row["margin_over_runner_up"]
    if margin == margin and margin < NOISE_R2:
        verdict += " (tied winner)"
    label = (f"{verdict}  —  {row['crop']}: predicted "
             f"{row['predicted_primary']}, observed "
             f"{row['observed_best_model']} "
             f"(R² {row['observed_best_r2']:.3f}")
    label += f", margin {margin:.3f})" if margin == margin else ")"
    with st.expander(label):
        st.markdown(f"**A-priori rationale:** {row['rationale']}")
        crop_meta = tax["crops"].get(row["crop"], {})
        if crop_meta:
            st.json({k: v for k, v in crop_meta.items() if k != "rationale"})

# --------------------------------------------------------------------------- #
# Reference
# --------------------------------------------------------------------------- #
divider("Reference — the two taxonomies")

col1, col2 = st.columns(2)
with col1:
    st.markdown("##### Crop taxonomy")
    st.dataframe(pd.DataFrame([{"Crop": k, **v} for k, v in tax["crops"].items()]),
                 hide_index=True, width="stretch")
with col2:
    st.markdown("##### Model taxonomy")
    model_df = pd.DataFrame([{"Model": k, **v} for k, v in tax["models"].items()])
    model_df.insert(1, "Class", model_df["Model"].map(model_class))
    st.dataframe(model_df, hide_index=True, width="stretch")

st.markdown("---")
st.caption(
    "`configs/taxonomy.yaml` holds the hypothesis; it is committed before a "
    "benchmark run and edited only to change the hypothesis. Everything above "
    "is recomputed from it plus `results/metrics/multicrop_summary.csv` and "
    "`results/metrics/climatology_baseline.csv`. "
    "`python scripts/14_taxonomy_validate.py` writes the same result to "
    "`results/metrics/taxonomy_validation.csv` for downstream use."
)

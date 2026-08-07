"""Cross-model fusion — does combining models beat the best single model?

One question, one table. The default view is a KPI row and the method
leaderboard (mean R² per fusion method, 5-model vs 6-model pool). The
per-crop showdown, the (crop x method) heatmap, the per-crop lift table and
the meta-learner coefficient maps are all still here, collapsed.

The climatology baseline sits in the KPI row on purpose: fusion mean R² and
the learn-nothing baseline are computed over the same ten crops, so a fusion
method "winning" at 0.47 has to be read against the 0.45 that predicting the
district baseline already scores.

Reads two paired output sets:
    fusion_summary_5models.csv  — TFT excluded from the pool
    fusion_summary_6models.csv  — TFT included

The 5-vs-6 comparison is the methodological question: when a specialist
model with uneven per-crop quality enters the fusion pool, which fusion
method degrades gracefully and which degrades pathologically?

The set of fusion methods is read from the CSV, not hardcoded here — the
generating script has added methods over time. `best_single_by_val`
(val-best single model per crop) is treated as the baseline, everything
else as a fusion method.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import (CHECKPOINTS_DIR, METRICS_DIR, kpi,
                   load_climatology_baseline, load_multicrop_summary,
                   load_taxonomy, page_header)


def _stale_banner(*metric_paths: Path) -> None:
    """Warn when a metric file predates the newest checkpoint on disk.

    Anything older than the newest checkpoint was, by construction, not
    produced by the models currently in results/checkpoints/. A firing
    warning stays at the top of the page, never inside an expander.

    Pass only the files this page actually read. Passing every candidate path
    put `fusion_summary.csv` — the legacy single-file output, which is not
    loaded at all once the 5/6 pair exists — into the banner as its oldest and
    loudest entry, so the one element that may never be collapsed was
    reporting rot in an artefact that contributes nothing to the page.
    """
    ckpts = [*CHECKPOINTS_DIR.glob("*.pt"), *CHECKPOINTS_DIR.glob("*.ubj")]
    if not ckpts:
        return
    newest = max(ckpts, key=lambda p: p.stat().st_mtime)
    newest_mt = newest.stat().st_mtime
    stale = [(p, p.stat().st_mtime) for p in metric_paths
             if p.exists() and p.stat().st_mtime < newest_mt]
    if not stale:
        return
    lines = "\n".join(
        f"- `{p.name}` — {datetime.fromtimestamp(mt):%Y-%m-%d %H:%M}, "
        f"{(newest_mt - mt) / 86400:.1f} days older"
        for p, mt in stale)
    st.warning(
        "**Stale — these numbers are not from the models currently on "
        f"disk.** Newest checkpoint `{newest.name}` is "
        f"{datetime.fromtimestamp(newest_mt):%Y-%m-%d %H:%M}:\n\n" + lines
    )


st.set_page_config(page_title="Fusion — Crop Yield Benchmark", layout="wide")
page_header("Cross-model fusion (5 vs 6 models)",
            "Does combining the models beat the best single model — and what "
            "happens when a specialist with uneven per-crop quality joins "
            "the pool?")

# Prefer new dual output; fall back to legacy single file for backward compat.
sum_5 = METRICS_DIR / "fusion_summary_5models.csv"
sum_6 = METRICS_DIR / "fusion_summary_6models.csv"
legacy = METRICS_DIR / "fusion_summary.csv"

if not (sum_5.exists() and sum_6.exists()):
    if legacy.exists():
        st.info("Only the legacy single-file fusion output is present. "
                "Regenerate the 5-vs-6 pair with:\n\n"
                "`python scripts/07_fusion.py --exclude tft --suffix _5models`\n"
                "`python scripts/07_fusion.py --suffix _6models`")
        summary = pd.read_csv(legacy)
        summary["variant"] = "legacy"
        _loaded = [legacy]
    else:
        st.warning("No fusion summaries found. Run `python scripts/07_fusion.py`.")
        st.stop()
else:
    df5 = pd.read_csv(sum_5); df5["variant"] = "5models"
    df6 = pd.read_csv(sum_6); df6["variant"] = "6models"
    summary = pd.concat([df5, df6], ignore_index=True)
    _loaded = [sum_5, sum_6]

coef_path = METRICS_DIR / "fusion_coefficients_6models.csv"
if not coef_path.exists():
    coef_path = METRICS_DIR / "fusion_coefficients.csv"
coef = pd.read_csv(coef_path) if coef_path.exists() else pd.DataFrame()
if coef_path.exists():
    _loaded.append(coef_path)

# Judged against what was actually read above, so the banner cannot name a
# file that contributes nothing to this page.
_stale_banner(*_loaded)


# Hoisted — used by several tables below.
def _highlight_lift(val):
    if pd.isna(val):
        return ""
    if val > 0.03:
        return "background-color: #d4edda; color: #155724"
    if val > 0:
        return "background-color: #fff3cd; color: #856404"
    if val == 0:
        return "background-color: #e2e3e5; color: #383d41"
    return "background-color: #f8d7da; color: #721c24"


def _highlight_delta(val):
    if pd.isna(val):
        return ""
    if val > 0.02:
        return "background-color: #d4edda; color: #155724"
    if val > 0:
        return "background-color: #fff3cd; color: #856404"
    if val == 0:
        return "background-color: #e2e3e5; color: #383d41"
    return "background-color: #f8d7da; color: #721c24"


# --------------------------------------------------------------------------- #
# Aggregate once
# --------------------------------------------------------------------------- #
variants_present = sorted(summary["variant"].unique())

lb = summary.groupby(["method", "variant"]).agg(
    mean_r2=("r2", "mean"),
    median_r2=("r2", "median"),
    crops_positive_r2=("r2", lambda s: int((s > 0).sum())),
).reset_index()

# Method-ordered pivot: rows = method, columns = variant, values = mean R²
pivot_r2 = lb.pivot(index="method", columns="variant", values="mean_r2")
pivot_r2 = pivot_r2.sort_values(pivot_r2.columns[-1], ascending=False)
if "5models" in pivot_r2.columns and "6models" in pivot_r2.columns:
    pivot_r2["Δ (6-5)"] = pivot_r2["6models"] - pivot_r2["5models"]

# Headline row for whichever variant is preferred (prefer 6 if present)
prefer = "6models" if "6models" in variants_present else variants_present[-1]
best_row = lb[lb["variant"] == prefer].sort_values("mean_r2", ascending=False).iloc[0]

_clim = load_climatology_baseline()
clim_mean = float(_clim["r2"].mean()) if not _clim.empty else None

# --------------------------------------------------------------------------- #
# PRIMARY 1 — the KPI row, with the learn-nothing reference in it
# --------------------------------------------------------------------------- #
k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    kpi(f"Best method ({prefer})", str(best_row["method"]))
with k2:
    kpi("Best mean R²", f"{best_row['mean_r2']:.3f}",
        delta=(f"{best_row['mean_r2'] - clim_mean:+.3f} vs climatology"
               if clim_mean is not None else None))
with k3:
    single = lb[(lb["variant"] == prefer) & (lb["method"] == "best_single_by_val")]
    kpi("Val-best single (baseline)",
        f"{float(single['mean_r2'].iloc[0]):.3f}" if not single.empty else "—")
with k4:
    if "5models" in variants_present and "6models" in variants_present:
        b5 = lb[(lb["variant"] == "5models") & (lb["method"] == best_row["method"])]
        kpi("Δ 6-vs-5 for this method",
            f"{best_row['mean_r2'] - float(b5['mean_r2'].iloc[0]):+.3f}"
            if not b5.empty else "—")
    else:
        kpi("Δ 6-vs-5 for this method", "—")
with k5:
    kpi("Climatology baseline mean R²",
        f"{clim_mean:.3f}" if clim_mean is not None else "—")

if clim_mean is not None:
    _n_beat = int((pivot_r2[prefer].dropna() > clim_mean).sum())
    st.caption(
        f"The climatology baseline emits each district's train-fitted "
        f"baseline and learns no anomaly, scoring mean R² {clim_mean:.4f} on "
        f"the same crops (`climatology_baseline.csv`). "
        f"{_n_beat} of {int(pivot_r2[prefer].notna().sum())} methods in the "
        f"{prefer} pool beat it."
    )
else:
    st.caption("No `results/metrics/climatology_baseline.csv` on disk, so "
               "there is no learn-nothing reference for the R² values below.")

# --------------------------------------------------------------------------- #
# PRIMARY 2 — the one table: mean R² per method × variant
# --------------------------------------------------------------------------- #
st.markdown("#### Method leaderboard — mean R² across "
            f"{summary['crop'].nunique()} crops")

disp = pivot_r2.reset_index().rename(columns={"method": "Method"})
fmt = {c: "{:.3f}" for c in disp.columns if c not in ("Method", "Δ (6-5)")}
if "Δ (6-5)" in disp.columns:
    fmt["Δ (6-5)"] = "{:+.3f}"
    styled = disp.style.format(fmt).map(_highlight_delta, subset=["Δ (6-5)"])
else:
    styled = disp.style.format(fmt)
st.dataframe(styled, hide_index=True, width="stretch")

if "Δ (6-5)" in pivot_r2.columns:
    _d = pivot_r2["Δ (6-5)"].dropna()
    _n_up = int((_d > 0).sum())
    _worst, _worst_v = _d.idxmin(), _d.min()
    _best, _best_v = _d.idxmax(), _d.max()
    # How often is TFT actually the val-selected model in the 6-model pool?
    _bsbv6 = summary[(summary["variant"] == "6models")
                     & (summary["method"] == "best_single_by_val")]
    _tft_picked = int((_bsbv6["best_model"].astype(str) == "tft").sum()) \
        if "best_model" in _bsbv6.columns else 0
    st.caption(
        f"Δ = adding TFT to the pool. {_n_up}/{len(_d)} methods improve; "
        f"worst is `{_worst}` ({_worst_v:+.3f}), best is `{_best}` "
        f"({_best_v:+.3f}). Validation picks TFT for {_tft_picked}/"
        f"{_bsbv6['crop'].nunique()} crops."
    )

# --------------------------------------------------------------------------- #
# SECONDARY — per-crop detail (the variant selector drives only this block)
# --------------------------------------------------------------------------- #
with st.expander("Per-crop detail", expanded=False):
    if len(variants_present) > 1:
        sel_variant = st.radio(
            "Variant", variants_present,
            index=variants_present.index("6models")
            if "6models" in variants_present else 0,
            horizontal=True, key="fusion_variant",
        )
    else:
        sel_variant = variants_present[0]
    sub = summary[summary["variant"] == sel_variant].copy()

    # ---- per-crop best-method showdown vs val-best single ------------------
    st.markdown(f"##### Best fusion method vs val-best single ({sel_variant})")

    baseline_per_crop = sub[sub["method"] == "best_single_by_val"][
        ["crop", "r2"]].rename(columns={"r2": "baseline_r2"})
    fusion_only = sub[sub["method"] != "best_single_by_val"]

    idx = fusion_only.groupby("crop")["r2"].idxmax()
    best_per_crop = fusion_only.loc[idx, ["crop", "method", "r2", "rmse"]].rename(
        columns={"method": "best_fusion_method",
                 "r2": "best_fusion_r2", "rmse": "best_fusion_rmse"})
    showdown = best_per_crop.merge(baseline_per_crop, on="crop")
    showdown["lift"] = showdown["best_fusion_r2"] - showdown["baseline_r2"]
    showdown = showdown.sort_values("lift", ascending=False)

    def _note(row):
        if pd.isna(row["lift"]):
            return ""
        if row["baseline_r2"] < 0 and row["best_fusion_r2"] > 0:
            return "fusion rescues broken single-model pick"
        if row["lift"] >= 0.15:
            return "large lift"
        if row["lift"] >= 0.05:
            return "meaningful lift"
        if row["lift"] >= 0:
            return "marginal"
        return "fusion cannot beat single best"

    showdown["note"] = showdown.apply(_note, axis=1)

    # Overlay taxonomy prediction + observed best model per crop
    tax = load_taxonomy()
    mc_here = load_multicrop_summary()
    if tax and not mc_here.empty:
        predicted_map = {c: (v or {}).get("primary", "—")
                         for c, v in tax.get("predicted_winners", {}).items()}
        observed_best = (mc_here.sort_values("r2", ascending=False)
                                .groupby("crop", as_index=False)
                                .first()[["crop", "model", "r2"]]
                                .rename(columns={"model": "observed_best_model",
                                                 "r2": "observed_best_r2"}))
        showdown = showdown.merge(observed_best, on="crop", how="left")
        showdown["taxonomy_predicted"] = showdown["crop"].map(predicted_map)
        showdown["taxonomy_hit"] = showdown.apply(
            lambda r: "hit" if r["taxonomy_predicted"] == r["observed_best_model"]
                      else "miss", axis=1)

    cols = {"crop": "Crop", "best_fusion_method": "Best fusion method",
            "best_fusion_r2": "Best fusion R²",
            "best_fusion_rmse": "Best fusion RMSE",
            "baseline_r2": "Val-best single R²",
            "lift": "Lift (Δ)", "note": "Note"}
    if "taxonomy_predicted" in showdown.columns:
        cols["taxonomy_predicted"] = "Taxonomy pick"
        cols["observed_best_model"] = "Observed best model"
        cols["observed_best_r2"] = "Observed best R²"
        cols["taxonomy_hit"] = "Taxonomy verdict"

    display = showdown.rename(columns=cols)
    fmt = {"Best fusion R²": "{:.3f}", "Best fusion RMSE": "{:.2f}",
           "Val-best single R²": "{:.3f}", "Lift (Δ)": "{:+.3f}"}
    if "Observed best R²" in display.columns:
        fmt["Observed best R²"] = "{:.3f}"
    styled = display.style.format(fmt).map(_highlight_lift, subset=["Lift (Δ)"])

    if "Taxonomy verdict" in display.columns:
        def _verdict_color(v):
            if v == "hit":
                return "background-color: #d4edda; color: #155724"
            if v == "miss":
                return "background-color: #f8d7da; color: #721c24"
            return ""
        styled = styled.map(_verdict_color, subset=["Taxonomy verdict"])

    st.dataframe(styled, hide_index=True, width="stretch")

    _bits = []
    if "Taxonomy verdict" in display.columns:
        _bits.append(f"taxonomy a-priori pick matches the observed best single "
                     f"model on {int((showdown['taxonomy_hit'] == 'hit').sum())}/"
                     f"{len(showdown)} crops")
    _bits.append(f"best fusion method beats the val-best single model on "
                 f"{int((showdown['lift'] > 0).sum())}/{len(showdown)} crops")
    _n_rescues = int(((showdown["baseline_r2"] < 0)
                      & (showdown["best_fusion_r2"] > 0)).sum())
    if _n_rescues:
        _bits.append(f"rescues {_n_rescues} crop(s) where validation picked a "
                     "model that went negative on the 2024-25 test split")
    st.caption("; ".join(_bits).capitalize() + ".")

    # ---- per-crop R² across all methods -----------------------------------
    st.markdown("##### R² per (crop, method)")
    pivot = sub.pivot(index="crop", columns="method", values="r2")
    # Baseline column first, then every fusion method present in the CSV ordered
    # by mean R². Do NOT hardcode a method list — the generating script has
    # added methods over time and a fixed list silently hides them.
    _rest = [m for m in pivot.mean(axis=0).sort_values(ascending=False).index
             if m != "best_single_by_val"]
    pivot = pivot.reindex(
        columns=(["best_single_by_val"] if "best_single_by_val" in pivot.columns
                 else []) + _rest)

    fig = px.imshow(
        pivot, color_continuous_scale="RdYlGn",
        color_continuous_midpoint=0, zmin=-0.5, zmax=0.8,
        labels=dict(x="Method", y="Crop", color="R²"),
        aspect="auto", text_auto=".2f",
        title=f"{pivot.shape[1]} methods — leftmost column is the "
              "best-single-by-val baseline",
    )
    fig.update_layout(height=max(440, 26 * len(pivot)))
    st.plotly_chart(fig, width="stretch")

    # ---- per-crop lift for one chosen method ------------------------------
    st.markdown("##### Per-crop lift vs the best-single baseline")
    fusion_methods = [m for m in sub["method"].unique()
                      if m != "best_single_by_val"]
    sel_method = st.selectbox("Fusion method", fusion_methods,
                              index=fusion_methods.index("simple_avg")
                              if "simple_avg" in fusion_methods else 0,
                              key="fusion_lift_method")
    method_rows = sub[sub["method"] == sel_method][
        ["crop", "r2", "rmse"]].merge(baseline_per_crop, on="crop")
    method_rows["lift"] = method_rows["r2"] - method_rows["baseline_r2"]
    method_rows = method_rows.sort_values("lift", ascending=False)

    st.dataframe(
        method_rows.rename(columns={
            "crop": "Crop", "r2": f"{sel_method} R²",
            "baseline_r2": "Best-single R²",
            "rmse": f"{sel_method} RMSE", "lift": "Lift (Δ)",
        }).style.format({f"{sel_method} R²": "{:.3f}",
                         "Best-single R²": "{:.3f}",
                         f"{sel_method} RMSE": "{:.2f}",
                         "Lift (Δ)": "{:+.3f}"})
        .map(_highlight_lift, subset=["Lift (Δ)"]),
        hide_index=True, width="stretch",
    )

# --------------------------------------------------------------------------- #
# SECONDARY — meta-learner coefficients (Ridge / NNLS / weighted-avg only)
# --------------------------------------------------------------------------- #
if not coef.empty:
    with st.expander("Meta-learner weights", expanded=False):
        available_methods = sorted(coef["method"].unique())
        coef_method = st.selectbox("Method", available_methods,
                                   key="coef_method")
        cdata = coef[(coef["method"] == coef_method) &
                     (coef["model"] != "_intercept_")]
        if cdata.empty:
            st.info("No coefficients for this method.")
        else:
            pivot_c = cdata.pivot(index="crop", columns="model",
                                  values="coefficient").fillna(0)
            fig = px.imshow(
                pivot_c, color_continuous_scale="RdBu_r",
                color_continuous_midpoint=0,
                labels=dict(x="Base model", y="Crop", color="Weight"),
                aspect="auto", text_auto=".2f",
                title=f"{coef_method} weights per (crop, base model)",
            )
            fig.update_layout(height=420)
            st.plotly_chart(fig, width="stretch")
            st.caption("Negative weights mean the meta-learner uses that base "
                       "prediction in the opposite direction — a sign the "
                       "validation and test years disagree.")

# --------------------------------------------------------------------------- #
# SECONDARY — provenance
# --------------------------------------------------------------------------- #
with st.expander("Provenance and methodology", expanded=False):
    _n_base = sorted(pd.read_csv(sum_6)["n_base"].unique()) if sum_6.exists() else []
    _cache_hits = sorted(METRICS_DIR.glob("fusion_predictions_*.parquet"))
    _files = [p for p in (sum_5, sum_6, legacy, coef_path) if p.exists()]
    st.dataframe(
        pd.DataFrame([{
            "Artefact": p.name,
            "Last written": datetime.fromtimestamp(
                p.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
        } for p in _files]),
        hide_index=True, width="stretch",
    )
    st.caption(
        "**Methodology:** base models predict on the 2023 validation split; "
        "fusion weights are fit there and applied to the 2024-25 test split, "
        "against a 'best single model by validation' baseline. The 5-model "
        "variant excludes TFT from the pool, the 6-model variant includes it"
        + (f" (pool sizes present in the CSV: {_n_base})." if _n_base else ".")
        + " Regenerate: `python scripts/07_fusion.py --exclude tft --suffix "
          "_5models` and `python scripts/07_fusion.py --suffix _6models`. "
        + (f"Prediction cache present ({len(_cache_hits)} parquet file(s)), so "
           "re-runs are fast."
           if _cache_hits else
           "No `fusion_predictions_*.parquet` cache on disk, so a re-run "
           "re-predicts from the checkpoints and takes tens of minutes.")
    )

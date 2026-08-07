"""Literature comparison — how does our best per-crop R² sit against
published EU/German crop-yield ML benchmarks?

One question, one chart. The default view is a KPI row and a dot plot: our
best R² per crop, the published range as a grey bar, and — new here — the
learn-nothing climatology baseline as its own marker on the same axis. That
third mark is what makes the comparison honest: our target is an anomaly
added back to a per-district climatological baseline, so a large share of
every R² below is the baseline, not learned signal.

The crop-by-crop table, the per-crop citations and the all-models table are
still here, collapsed.

IMPORTANT CAVEAT (also on the page): each published study uses a different
region, feature set, year range, target definition, and split, so the
"Verdict" column is directional only. Use this page to frame related work,
not as a citation table.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import (CHECKPOINTS_DIR, METRICS_DIR, kpi,
                   load_climatology_baseline, load_taxonomy, page_header)


def _stale_banner(*metric_paths: Path) -> None:
    """Warn when a metric file predates the newest checkpoint on disk."""
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

st.set_page_config(page_title="Literature — Crop Yield Benchmark",
                   layout="wide")
page_header("Literature comparison",
            "How does our best per-crop R² sit against published EU / German "
            "crop-yield ML studies?")

# --------------------------------------------------------------------------- #
# Published benchmarks (hand-curated, EU/German focus)
# --------------------------------------------------------------------------- #
LITERATURE = pd.DataFrame([
    # Winter wheat — heavily studied
    dict(crop="winter_wheat", study="Mateo-Sanchís et al. 2019",
         region="Iberia / EU", model="Deep learning + NDVI/LST",
         r2_low=0.55, r2_high=0.65,
         ref="IEEE TGRS, doi:10.1109/TGRS.2018.2870527"),
    dict(crop="winter_wheat", study="Paudel et al. 2021",
         region="EU MARS (pan-EU)", model="XGBoost + biophysical",
         r2_low=0.48, r2_high=0.55,
         ref="Sci. Reports 11, doi:10.1038/s41598-021-94175-1"),
    dict(crop="winter_wheat", study="Cao et al. 2021",
         region="Global", model="CNN + LSTM",
         r2_low=0.60, r2_high=0.65,
         ref="Remote Sens. Environ. 264, doi:10.1016/j.rse.2021.112408"),

    # Winter rapeseed — less studied
    dict(crop="winter_rapeseed", study="Marszalek 2022",
         region="Poland", model="Random Forest + NDVI",
         r2_low=0.45, r2_high=0.55,
         ref="Remote Sensing 14, doi:10.3390/rs14041056 (illustrative)"),
    dict(crop="winter_rapeseed", study="Sus et al. 2010 (crop-model + ML)",
         region="EU", model="Process model + statistical",
         r2_low=0.45, r2_high=0.55,
         ref="Ag. & Forest Meteorology 150, doi:10.1016/j.agrformet.2010.04.006"),

    # Silage maize
    dict(crop="silage_maize", study="Schlund et al. 2020",
         region="Germany", model="Random Forest",
         r2_low=0.50, r2_high=0.60,
         ref="Sensors 20, German maize specific"),
    dict(crop="silage_maize", study="Tao et al. 2020",
         region="EU", model="RF + biophysical drivers",
         r2_low=0.40, r2_high=0.50,
         ref="Global Change Biology 26"),
    dict(crop="silage_maize", study="Khaki & Wang 2019 (US corn — non-EU)",
         region="US Corn Belt", model="DNN",
         r2_low=0.80, r2_high=0.85,
         ref="Frontiers in Plant Science 10 — not directly comparable to DE"),

    # Sugar beet — structurally hard
    dict(crop="sugar_beet", study="Sieling et al. 2018",
         region="Germany", model="Multiple regression",
         r2_low=0.35, r2_high=0.50,
         ref="Field Crops Research 224"),

    # Winter barley
    dict(crop="winter_barley", study="Folberth et al. 2019",
         region="EU", model="Process model + ML residuals",
         r2_low=0.50, r2_high=0.60,
         ref="Environmental Research Letters 14"),
    dict(crop="winter_barley", study="Schauberger et al. 2018",
         region="Germany (cereals)", model="Statistical models",
         r2_low=0.45, r2_high=0.55,
         ref="Sci. Reports 8, doi:10.1038/s41598-018-25946-z"),

    # Rye, triticale, oats — limited crop-specific literature, grouped as small grains
    dict(crop="rye", study="Schauberger et al. 2018",
         region="Germany", model="Statistical (small grains)",
         r2_low=0.40, r2_high=0.50,
         ref="Sci. Reports 8"),
    dict(crop="triticale", study="(no dedicated study located)",
         region="—", model="—",
         r2_low=float("nan"), r2_high=float("nan"),
         ref="Triticale lacks crop-specific ML studies — usually lumped into small grains"),
    dict(crop="oats", study="Schauberger et al. 2018",
         region="Germany", model="Statistical (small grains)",
         r2_low=0.40, r2_high=0.50,
         ref="Sci. Reports 8"),

    # Potatoes
    dict(crop="potatoes", study="Sun et al. 2020",
         region="China", model="RF + multi-source remote sensing",
         r2_low=0.60, r2_high=0.70,
         ref="Computers and Electronics in Agriculture 175"),
    dict(crop="potatoes", study="Vasilakos et al. 2021",
         region="Greece", model="ML ensemble",
         r2_low=0.45, r2_high=0.55,
         ref="Agronomy 11"),

    # Summer barley — no good NUTS-3 EU benchmark found
    dict(crop="summer_barley", study="(no NUTS-3 EU benchmark located)",
         region="—", model="—",
         r2_low=float("nan"), r2_high=float("nan"),
         ref="Most cereal yield studies pool winter+summer barley"),
])
LITERATURE["r2_mid"] = (LITERATURE["r2_low"] + LITERATURE["r2_high"]) / 2

# --------------------------------------------------------------------------- #
# Load our results
# --------------------------------------------------------------------------- #
summary_path = METRICS_DIR / "multicrop_summary.csv"
if not summary_path.exists():
    st.error(
        "`results/metrics/multicrop_summary.csv` not found. "
        "Run `python scripts/05b_eval_multicrop.py` to generate it."
    )
    st.stop()
_stale_banner(summary_path)
ours = pd.read_csv(summary_path)
# All models compete. TFT used to be hard-excluded here on the grounds of a
# broken checkpoint; that is no longer true, and its exclusion changed no
# row anyway because it is never the per-crop best.
best_per_crop = (ours.loc[ours.groupby("crop")["r2"].idxmax()]
                 [["crop", "model", "r2", "rmse", "n"]]
                 .rename(columns={"model": "our_best_model",
                                  "r2": "our_r2", "rmse": "our_rmse",
                                  "n": "our_n"}))

climatology = load_climatology_baseline()
clim_by_crop = (climatology.set_index("crop")["r2"]
                if not climatology.empty else pd.Series(dtype=float))


def _verdict(our_r2: float, lit_low: float, lit_high: float) -> str:
    """Classify our R² vs the literature range."""
    if pd.isna(lit_low) or pd.isna(lit_high):
        return "no benchmark found"
    if our_r2 >= lit_high - 0.02:
        return "above range"
    if our_r2 >= lit_low - 0.02:
        return "within range"
    if our_r2 >= lit_low - 0.10:
        return "just below"
    return "below range"


# Aggregate literature: one row per crop with min-low and max-high
lit_per_crop = LITERATURE.groupby("crop", dropna=False).agg(
    lit_low=("r2_low", "min"),
    lit_high=("r2_high", "max"),
    # Count only rows that actually carry an R² range — the "(no dedicated
    # study located)" placeholders are not studies.
    n_studies=("r2_low", "count"),
).reset_index()

leaderboard = best_per_crop.merge(lit_per_crop, on="crop", how="left")
leaderboard["verdict"] = leaderboard.apply(
    lambda r: _verdict(r["our_r2"], r["lit_low"], r["lit_high"]), axis=1)
leaderboard["climatology_r2"] = leaderboard["crop"].map(clim_by_crop)
leaderboard = leaderboard.sort_values("our_r2", ascending=False)

# --------------------------------------------------------------------------- #
# PRIMARY 1 — the KPI row, with the learn-nothing reference in it
# --------------------------------------------------------------------------- #
n_crops = len(leaderboard)
n_bench = int(leaderboard["lit_low"].notna().sum())
n_within = int(leaderboard["verdict"].isin(["within range", "above range"]).sum())
our_mean = float(leaderboard["our_r2"].mean())
clim_mean = float(clim_by_crop.mean()) if not clim_by_crop.empty else None
n_beat_clim = int((leaderboard["our_r2"] > leaderboard["climatology_r2"]).sum()) \
    if leaderboard["climatology_r2"].notna().any() else 0

k1, k2, k3, k4 = st.columns(4)
with k1:
    kpi("Crops with a published benchmark", f"{n_bench} / {n_crops}")
with k2:
    kpi("Within or above published range", f"{n_within} / {n_bench}")
with k3:
    kpi("Our mean best-per-crop R²", f"{our_mean:.3f}",
        delta=(f"{our_mean - clim_mean:+.3f} vs climatology"
               if clim_mean is not None else None))
with k4:
    kpi("Climatology baseline mean R²",
        f"{clim_mean:.3f}" if clim_mean is not None else "—")

st.caption(
    f"'Ours' is the best of {ours['model'].nunique()} models picked per crop "
    "on the test split — an oracle upper bound, not a deployable single "
    "model. "
    + (f"It beats the learn-nothing climatology baseline on {n_beat_clim}/"
       f"{n_crops} crops (`climatology_baseline.csv`)."
       if clim_mean is not None else
       "No `climatology_baseline.csv` on disk, so the learn-nothing "
       "reference is unknown.")
)

# --------------------------------------------------------------------------- #
# PRIMARY 2 — the one chart
# --------------------------------------------------------------------------- #
st.markdown("#### Our best R² vs published range, per crop")

order = leaderboard["crop"].tolist()
fig = go.Figure()
for _, row in leaderboard.iterrows():
    if pd.notna(row["lit_low"]):
        fig.add_shape(
            type="line",
            x0=row["lit_low"], x1=row["lit_high"],
            y0=row["crop"], y1=row["crop"],
            line=dict(color="rgba(120,120,120,0.55)", width=8),
        )
if leaderboard["climatology_r2"].notna().any():
    fig.add_trace(go.Scatter(
        x=leaderboard["climatology_r2"], y=leaderboard["crop"],
        mode="markers", name="Climatology baseline (learn-nothing)",
        marker=dict(symbol="diamond", size=13, color="#7a5195",
                    line=dict(color="white", width=1.5)),
        hovertemplate="%{y}<br>climatology R²: %{x:.3f}<extra></extra>",
    ))
fig.add_trace(go.Scatter(
    x=leaderboard["our_r2"], y=leaderboard["crop"],
    mode="markers", name=f"Ours (best of {ours['model'].nunique()})",
    marker=dict(size=14, color="#1f77b4", line=dict(color="white", width=2)),
    customdata=leaderboard[["our_best_model"]].values,
    hovertemplate="%{y}<br>our R²: %{x:.3f}<br>model: %{customdata[0]}"
                  "<extra></extra>",
))
fig.add_vline(x=0, line_dash="dot", line_color="grey")
fig.update_layout(
    height=460, margin=dict(l=10, r=10, t=30, b=10),
    xaxis=dict(range=[-0.1, 1.0], title="R²", title_standoff=8),
    yaxis=dict(categoryorder="array", categoryarray=order[::-1], title=""),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
)
st.plotly_chart(fig, width="stretch")
st.caption(
    "Grey bar = published range (lowest to highest reported R² across the "
    "curated studies for that crop). The purple diamond is the learn-nothing "
    "climatology baseline for the same crop and the same split: the gap "
    "between diamond and circle is the only part of our R² that is learned "
    "signal. Published studies differ in region, target, and split — read "
    "the bars as framing, not as a head-to-head."
)

# --------------------------------------------------------------------------- #
# SECONDARY — crop-by-crop table
# --------------------------------------------------------------------------- #
with st.expander("Crop-by-crop table", expanded=False):
    # Overlay taxonomy: phenology family + taxonomy primary + whether the
    # taxonomy's primary matches our best model.
    tax = load_taxonomy()
    if tax:
        tax_families = {c: v.get("phenology_family", "—")
                        for c, v in tax.get("crops", {}).items()}
        tax_predictions = {c: (v or {}).get("primary", "—")
                           for c, v in tax.get("predicted_winners", {}).items()}
        leaderboard["phenology_family"] = (leaderboard["crop"]
                                           .map(tax_families).fillna("—"))
        leaderboard["taxonomy_pick"] = (leaderboard["crop"]
                                        .map(tax_predictions).fillna("—"))
        leaderboard["taxonomy_match"] = leaderboard.apply(
            lambda r: "hit" if r["taxonomy_pick"] == r["our_best_model"]
                      else ("—" if r["taxonomy_pick"] == "—" else "miss"),
            axis=1,
        )

    def _fmt_range(lo, hi):
        if pd.isna(lo) or pd.isna(hi):
            return "—"
        return f"{lo:.2f} – {hi:.2f}"

    disp = leaderboard.copy()
    disp["lit_range"] = disp.apply(
        lambda r: _fmt_range(r["lit_low"], r["lit_high"]), axis=1)

    base_cols = ["crop", "our_best_model", "our_r2", "climatology_r2",
                 "lit_range", "n_studies", "verdict"]
    rename_map = {
        "crop": "Crop", "our_best_model": "Our best model",
        "our_r2": "Our R²", "climatology_r2": "Climatology R²",
        "lit_range": "Published R² range",
        "n_studies": "# studies", "verdict": "Verdict",
    }
    if "phenology_family" in disp.columns:
        base_cols = ["crop", "phenology_family", "our_best_model",
                     "taxonomy_pick", "taxonomy_match", "our_r2",
                     "climatology_r2", "lit_range", "n_studies", "verdict"]
        rename_map.update({
            "phenology_family": "Phenology family",
            "taxonomy_pick": "Taxonomy pick",
            "taxonomy_match": "Taxonomy vs ours",
        })

    disp = disp[base_cols].rename(columns=rename_map)

    def _tax_color(val):
        if val == "hit":
            return "background-color: #d4edda; color: #155724"
        if val == "miss":
            return "background-color: #f8d7da; color: #721c24"
        return ""

    styled = disp.style.format({"Our R²": "{:.3f}",
                                "Climatology R²": "{:.3f}"}, na_rep="—")
    if "Taxonomy vs ours" in disp.columns:
        styled = styled.map(_tax_color, subset=["Taxonomy vs ours"])
    st.dataframe(styled, hide_index=True, width="stretch")

    if tax:
        n_hits = int((leaderboard["taxonomy_match"] == "hit").sum())
        n_total = int((leaderboard["taxonomy_match"] != "—").sum())
        st.caption(f"Taxonomy's a-priori pick matches our observed best model "
                   f"on {n_hits}/{n_total} crops.")

# --------------------------------------------------------------------------- #
# SECONDARY — per-crop drill-down with citations
# --------------------------------------------------------------------------- #
with st.expander("Per-crop detail and citations", expanded=False):
    crop_opts = sorted(best_per_crop["crop"].tolist())
    sel_crop = st.selectbox(
        "Crop", crop_opts,
        index=crop_opts.index("winter_wheat")
        if "winter_wheat" in crop_opts else 0,
    )

    our_row = best_per_crop[best_per_crop["crop"] == sel_crop].iloc[0]
    _clim_here = float(clim_by_crop[sel_crop]) \
        if sel_crop in clim_by_crop.index else None

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi("Our best model", str(our_row["our_best_model"]).upper())
    with c2:
        kpi("Our R² (test split)", f"{our_row['our_r2']:.3f}",
            delta=(f"{our_row['our_r2'] - _clim_here:+.3f} vs climatology"
                   if _clim_here is not None else None))
    with c3:
        kpi("Climatology R²",
            f"{_clim_here:.3f}" if _clim_here is not None else "—")
    with c4:
        kpi("Our RMSE (dt/ha)", f"{our_row['our_rmse']:.1f}")

    lit_rows = LITERATURE[LITERATURE["crop"] == sel_crop].copy()
    st.markdown("##### Published benchmarks")
    if lit_rows.empty:
        st.info("No curated literature for this crop in the table.")
    else:
        lit_disp = lit_rows[["study", "region", "model", "r2_low", "r2_high",
                             "ref"]].rename(columns={
            "study": "Study", "region": "Region", "model": "Model",
            "r2_low": "R² low", "r2_high": "R² high", "ref": "Citation",
        })
        st.dataframe(
            lit_disp.style.format({"R² low": "{:.2f}", "R² high": "{:.2f}"},
                                  na_rep="—"),
            hide_index=True, width="stretch",
        )

    st.markdown(f"##### All {ours['model'].nunique()} models on {sel_crop}")
    all_models = ours[ours["crop"] == sel_crop].sort_values("r2",
                                                           ascending=False)
    st.dataframe(
        all_models[["model", "r2", "rmse", "mae", "n"]].rename(
            columns={"model": "Model", "r2": "R²", "rmse": "RMSE",
                     "mae": "MAE", "n": "n"}
        ).style.format({"R²": "{:.3f}", "RMSE": "{:.2f}", "MAE": "{:.2f}"}),
        hide_index=True, width="stretch",
    )

# --------------------------------------------------------------------------- #
# SECONDARY — comparability caveat and provenance
# --------------------------------------------------------------------------- #
with st.expander("Why these R² are not directly comparable", expanded=False):
    st.info(
        "**These R² values are not directly comparable to the published "
        "ones.** Our prediction is an anomaly that is added back to a "
        "per-district, per-crop climatological baseline before scoring, so "
        "part of the R² above comes from that baseline offset rather than "
        "from learned signal — which is exactly what the climatology marker "
        "on the chart quantifies. The split is also only the 2024-25 seasons, "
        "so the denominator carries almost no inter-annual variance. The "
        "published studies differ again in region, target, and split. Treat "
        "the Verdict column as directional."
        + ("" if not climatology.empty else
           " No climatology-baseline row exists in `results/metrics/`, so the "
           "page cannot show how much of each R² a learn-nothing predictor "
           "already achieves.")
    )
    _means = ours.groupby("model")["r2"].mean().sort_values(ascending=False)
    st.caption(
        "Literature entries live in the `LITERATURE` DataFrame at the top of "
        "`dashboard/pages/16_Literature.py`; the R² ranges there are "
        "best-effort estimates, not verified replications — check them "
        "against the PDFs before the report. Our side is read live from "
        "`multicrop_summary.csv` (written "
        f"{datetime.fromtimestamp(summary_path.stat().st_mtime):%Y-%m-%d %H:%M}"
        "); mean R² across the crops is "
        + ", ".join(f"{m} {v:.3f}" for m, v in _means.items()) + "."
    )

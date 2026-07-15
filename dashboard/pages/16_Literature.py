"""Literature comparison — our best per-crop R² vs published EU/German
crop-yield ML benchmarks.

IMPORTANT CAVEAT (also shown to the user): each published study uses a
different region, feature set, year range, and validation split. The
numbers here are directional comparisons ("same ballpark", "below",
"above"), not strict apples-to-apples R² values. Use this page to frame
the related-work section of the report, not as a final citation table —
verify the exact R² values against the PDFs before submission.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import METRICS_DIR, divider, kpi, load_taxonomy, page_header

st.set_page_config(page_title="Literature — Crop Yield Benchmark",
                   layout="wide")
page_header("Literature comparison",
            "Our best per-crop R² (from `multicrop_summary.csv`) vs "
            "published EU / German crop-yield ML studies. Different "
            "studies use different data, splits, and metrics — treat these "
            "as directional comparisons, not strict head-to-heads.")
st.caption("_How our best-per-crop R² sits against hand-curated published EU and German benchmarks — directional, apples-to-oranges, with caveats loud and up front._")

st.warning(
    "**Apples-to-oranges warning.** Each published paper here uses a "
    "different region, feature set, year range, and validation split. "
    "Some report R² on temporal splits (like ours), others on spatial "
    "leave-one-out, others on within-year cross-validation. The numbers "
    "shown are *best-effort literature ranges*, not exact replications. "
    "Before the final report, replace the R² ranges below with values "
    "verified against the actual PDFs."
)

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
ours = pd.read_csv(summary_path)
# Exclude TFT — see project memory: checkpoint is broken (R² ≈ -23 mean)
ours_no_tft = ours[ours["model"] != "tft"].copy()
best_per_crop = (ours_no_tft.loc[ours_no_tft.groupby("crop")["r2"].idxmax()]
                 [["crop", "model", "r2", "rmse", "n"]]
                 .rename(columns={"model": "our_best_model",
                                  "r2": "our_r2", "rmse": "our_rmse",
                                  "n": "our_n"}))

# --------------------------------------------------------------------------- #
# View 1 — Top-level leaderboard
# --------------------------------------------------------------------------- #
divider("Crop-by-crop summary")


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
    n_studies=("study", "count"),
).reset_index()

leaderboard = best_per_crop.merge(lit_per_crop, on="crop", how="left")
leaderboard["verdict"] = leaderboard.apply(
    lambda r: _verdict(r["our_r2"], r["lit_low"], r["lit_high"]), axis=1)
leaderboard = leaderboard.sort_values("our_r2", ascending=False)


# Build a display column for the literature range
def _fmt_range(lo, hi):
    if pd.isna(lo) or pd.isna(hi):
        return "—"
    return f"{lo:.2f} – {hi:.2f}"


# Overlay taxonomy: for each crop, show phenology family + taxonomy primary +
# whether the taxonomy's primary matches our best model
tax = load_taxonomy()
if tax:
    tax_families = {c: v.get("phenology_family", "—")
                    for c, v in tax.get("crops", {}).items()}
    tax_predictions = {c: (v or {}).get("primary", "—")
                       for c, v in tax.get("predicted_winners", {}).items()}
    leaderboard["phenology_family"] = leaderboard["crop"].map(tax_families).fillna("—")
    leaderboard["taxonomy_pick"] = leaderboard["crop"].map(tax_predictions).fillna("—")
    leaderboard["taxonomy_match"] = leaderboard.apply(
        lambda r: "hit" if r["taxonomy_pick"] == r["our_best_model"]
                  else ("—" if r["taxonomy_pick"] == "—" else "miss"),
        axis=1,
    )

leaderboard_disp = leaderboard.copy()
leaderboard_disp["lit_range"] = leaderboard_disp.apply(
    lambda r: _fmt_range(r["lit_low"], r["lit_high"]), axis=1)

base_cols = ["crop", "our_best_model", "our_r2", "lit_range", "n_studies", "verdict"]
rename_map = {
    "crop": "Crop", "our_best_model": "Our best model",
    "our_r2": "Our R²", "lit_range": "Published R² range",
    "n_studies": "# studies", "verdict": "Verdict",
}
if "phenology_family" in leaderboard_disp.columns:
    base_cols = (["crop", "phenology_family", "our_best_model", "taxonomy_pick",
                  "taxonomy_match", "our_r2", "lit_range", "n_studies", "verdict"])
    rename_map.update({
        "phenology_family": "Phenology family",
        "taxonomy_pick": "Taxonomy pick",
        "taxonomy_match": "Taxonomy vs ours",
    })

leaderboard_disp = leaderboard_disp[base_cols].rename(columns=rename_map)


def _tax_color(val):
    if val == "hit":
        return "background-color: #d4edda; color: #155724"
    if val == "miss":
        return "background-color: #f8d7da; color: #721c24"
    return ""


styled = leaderboard_disp.style.format({"Our R²": "{:.3f}"})
if "Taxonomy vs ours" in leaderboard_disp.columns:
    styled = styled.map(_tax_color, subset=["Taxonomy vs ours"])

st.dataframe(styled, hide_index=True, width="stretch")

if tax:
    n_hits = int((leaderboard["taxonomy_match"] == "hit").sum())
    n_total = int((leaderboard["taxonomy_match"] != "—").sum())
    st.caption(
        f"**Bridge between taxonomy and literature:** on **{n_hits}/{n_total}** "
        "crops the taxonomy's a-priori primary pick matches our observed best "
        "model, which is what we then compare to published EU/German "
        "benchmarks in the 'Verdict' column. The chain of reasoning is: "
        "crop biology → taxonomy prediction → empirical validation → "
        "literature comparison. Each column above is one link in that chain."
    )

# Visual: dot plot of our R² with literature range bars per crop
fig = px.scatter(
    leaderboard, x="our_r2", y="crop",
    hover_data=["our_best_model", "lit_low", "lit_high"],
    title="Our best R² vs published literature range, per crop",
    labels={"our_r2": "Our R² (best of 5 models)", "crop": ""},
)
for _, row in leaderboard.iterrows():
    if pd.notna(row["lit_low"]):
        fig.add_shape(
            type="line",
            x0=row["lit_low"], x1=row["lit_high"],
            y0=row["crop"], y1=row["crop"],
            line=dict(color="rgba(120,120,120,0.55)", width=8),
        )
fig.update_traces(marker=dict(size=14, color="#1f77b4",
                              line=dict(color="white", width=2)))
fig.update_layout(height=420,
                  xaxis=dict(range=[-0.1, 1.0], title_standoff=8),
                  showlegend=False)
fig.add_vline(x=0, line_dash="dot", line_color="grey")
st.plotly_chart(fig, width="stretch")
st.caption(
    "Blue dot = our best model's R² for the crop on the 2022-2023 test set. "
    "Grey bar = published literature range (lowest to highest reported R² "
    "across the curated studies)."
)

# --------------------------------------------------------------------------- #
# View 2 — Per-crop drill-down with citations
# --------------------------------------------------------------------------- #
divider("Per-crop drill-down")

sel_crop = st.selectbox(
    "Crop",
    sorted(best_per_crop["crop"].tolist()),
    index=sorted(best_per_crop["crop"].tolist()).index("winter_wheat")
    if "winter_wheat" in best_per_crop["crop"].tolist() else 0,
)

our_row = best_per_crop[best_per_crop["crop"] == sel_crop].iloc[0]
lit_rows = LITERATURE[LITERATURE["crop"] == sel_crop].copy()

k1, k2, k3 = st.columns(3)
with k1:
    kpi("Our best model", str(our_row["our_best_model"]).upper())
with k2:
    kpi("Our R² (2022-23 test)", f"{our_row['our_r2']:.3f}")
with k3:
    kpi("Our RMSE (dt/ha)", f"{our_row['our_rmse']:.1f}")

if lit_rows.empty:
    st.info("No curated literature for this crop in the table.")
else:
    st.markdown("##### Published benchmarks")
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

# --------------------------------------------------------------------------- #
# View 3 — All other models per crop (for "best wasn't the only one")
# --------------------------------------------------------------------------- #
divider("All 5 models on the selected crop")

all_models = ours_no_tft[ours_no_tft["crop"] == sel_crop].sort_values(
    "r2", ascending=False)
st.dataframe(
    all_models[["model", "r2", "rmse", "mae", "n"]].rename(
        columns={"model": "Model", "r2": "R²", "rmse": "RMSE",
                 "mae": "MAE", "n": "n"}
    ).style.format({"R²": "{:.3f}", "RMSE": "{:.2f}", "MAE": "{:.2f}"}),
    hide_index=True, width="stretch",
)

# --------------------------------------------------------------------------- #
# Footer
# --------------------------------------------------------------------------- #
st.markdown("---")
st.caption(
    "Editing this page: literature entries live in the `LITERATURE` "
    "DataFrame at the top of `dashboard/pages/13_Literature.py`. Update "
    "the R² ranges and citations from your actual reading list before "
    "the final report — the values shipped here are best-effort literature "
    "estimates, not verified replications. TFT is excluded from the "
    "leaderboard (broken checkpoint, R² ≈ -23 mean)."
)

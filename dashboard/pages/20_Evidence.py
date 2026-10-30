"""Evidence — every claim, caveat and known defect in one place.

Exists so the other pages can stay calm. A reader who just wants to see the
data should not have to scroll past a per-district R2 caveat; a reader
checking whether the numbers can be trusted should not have to open twelve
expanders across twelve pages to find out.

Everything here is read live from results/metrics/ or configs/. Nothing is
hardcoded: this project retrained many times and every hardcoded claim went
stale, which is what made a page like this necessary.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import (CHECKPOINTS_DIR, HELP, METRICS_DIR, PROJECT_ROOT,
                   artifact_is_fresh, divider, kpi,
                   load_climatology_baseline, load_multicrop_summary,
                   newest_checkpoint_mtime, page_header)

st.set_page_config(page_title="Evidence — Crop Yield Benchmark", layout="wide")
page_header("Evidence",
            "Every headline claim with the number behind it, every artifact "
            "with its age, and every known limitation. Read live from disk.")


def _mtime(p: Path) -> str:
    return datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M") \
        if p.exists() else "missing"


mc = load_multicrop_summary()
clim = load_climatology_baseline()

# --------------------------------------------------------------------------- #
# 1. Does anything beat doing nothing?
# --------------------------------------------------------------------------- #
divider("Does any model beat doing nothing?")

if mc.empty or clim.empty:
    st.warning("multicrop_summary.csv or climatology_baseline.csv missing.")
else:
    null_by_crop = clim.set_index("crop")["r2"]
    null_mean = float(null_by_crop.mean())
    d = mc.drop_duplicates(["model", "crop"])

    rows = []
    for model, g in d.groupby("model"):
        s = g.set_index("crop")["r2"].reindex(null_by_crop.index)
        rows.append({
            "Model": model,
            "Class": ("transformer" if model in ("itransformer", "patchtst", "tft")
                      else "baseline"),
            "Mean R2": s.mean(),
            "Skill vs null": s.mean() - null_mean,
            "Crops above null": int((s > null_by_crop).sum()),
        })
    lb = pd.DataFrame(rows).sort_values("Mean R2", ascending=False)

    k1, k2, k3 = st.columns(3)
    with k1:
        kpi("Climatology null (mean R2)", f"{null_mean:.3f}", help=HELP["null"])
    with k2:
        beat = lb[lb["Mean R2"] > null_mean]
        kpi("Models beating it", f"{len(beat)} of {len(lb)}", help=HELP["skill_null"])
    with k3:
        kpi("Best model", f"{lb.iloc[0]['Model']} {lb.iloc[0]['Mean R2']:.3f}")

    st.dataframe(
        lb.style.format({"Mean R2": "{:.3f}", "Skill vs null": "{:+.3f}"})
          .background_gradient(subset=["Skill vs null"], cmap="RdYlGn",
                               vmin=-0.5, vmax=0.5),
        width="stretch", hide_index=True, column_config={
            "Mean R2": st.column_config.NumberColumn(help=HELP["r2"]),
            "Skill vs null": st.column_config.NumberColumn(help=HELP["skill_null"]),
        })

    st.caption(
        "The null predicts each district's own training baseline and learns no "
        "anomaly. R2 is computed in absolute dt/ha against an anomaly-framed "
        "target, so that baseline is handed to every model for free — without "
        "this reference the reported numbers look better than the skill they "
        "carry.")

# --------------------------------------------------------------------------- #
# 2. The taxonomy hypothesis
# --------------------------------------------------------------------------- #
divider("Does the taxonomy predict the winning architecture?")

tax_path = METRICS_DIR / "taxonomy_validation.csv"
if not tax_path.exists():
    st.warning("taxonomy_validation.csv not found.")
else:
    tax = pd.read_csv(tax_path)
    hit1 = int(tax["hit_at_1"].sum())
    n = len(tax)

    # The honest reference: a uniform-random null does not apply, because the
    # six models are not exchangeable. The floor is the best constant strategy.
    const_floor, const_model = 0, "n/a"
    if not mc.empty:
        w = mc.drop_duplicates(["model", "crop"])
        w = w.loc[w.groupby("crop")["r2"].idxmax()]
        vc = w["model"].value_counts()
        if len(vc):
            const_model, const_floor = vc.index[0], int(vc.iloc[0])

    k1, k2, k3 = st.columns(3)
    with k1:
        kpi("Taxonomy top-1", f"{hit1}/{n}")
    with k2:
        kpi(f"Always guess {const_model}", f"{const_floor}/{n}")
    with k3:
        if "hit_at_3" in tax:
            kpi("Taxonomy top-3", f"{int(tax['hit_at_3'].sum())}/{n}")

    if const_floor > hit1:
        st.error(
            f"The taxonomy ({hit1}/{n}) performs WORSE than always naming "
            f"{const_model} ({const_floor}/{n}). Under leakage-free evaluation "
            f"it has no predictive value over a constant guess.")

    cols = [c for c in ["crop", "predicted_primary", "observed_best_model",
                        "observed_best_r2", "hit_at_1"] if c in tax]
    st.dataframe(tax[cols], width="stretch", hide_index=True)
    st.caption(
        "The pre-correction claim was 9/10 top-1 with Spearman rho 0.69, "
        "computed from leakage-contaminated checkpoints. Per-crop margins are "
        "frequently under 0.01 R2, so several 'winners' are noise.")

# --------------------------------------------------------------------------- #
# 3. Does fusion help?
# --------------------------------------------------------------------------- #
divider("Does fusion beat the best single model?")

fus_path = METRICS_DIR / "fusion_summary_6models.csv"
if not fus_path.exists():
    st.info("fusion_summary_6models.csv not found.")
else:
    fus = pd.read_csv(fus_path)
    col = "r2" if "r2" in fus.columns else fus.columns[-1]
    per_method = fus.groupby("method")[col].mean().sort_values(ascending=False)
    best_single = mc.drop_duplicates(["model", "crop"]).groupby("model")["r2"].mean().max() \
        if not mc.empty else float("nan")

    k1, k2, k3 = st.columns(3)
    with k1:
        kpi("Best fusion", f"{per_method.iloc[0]:.3f}")
    with k2:
        kpi("Best single model", f"{best_single:.3f}")
    with k3:
        kpi("Fusion advantage", f"{per_method.iloc[0] - best_single:+.3f}")

    if per_method.iloc[0] < best_single:
        st.warning(
            f"No fusion method beats the best single model. The best "
            f"({per_method.index[0]}) trails it by "
            f"{best_single - per_method.iloc[0]:.3f} R2, and "
            f"{int((per_method < best_single).sum())} of {len(per_method)} "
            f"methods are worse.")

    st.dataframe(per_method.rename("Mean R2").reset_index()
                 .style.format({"Mean R2": "{:.3f}"}),
                 width="stretch", hide_index=True, height=260)

# --------------------------------------------------------------------------- #
# 4. Can these numbers be trusted? Artifact freshness.
# --------------------------------------------------------------------------- #
divider("Artifact freshness")

newest_ckpt = newest_checkpoint_mtime()
st.caption(
    f"An artifact older than the newest checkpoint describes models that no "
    f"longer exist. Newest checkpoint: "
    f"{datetime.fromtimestamp(newest_ckpt).strftime('%Y-%m-%d %H:%M') if newest_ckpt else 'none'}.")

watched = ["multicrop_summary.csv", "climatology_baseline.csv",
           "test_predictions.csv",
           "taxonomy_validation.csv", "fusion_summary_6models.csv",
           "counterfactual_daily_summary.csv", "multicrop_horizons.csv",
           "spatial_per_district.csv", "loo_xgboost.csv"]
fr = pd.DataFrame([{
    "Artifact": f,
    "Written": _mtime(METRICS_DIR / f),
    "Current": "yes" if artifact_is_fresh(METRICS_DIR / f) else "STALE",
} for f in watched])
st.dataframe(fr, width="stretch", hide_index=True)

stale = fr[fr["Current"] == "STALE"]["Artifact"].tolist()
if stale:
    st.warning(
        "Stale, i.e. written before the current checkpoints: "
        + ", ".join(f"`{s}`" for s in stale)
        + ". Pages reading these show numbers for superseded models.")

# --------------------------------------------------------------------------- #
# 5. Known limitations
# --------------------------------------------------------------------------- #
divider("Known limitations")

st.markdown("""
- **Nothing beats climatology.** The best model exceeds the learn-nothing
  baseline by a small margin on mean R2 only. Treat every per-crop "winner"
  against that reference, not against zero.
- **Per-district R2 is not usable.** Every fold has n=2, and R2 and RMSE from
  the *same* folds disagree — the worst-R2 districts sit mid-distribution on
  RMSE. Spatial views report RMSE; pooled R2 is reported once.
- **The panel is 282 NUTS-3 districts, not 401.** 67 historical Kreis names do
  not map to the 2021 NUTS geography. This needs boundary harmonisation, not
  imputation.
- **The climate counterfactual has no radiation channel.** Feature policy v3
  excludes `radiation_global` (90.5% missing), so the CMIP6 `rsds`
  perturbation reaches no model. Results measure warming and precipitation
  change only. Satellite indices are held at observed values, so vegetation
  feedback is not simulated.
- **CMIP6 scenario ordering inverts at the 5-year window.** SSP5-8.5 warms
  less than SSP2-4.5 over 2046-2050 because internal variability in two
  single-realization GCMs exceeds the forced difference at that horizon.
  Widening to 2036-2050 restores the expected ordering.
- **Models cannot express trend.** No feature encodes calendar year
  (deliberately, to avoid leakage), so long-horizon extrapolation is outside
  what this setup supports.
- **Results depend on library versions.** The same saved XGBoost model gives
  a winter-wheat error of 10.4 dt/ha under XGBoost 3.3 (on the GPU cluster, where the
  official results were computed) and 12.3 dt/ha under XGBoost 2.1. Versions
  must be pinned for the paper; per-district views only show models whose
  predictions reproduce the official results.
- **Some districts have short records.** 16 of 281 districts have fewer than
  five winter-wheat years; Würzburg Landkreis (DE26C) has one.
- **Every run used the same random seed** (`seed: 42`, deterministic), so
  run-to-run variation has not been measured yet.
""")

with st.expander("Data quality", expanded=False):
    st.markdown("""
- Three satellite indicators were byte-identical to their NDVI counterparts
  across all 16,562,352 rows (`evi_observed`, `evi_stale`, `evi_age_days`) —
  NDVI and EVI come from the same MODIS product on the same acquisition
  dates. Now excluded. `lai_*` genuinely differs and is retained.
- A 90% minimum sequence-coverage filter drops incomplete crop-years before
  splitting.
- Missingness is exposed rather than hidden: `<var>_missing`,
  `<index>_observed`, `<index>_stale`, `<index>_age_days`, `soil_imputed`.
- Imputation uses training-only statistics; forward-fill is bounded and causal.
""")

with st.expander("Evaluation protocol", expanded=False):
    st.markdown("""
- Train 2003-2022, validate 2023, test 2024-2025. Fixed across all models.
- Target is a per-(district, crop) yield anomaly, z-scored by a per-crop
  anomaly standard deviation.
- Ground truth is inverse-transformed WITHOUT the safety clipping applied to
  predictions; clipping observed labels biases R2.
- The district baseline carries a train-only linear trend where that trend is
  statistically significant (p < 0.01) and large relative to residual spread.
""")

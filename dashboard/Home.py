"""Home page — project overview and model leaderboard.

Streamlit looks for a top-level script (this file) as the landing page. All
other pages live in ``pages/`` and appear in the sidebar in numeric order.
"""
from __future__ import annotations

# Pin OpenMP / MKL to one thread *before* any library that ships its own
# libomp.dylib is imported. Without this the dashboard segfaults on macOS
# Intel when two such libraries (torch, xgboost, lightning) end up loaded
# in the same Python process.
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import plotly.express as px
import streamlit as st

from utils import (MODEL_INFO, MODELS, divider, kpi, load_all_metrics,
                   load_multicrop_summary, load_taxonomy,
                   load_taxonomy_validation, load_unified, page_header,
                   taxonomy_verdict)

st.set_page_config(
    page_title="Crop Yield Benchmark — JMU Würzburg",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.markdown(
    "# Transformer-based Crop Yield Prediction\n"
    "**Praktikum — Julius-Maximilians-Universität Würzburg · Institut für Informatik**"
)
st.caption(
    "Head-to-head benchmark of six model families on per-Landkreis "
    "(NUTS-3) German crop yield, 2003–2023."
)

# --------------------------------------------------------------------------- #
# KPI strip
# --------------------------------------------------------------------------- #
df = load_unified()
metrics = load_all_metrics()

tax = load_taxonomy()
tax_val = load_taxonomy_validation()

c1, c2, c3, c4, c5, c6 = st.columns(6)
with c1:
    kpi("Landkreise (NUTS-3)", f"{df['landkreis'].nunique() if not df.empty else 0}")
with c2:
    yr_range = (
        f"{int(df['year'].min())}–{int(df['year'].max())}"
        if not df.empty else "—"
    )
    kpi("Years covered", yr_range)
with c3:
    kpi("Crops", f"{df['crop'].nunique() if not df.empty else 0}")
with c4:
    kpi("Unified rows", f"{len(df):,}" if not df.empty else "—")
with c5:
    kpi("Models trained", f"{metrics['model'].nunique() if not metrics.empty else 0} / 6")
with c6:
    if not tax_val.empty:
        n = len(tax_val)
        hit1 = int(tax_val["hit_at_1"].sum())
        kpi("Taxonomy top-1",
            f"{hit1}/{n} ({hit1/n:.0%})",
            delta=f"top-3: {int(tax_val['hit_at_3'].sum())}/{n}")
    else:
        kpi("Taxonomy top-1", "—")

st.markdown("---")

# --------------------------------------------------------------------------- #
# Leaderboard
# --------------------------------------------------------------------------- #
page_header("Model leaderboard",
            "Best R² per model on the 2022–2023 temporal test split.")

if metrics.empty:
    st.warning(
        "No trained models found yet. Run a model from the **Train** page, "
        "or from the CLI: `python scripts/03_train.py --model <name>`."
    )
else:
    by_model = (metrics.sort_values("r2", ascending=False)
                       .groupby("model", as_index=False)
                       .first()
                       .sort_values("r2", ascending=False))
    by_model["best_horizon"] = by_model["horizon"]
    by_model["family"] = by_model["model"].map(
        lambda m: MODEL_INFO.get(m.replace("_abs", ""), {}).get("kind", "—"))
    by_model["reference"] = by_model["model"].map(
        lambda m: MODEL_INFO.get(m.replace("_abs", ""), {}).get("ref", "—"))
    show = by_model[["model", "family", "best_horizon", "r2", "rmse", "mae",
                     "mape", "reference"]].rename(columns={
        "model": "Model", "family": "Family", "best_horizon": "Best @",
        "r2": "R²", "rmse": "RMSE (dt/ha)", "mae": "MAE",
        "mape": "MAPE (%)", "reference": "Reference",
    })
    st.dataframe(
        show.style.format({"R²": "{:.3f}", "RMSE (dt/ha)": "{:.2f}",
                           "MAE": "{:.2f}", "MAPE (%)": "{:.2f}"})
                  .background_gradient(subset=["R²"], cmap="RdYlGn"),
        width="stretch", hide_index=True,
    )

    # Quick visual: R² per model per horizon
    divider("R² across in-season horizons")
    metrics_main = metrics[~metrics["model"].str.contains("_abs")]
    fig = px.line(
        metrics_main.sort_values("h_days"),
        x="h_days", y="r2", color="model", markers=True,
        labels={"h_days": "Days pre-harvest", "r2": "R²", "model": "Model"},
        title="R² vs. in-season horizon — winter wheat, 2022–2023 test",
    )
    fig.add_hline(y=0.64, line_dash="dash", line_color="grey",
                  annotation_text="per-district mean baseline (0.64)",
                  annotation_position="bottom right")
    fig.update_layout(height=440, hovermode="x unified")
    st.plotly_chart(fig, width="stretch")

# --------------------------------------------------------------------------- #
# Per-crop leaderboard (multi-crop benchmark)
# --------------------------------------------------------------------------- #
mc = load_multicrop_summary()
if not mc.empty:
    divider("Per-crop leaderboard with taxonomy overlay")
    st.caption(
        "R² per (model, crop) on the 2022–2023 test split. Each cell tagged "
        "with **[P]** if the taxonomy predicted this model as PRIMARY for "
        "this crop, **[A]** if alternate. If the observed best model (green "
        "text) matches taxonomy [P] — that's a top-1 hit."
    )

    # ---- Matrix view: crops × models with R² gradient ----
    model_order = (mc.groupby("model")["r2"].mean()
                     .sort_values(ascending=False).index.tolist())
    crop_order = (mc.groupby("crop")["r2"].mean()
                    .sort_values(ascending=False).index.tolist())
    pivot = (mc.pivot(index="crop", columns="model", values="r2")
               .reindex(index=crop_order, columns=model_order)
               .round(3))
    pivot["mean_across_models"] = pivot.mean(axis=1).round(3)

    # Build a parallel matrix of "P"/"A"/"" markers per (crop, model)
    verdict_pivot = pd.DataFrame(
        index=pivot.index, columns=[c for c in pivot.columns if c != "mean_across_models"]
    )
    for crop in verdict_pivot.index:
        for model in verdict_pivot.columns:
            v = taxonomy_verdict(crop, model, tax)
            verdict_pivot.at[crop, model] = "P" if v == "primary" else ("A" if v == "alternate" else "")

    def _fmt_cell(x):
        return f"{x:.3f}" if pd.notna(x) else "—"

    # Compose a display frame: R² formatted + optional [P]/[A] suffix
    disp = pivot.copy()
    for crop in pivot.index:
        for model in verdict_pivot.columns:
            r2v = pivot.at[crop, model]
            tag = verdict_pivot.at[crop, model]
            if pd.notna(r2v):
                if tag == "P":
                    disp.at[crop, model] = f"{r2v:.3f} [P]"
                elif tag == "A":
                    disp.at[crop, model] = f"{r2v:.3f} [A]"
                else:
                    disp.at[crop, model] = f"{r2v:.3f}"
    disp["mean_across_models"] = pivot["mean_across_models"].map(_fmt_cell)

    st.dataframe(disp, width="stretch")

    # Companion: pure-numeric view with R² gradient (for scanning by color)
    st.markdown("##### Same data, colored by R² (no tags — easier to scan)")
    st.dataframe(
        pivot.style.format("{:.3f}", na_rep="—")
                   .background_gradient(cmap="RdYlGn", vmin=-0.2, vmax=0.7,
                                        subset=model_order),
        width="stretch",
    )

    # ---- New: Compatibility heatmap by family ----
    if tax:
        divider("Compatibility heatmap — crop-phenology family × model architecture family")
        st.caption(
            "Aggregates our per-(crop, model) R² by crop phenology family "
            "(winter cereal, root crop, …) and model architecture family "
            "(tree, transformer, recurrent, …). The clusters that light up "
            "green are the theoretically predicted compatibilities — this "
            "visualizes the taxonomy's core claim in one shot."
        )
        crop_fam = {c: v.get("phenology_family", "unknown")
                    for c, v in tax.get("crops", {}).items()}
        model_fam = {m: v.get("architecture_family", "unknown")
                     for m, v in tax.get("models", {}).items()}
        mc2 = mc.copy()
        mc2["crop_family"] = mc2["crop"].map(crop_fam)
        mc2["model_family"] = mc2["model"].map(model_fam)
        fam_pivot = (mc2.groupby(["crop_family", "model_family"])["r2"]
                       .mean().unstack())
        fig = px.imshow(
            fam_pivot, color_continuous_scale="RdYlGn",
            color_continuous_midpoint=0, aspect="auto", text_auto=".3f",
            labels=dict(x="Model architecture family",
                        y="Crop phenology family", color="Mean R²"),
            title="Mean test R² per (crop family × model family)",
        )
        fig.update_layout(height=420)
        st.plotly_chart(fig, width="stretch")

    # ---- Single-crop drilldown: pick a crop, rank models ----
    st.markdown("##### Pick a crop to see the model ranking")
    sel_crop = st.selectbox(
        "Crop", crop_order,
        index=crop_order.index("winter_wheat") if "winter_wheat" in crop_order else 0,
    )

    # Taxonomy prediction banner for this crop
    if tax:
        pred = tax.get("predicted_winners", {}).get(sel_crop, {})
        if pred:
            primary = pred.get("primary", "?")
            alts = pred.get("alternates", []) or []
            rationale = pred.get("rationale", "")
            st.info(
                f"**Taxonomy prediction for {sel_crop}:** "
                f"primary → `{primary}`, alternates → "
                f"{', '.join('`' + a + '`' for a in alts) if alts else '—'}. "
                f"\n\n{rationale}"
            )

    cw = (mc[mc["crop"] == sel_crop]
          .sort_values("r2", ascending=False)
          .reset_index(drop=True))
    cw["rank"] = cw.index + 1
    cw["family"] = cw["model"].map(
        lambda m: MODEL_INFO.get(m, {}).get("kind", "—"))
    cw["taxonomy"] = cw["model"].map(
        lambda m: {"primary": "PRIMARY", "alternate": "alt", "": "—"}[
            taxonomy_verdict(sel_crop, m, tax)])
    show = cw[["rank", "model", "family", "taxonomy", "r2", "rmse", "mae", "n"]].rename(
        columns={"rank": "Rank", "model": "Model", "family": "Family",
                 "taxonomy": "Taxonomy pick",
                 "r2": "R²", "rmse": "RMSE (dt/ha)", "mae": "MAE", "n": "n"})
    st.dataframe(
        show.style.format({"R²": "{:.3f}", "RMSE (dt/ha)": "{:.2f}",
                           "MAE": "{:.2f}"})
                   .background_gradient(subset=["R²"], cmap="RdYlGn"),
        width="stretch", hide_index=True,
    )

# --------------------------------------------------------------------------- #
# Project context
# --------------------------------------------------------------------------- #
divider("About the project")
st.markdown("""
This dashboard wraps the reproducible benchmark described in
`Crop_Yield_Transformer_Preliminary.pptx`. The research question is:

> *On a single standardized German crop-yield dataset, which transformer
> family delivers the best accuracy–interpretability trade-off for
> **in-season** prediction (30 / 60 / 90 / 120 days before harvest)?*

**Pages**

| Page | Purpose |
|---|---|
| **Data** | Source provenance for weather, satellite, soil, yield, geography; live previews. |
| **Train** | Launch any of the six training scripts in the background; tail logs live. |
| **Models** (XGBoost / LSTM / DLinear / PatchTST / iTransformer / TFT) | One page per model — config, predictions vs. truth, residuals, feature lists. |
| **Geographic** | Choropleth map of yield / predictions across the ~400 German Landkreise. |

**Why anomaly framing?** 97 % of yield variance is between-district. Models
predict the (district, year) **anomaly** rather than the absolute yield, so
R² measures the part of the signal models actually contribute — not just
"knowing the district".
""")

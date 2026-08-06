"""Head-to-head leaderboard — Chapter 4 landing page.

Six architectures across ten crops: best-per-model table, per-crop-by-model
matrix with taxonomy [P]/[A] overlay, crop-family x model-family
compatibility heatmap, and a single-crop drilldown. Lifted off Home so this
stays the canonical Chapter 4 entry point.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import (MODEL_INFO, divider, load_all_metrics,
                   load_multicrop_summary, load_taxonomy, page_header,
                   taxonomy_verdict)

st.set_page_config(page_title="Leaderboard — Crop Yield Benchmark",
                   layout="wide")

page_header("Head-to-head Leaderboard",
            "Best R² per model, per-crop matrix with taxonomy overlay, "
            "family compatibility heatmap, and single-crop drilldown.")

metrics = load_all_metrics()
tax = load_taxonomy()

# --------------------------------------------------------------------------- #
# Best-per-model leaderboard
# --------------------------------------------------------------------------- #
divider("Best-per-model leaderboard")
st.caption(
    "Best R² per model on the 2024-2025 temporal test split — the single "
    "number the six-way head-to-head collapses to."
)

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
        title="R² vs. in-season horizon — winter wheat, 2024-2025 test",
    )
    fig.update_layout(height=440, hovermode="x unified")
    st.plotly_chart(fig, width="stretch")

# --------------------------------------------------------------------------- #
# Per-crop leaderboard (multi-crop benchmark)
# --------------------------------------------------------------------------- #
mc = load_multicrop_summary()
if not mc.empty:
    divider("Per-crop leaderboard with taxonomy overlay")
    st.caption(
        "R² per (model, crop) on the 2024-2025 test split. Each cell tagged "
        "with **[P]** if the taxonomy predicted this model as PRIMARY for "
        "this crop, **[A]** if alternate. If the observed best model (green "
        "text) matches taxonomy [P] — that's a top-1 hit."
    )

    model_order = (mc.groupby("model")["r2"].mean()
                     .sort_values(ascending=False).index.tolist())
    crop_order = (mc.groupby("crop")["r2"].mean()
                    .sort_values(ascending=False).index.tolist())
    pivot = (mc.pivot(index="crop", columns="model", values="r2")
               .reindex(index=crop_order, columns=model_order)
               .round(3))
    pivot["mean_across_models"] = pivot.mean(axis=1).round(3)

    verdict_pivot = pd.DataFrame(
        index=pivot.index,
        columns=[c for c in pivot.columns if c != "mean_across_models"],
    )
    for crop in verdict_pivot.index:
        for model in verdict_pivot.columns:
            v = taxonomy_verdict(crop, model, tax)
            verdict_pivot.at[crop, model] = "P" if v == "primary" else (
                "A" if v == "alternate" else "")

    def _fmt_cell(x):
        return f"{x:.3f}" if pd.notna(x) else "—"

    disp = pivot.copy().astype(object)
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

    st.markdown("##### Same data, colored by R² (no tags — easier to scan)")
    st.dataframe(
        pivot.style.format("{:.3f}", na_rep="—")
                   .background_gradient(cmap="RdYlGn", vmin=-0.2, vmax=0.7,
                                        subset=model_order),
        width="stretch",
    )

    # ---- Compatibility heatmap by family ----
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

    # ---- Single-crop drilldown ----
    st.markdown("##### Pick a crop to see the model ranking")
    sel_crop = st.selectbox(
        "Crop", crop_order,
        index=crop_order.index("winter_wheat") if "winter_wheat" in crop_order else 0,
    )

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

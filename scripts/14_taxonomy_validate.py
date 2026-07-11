"""Validate the crop-model compatibility taxonomy against empirical results.

Compares the a priori predicted winner per crop (from
``configs/taxonomy.yaml``) with the observed best-single-model per crop
(from ``results/metrics/multicrop_summary.csv``). Reports how many crops
the taxonomy called correctly (top-1 hit) and how many the observed
winner appears in the top-3 taxonomy list.

Outputs:
    results/metrics/taxonomy_validation.csv
        columns: crop, predicted_primary, predicted_alternates,
                 observed_best_model, observed_best_r2,
                 predicted_primary_r2, hit_at_1, hit_at_3, rationale

    results/metrics/taxonomy_rank_correlation.csv
        columns: crop, model, predicted_rank, observed_rank, r2
        Long form for per-crop rank correlation analysis.

Usage:
    python scripts/14_taxonomy_validate.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
TAXONOMY_PATH = ROOT / "configs" / "taxonomy.yaml"
MULTICROP_PATH = ROOT / "results" / "metrics" / "multicrop_summary.csv"
OUT_SUMMARY = ROOT / "results" / "metrics" / "taxonomy_validation.csv"
OUT_RANKS = ROOT / "results" / "metrics" / "taxonomy_rank_correlation.csv"


def rank_models_from_taxonomy(crop: str, tax: dict) -> list[str]:
    """Return an ordered list of predicted winners for a crop:
    primary first, then alternates in listed order, then everything else."""
    pred = tax["predicted_winners"].get(crop, {})
    primary = pred.get("primary")
    alts = pred.get("alternates", []) or []
    all_models = list(tax["models"].keys())
    ordered = []
    if primary:
        ordered.append(primary)
    ordered.extend(m for m in alts if m not in ordered)
    ordered.extend(m for m in all_models if m not in ordered)
    return ordered


def main() -> None:
    tax = yaml.safe_load(TAXONOMY_PATH.read_text())
    scores = pd.read_csv(MULTICROP_PATH)
    crops = list(tax["predicted_winners"].keys())
    all_models = list(tax["models"].keys())

    summary_rows = []
    rank_rows = []
    for crop in crops:
        pred = tax["predicted_winners"].get(crop, {})
        primary = pred.get("primary", "")
        alts = pred.get("alternates", []) or []
        top3 = [primary] + list(alts)
        top3 = top3[:3]

        crop_scores = scores[scores["crop"] == crop].sort_values("r2", ascending=False)
        if crop_scores.empty:
            continue
        observed_best_model = crop_scores.iloc[0]["model"]
        observed_best_r2 = float(crop_scores.iloc[0]["r2"])
        primary_score_row = crop_scores[crop_scores["model"] == primary]
        primary_r2 = float(primary_score_row["r2"].iloc[0]) if not primary_score_row.empty else float("nan")

        hit_1 = int(observed_best_model == primary)
        hit_3 = int(observed_best_model in top3)

        summary_rows.append({
            "crop": crop,
            "predicted_primary": primary,
            "predicted_alternates": ",".join(alts),
            "observed_best_model": observed_best_model,
            "observed_best_r2": round(observed_best_r2, 3),
            "predicted_primary_r2": round(primary_r2, 3) if primary_r2 == primary_r2 else None,
            "primary_gap_vs_observed_r2": round(observed_best_r2 - primary_r2, 3) if primary_r2 == primary_r2 else None,
            "hit_at_1": hit_1,
            "hit_at_3": hit_3,
            "rationale": pred.get("rationale", ""),
        })

        # Rank-correlation input: predicted rank (from taxonomy) vs observed rank
        predicted_order = rank_models_from_taxonomy(crop, tax)
        model_to_pred_rank = {m: i + 1 for i, m in enumerate(predicted_order)}
        observed_order = list(crop_scores["model"])
        model_to_obs_rank = {m: i + 1 for i, m in enumerate(observed_order)}
        model_to_r2 = dict(zip(crop_scores["model"], crop_scores["r2"]))
        for m in all_models:
            if m not in model_to_r2:
                continue
            rank_rows.append({
                "crop": crop,
                "model": m,
                "predicted_rank": model_to_pred_rank.get(m, len(all_models)),
                "observed_rank": model_to_obs_rank.get(m, len(all_models)),
                "r2": round(float(model_to_r2[m]), 3),
            })

    out_summary = pd.DataFrame(summary_rows)
    OUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    out_summary.to_csv(OUT_SUMMARY, index=False)
    out_ranks = pd.DataFrame(rank_rows)
    out_ranks.to_csv(OUT_RANKS, index=False)

    total = len(out_summary)
    hit1 = int(out_summary["hit_at_1"].sum())
    hit3 = int(out_summary["hit_at_3"].sum())
    print("=== Taxonomy validation summary ===")
    print(out_summary[["crop", "predicted_primary", "observed_best_model",
                        "observed_best_r2", "predicted_primary_r2",
                        "primary_gap_vs_observed_r2", "hit_at_1", "hit_at_3"]]
          .to_string(index=False))
    print()
    print(f"top-1 hit rate: {hit1}/{total} = {hit1/total:.1%}")
    print(f"top-3 hit rate: {hit3}/{total} = {hit3/total:.1%}")

    print("\n=== Per-crop Spearman rank correlation (predicted vs observed) ===")
    corrs = []
    for crop, g in out_ranks.groupby("crop"):
        rho, p = spearmanr(g["predicted_rank"], g["observed_rank"])
        corrs.append({"crop": crop, "rho": round(rho, 3), "p": round(p, 3),
                      "n_models": int(len(g))})
    print(pd.DataFrame(corrs).sort_values("rho", ascending=False).to_string(index=False))

    mean_rho = sum(c["rho"] for c in corrs) / len(corrs)
    print(f"\nmean Spearman ρ across crops: {mean_rho:.3f}")
    print(f"(positive ρ = taxonomy's predicted ranking correlates with observed)")


if __name__ == "__main__":
    main()

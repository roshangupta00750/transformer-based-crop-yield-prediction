"""Supplementary multi-criteria model recommendation per crop.

Combines four normalized component scores in [0, 1] per (crop, model)
into a weighted composite, then ranks models within each crop:

    composite = w_acc     * accuracy
              + w_climate * climate_stability
              + w_interp  * interpretability
              + w_compute * compute_cost

Defaults: w_acc=0.5, w_climate=0.2, w_interp=0.15, w_compute=0.15.

This is SUPPLEMENTARY analysis — it is not the paper's headline claim.
The headline remains per-crop accuracy validated against the a priori
taxonomy (see ``scripts/14_taxonomy_validate.py``). Use this composite
only as a "we also explored" pointer for readers who want to see how
non-accuracy criteria (climate robustness, interpretability, training
cost) would shift the picture if they carried real weight.

Reads:
    results/metrics/multicrop_summary.csv
    results/metrics/counterfactual_daily_summary.csv
    configs/taxonomy.yaml

Writes:
    results/metrics/composite_ranking.csv
        columns: crop, model, acc_score, climate_score, interp_score,
                 compute_score, composite_score, rank, best_model_for_crop
"""
from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
CONFIGS_DIR = PROJECT_ROOT / "configs"

DEFAULT_MULTICROP = METRICS_DIR / "multicrop_summary.csv"
DEFAULT_COUNTERFACTUAL = METRICS_DIR / "counterfactual_daily_summary.csv"
DEFAULT_TAXONOMY = CONFIGS_DIR / "taxonomy.yaml"
DEFAULT_OUT = METRICS_DIR / "composite_ranking.csv"


# --------------------------------------------------------------------------- #
# 1. Accuracy component
# --------------------------------------------------------------------------- #
def accuracy_score(summary_csv: str | Path) -> pd.DataFrame:
    """Per-crop min-max normalization of R² with negative-R² clipping.

    Returns long-form DataFrame with columns [crop, model, acc_score].
    Within each crop: 1.0 = best R², 0.0 = worst (after clipping <0 to 0).
    Degenerate ties award 1.0 to everyone (contributes neutrally).
    """
    df = pd.read_csv(summary_csv)[["crop", "model", "r2"]].copy()
    df["r2_clip"] = df["r2"].clip(lower=0.0)

    # Per-crop min-max on the clipped column. Using .transform keeps the
    # frame aligned and avoids the pandas 2.x apply/group-columns warning.
    lo = df.groupby("crop")["r2_clip"].transform("min")
    hi = df.groupby("crop")["r2_clip"].transform("max")
    span = hi - lo
    # Where hi == lo (all models tied — e.g. every one clipped to 0) we
    # award 1.0 so the component contributes neutrally rather than zero
    # out an entire crop column of the composite.
    df["acc_score"] = np.where(span > 0, (df["r2_clip"] - lo) / span, 1.0)
    return df[["crop", "model", "acc_score"]].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 2. Climate-stability component
# --------------------------------------------------------------------------- #
SIGNAL_THRESHOLD_PCT = 0.05   # ensemble mean must exceed this to trigger sign-flip penalty
DAMPING_RATIO = 0.10          # |delta| below this * median_mag = "ignoring signal"
FLIP_RAW = 0.30
OVER_RAW = 0.50
DAMPED_RAW = 0.50
FULL_RAW = 1.00


def _per_scenario_raw(model_delta: float,
                      ensemble_deltas: pd.Series) -> float:
    """Score one (model, crop, scenario) cell against its 6-model peer ensemble."""
    ensemble_mean = ensemble_deltas.mean()
    mags = ensemble_deltas.abs()
    median_mag = mags.median()
    mag_std = float(np.std(mags, ddof=0))
    mag_std = max(mag_std, 1e-6)  # guard degenerate clusters

    model_mag = abs(model_delta)

    # 1. Damping guard — TFT-style ~0 deltas land here, not in the flip branch.
    if median_mag > 0 and model_mag < DAMPING_RATIO * median_mag:
        return DAMPED_RAW

    # 2. Sign flip (only when ensemble has a clear directional signal).
    if abs(ensemble_mean) >= SIGNAL_THRESHOLD_PCT:
        if np.sign(model_delta) != np.sign(ensemble_mean) and model_delta != 0:
            return FLIP_RAW

    # 3. Magnitude sanity vs ensemble median magnitude, in ensemble-std units.
    dev = abs(model_mag - median_mag)
    if dev <= mag_std:
        return FULL_RAW
    if dev >= 3 * mag_std:
        return OVER_RAW
    return FULL_RAW - (FULL_RAW - OVER_RAW) * (dev - mag_std) / (2 * mag_std)


def climate_stability_score(counterfactual_csv: str | Path) -> pd.DataFrame:
    """Score how physically defensible each model's CMIP6 response is.

    Returns DataFrame with columns [crop, model, climate_score] in [0.3, 1.0].
    Per-scenario raw scores are averaged across the two SSP scenarios.
    """
    cf = pd.read_csv(counterfactual_csv)[
        ["model", "crop", "scenario", "delta_pct"]
    ].copy()

    rows = []
    for (crop, scenario), g in cf.groupby(["crop", "scenario"]):
        ensemble_deltas = g.set_index("model")["delta_pct"].astype(float)
        for model, delta in ensemble_deltas.items():
            raw = _per_scenario_raw(delta, ensemble_deltas)
            rows.append({"crop": crop, "model": model,
                         "scenario": scenario, "raw": raw})

    per_scen = pd.DataFrame(rows)
    return (per_scen.groupby(["crop", "model"])["raw"]
                    .mean()
                    .rename("climate_score")
                    .reset_index())


# --------------------------------------------------------------------------- #
# 3. Interpretability component (per-model constant, broadcast to crops)
# --------------------------------------------------------------------------- #
INTERPRETABILITY_BY_MODEL: dict[str, float] = {
    "dlinear":      1.00,  # raw coefficients ARE the explanation
    "xgboost":      0.90,  # SHAP + gain importances first-class
    "tft":          0.85,  # variable-selection + interpretable attention by design
    "itransformer": 0.65,  # cross-variable attention weights, needs aggregation
    "patchtst":     0.60,  # patch-level attention, coarser than TFT
    "lstm":         0.30,  # opaque gated hidden state; only post-hoc methods
}


def interpretability_score(taxonomy_path: str | Path) -> pd.DataFrame:
    """Return [crop, model, interp_score] — crop-invariant expert scalar."""
    with open(taxonomy_path) as f:
        taxonomy = yaml.safe_load(f)
    crops = list(taxonomy["crops"].keys())
    models = list(INTERPRETABILITY_BY_MODEL.keys())
    rows = [
        {"crop": c, "model": m, "interp_score": INTERPRETABILITY_BY_MODEL[m]}
        for c, m in product(crops, models)
    ]
    df = pd.DataFrame(rows)
    df["interp_score"] = df["interp_score"].clip(0.0, 1.0)
    return df


# --------------------------------------------------------------------------- #
# 4. Compute-cost component (per-model constant, broadcast to crops)
# --------------------------------------------------------------------------- #
# Hard-coded training minutes from prior sessions. See docstring caveats.
TRAINING_MINUTES: dict[str, float] = {
    "xgboost":      1,     # CPU
    "dlinear":      8,     # RTX 3090
    "itransformer": 8,     # RTX 3090
    "lstm":         9,     # RTX 3090
    "patchtst":     11,    # RTX 3090
    "tft":          200,   # RTX 3090 / H100, memory-bound (range 55-360 min)
}


def compute_cost_score(crops: list[str]) -> pd.DataFrame:
    """Per-model score = 1 / (1 + minutes/max_minutes), broadcast to crops.

    Bounded to (0.5, 1.0]. Fastest -> ~1.0; slowest -> exactly 0.5.
    """
    max_minutes = max(TRAINING_MINUTES.values())
    per_model = {
        m: 1.0 / (1.0 + mins / max_minutes)
        for m, mins in TRAINING_MINUTES.items()
    }
    rows = [
        {"crop": c, "model": m, "compute_score": s}
        for m, s in per_model.items() for c in crops
    ]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Composite assembly
# --------------------------------------------------------------------------- #
def build_composite(multicrop_csv: str | Path,
                    counterfactual_csv: str | Path,
                    taxonomy_path: str | Path,
                    w_acc: float, w_climate: float,
                    w_interp: float, w_compute: float) -> pd.DataFrame:
    """Assemble the four components into a per-(crop, model) composite table."""
    acc = accuracy_score(multicrop_csv)
    climate = climate_stability_score(counterfactual_csv)
    interp = interpretability_score(taxonomy_path)

    # Use the crop list from the taxonomy for the compute broadcast so we
    # cover all 10 crops even if a crop happens to be missing from the
    # accuracy or counterfactual CSV.
    with open(taxonomy_path) as f:
        crops = list(yaml.safe_load(f)["crops"].keys())
    compute = compute_cost_score(crops)

    df = (interp
          .merge(compute, on=["crop", "model"], how="outer")
          .merge(acc, on=["crop", "model"], how="left")
          .merge(climate, on=["crop", "model"], how="left"))

    # If a (crop, model) is missing accuracy or climate data, treat as 0.
    df["acc_score"] = df["acc_score"].fillna(0.0)
    df["climate_score"] = df["climate_score"].fillna(0.0)

    df["composite_score"] = (
        w_acc * df["acc_score"]
        + w_climate * df["climate_score"]
        + w_interp * df["interp_score"]
        + w_compute * df["compute_score"]
    )

    # Per-crop rank (1 = best composite).
    df["rank"] = (df.groupby("crop")["composite_score"]
                    .rank(ascending=False, method="min")
                    .astype(int))

    best = (df.sort_values(["crop", "composite_score"], ascending=[True, False])
              .drop_duplicates("crop")[["crop", "model"]]
              .rename(columns={"model": "best_model_for_crop"}))
    df = df.merge(best, on="crop", how="left")

    return df.sort_values(["crop", "rank"]).reset_index(drop=True)


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per crop: winner, composite, top-2 contributing components."""
    comps = ["acc_score", "climate_score", "interp_score", "compute_score"]
    weights = {"acc_score": "w_acc", "climate_score": "w_climate",
               "interp_score": "w_interp", "compute_score": "w_compute"}
    rows = []
    for crop, g in df.groupby("crop"):
        winner = g.sort_values("composite_score", ascending=False).iloc[0]
        # Rank the four component scores for the winner and pick top 2.
        comp_vals = {c: float(winner[c]) for c in comps}
        top2 = sorted(comp_vals.items(), key=lambda kv: kv[1], reverse=True)[:2]
        rows.append({
            "crop": crop,
            "winner": winner["model"],
            "composite": round(float(winner["composite_score"]), 3),
            "top_component_1": f"{top2[0][0].replace('_score','')}={top2[0][1]:.2f}",
            "top_component_2": f"{top2[1][0].replace('_score','')}={top2[1][1]:.2f}",
        })
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--w-acc", type=float, default=0.50, help="Accuracy weight")
    p.add_argument("--w-climate", type=float, default=0.20, help="Climate weight")
    p.add_argument("--w-interp", type=float, default=0.15,
                   help="Interpretability weight")
    p.add_argument("--w-compute", type=float, default=0.15,
                   help="Compute-cost weight")
    p.add_argument("--multicrop-csv", type=Path, default=DEFAULT_MULTICROP)
    p.add_argument("--counterfactual-csv", type=Path,
                   default=DEFAULT_COUNTERFACTUAL)
    p.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    total = args.w_acc + args.w_climate + args.w_interp + args.w_compute
    if abs(total - 1.0) > 1e-6:
        print(f"[warn] weights sum to {total:.4f}, not 1.0 — proceeding anyway")

    df = build_composite(
        args.multicrop_csv, args.counterfactual_csv, args.taxonomy,
        args.w_acc, args.w_climate, args.w_interp, args.w_compute,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"[ok] wrote {args.out}  ({len(df)} rows)")

    print("\nPer-crop composite winners (weights: "
          f"acc={args.w_acc}, climate={args.w_climate}, "
          f"interp={args.w_interp}, compute={args.w_compute}):")
    print(summary_table(df).to_string(index=False))


if __name__ == "__main__":
    main()

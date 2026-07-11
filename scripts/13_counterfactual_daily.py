"""Climate counterfactual on daily features via CMIP6 delta method.

Reuses the existing daily-trained models (xgboost/dlinear/itransformer/
lstm/patchtst/tft). Applies CMIP6 (SSP future - historical baseline)
monthly deltas to every day of the observed 2022-23 test set, then runs
each model on the perturbed features.

Delta application rules:
    temperature_air_max_2m  += tasmax_delta       (additive °C — K deltas = °C deltas)
    temperature_air_min_2m  += tasmin_delta       (additive)
    precipitation_height    *= (1 + pr_pct_delta) (multiplicative — unit-invariant)
    radiation_global        *= (1 + rsds_pct_delta) (multiplicative)
    humidity, ndvi, evi, lai, sin_doy, cos_doy    (no CMIP6 counterpart — unchanged)

The multiplicative form for pr/rsds sidesteps the unit mismatch between
DWD (J/cm²/day, mm/day) and CMIP6 (W/m², kg/m²/s) — we're only using
the relative change, not the absolute magnitude.

Outputs
-------
results/metrics/counterfactual_daily_summary.csv
    columns: model, crop, scenario, baseline_mean_pred, counterfactual_mean_pred,
             delta_dt_ha, delta_pct, n_districts_years,
             tasmax_delta_C, pr_pct_delta

results/metrics/counterfactual_daily_district.csv
    long: per (model, crop, scenario, landkreis, year) baseline + counterfactual pred

Usage:
    python scripts/13_counterfactual_daily.py
    python scripts/13_counterfactual_daily.py --models xgboost,itransformer
    python scripts/13_counterfactual_daily.py --future-window 2046 2050
"""
from __future__ import annotations

import argparse
import os
import sys
import types
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dashboard"))

# Stub streamlit so importing the dashboard helper works headless
st = types.ModuleType("streamlit")
st.cache_data = lambda **kw: (lambda f: f)
st.cache_resource = lambda **kw: (lambda f: f)
st.set_page_config = lambda **kw: None
sys.modules["streamlit"] = st

import importlib.util
spec = importlib.util.spec_from_file_location("mt", str(ROOT / "dashboard" / "_model_template.py"))
mt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mt)

from src.utils import load_data_config

CMIP6_PATH = ROOT / "data" / "processed" / "cmip6_nuts3.parquet"
OBS_PATH = ROOT / "data" / "processed" / "dataset.parquet"
OUT_SUMMARY = ROOT / "results" / "metrics" / "counterfactual_daily_summary.csv"
OUT_DISTRICT = ROOT / "results" / "metrics" / "counterfactual_daily_district.csv"

DEFAULT_MODELS = ["xgboost", "dlinear", "itransformer", "lstm", "patchtst", "tft"]
SCENARIOS = ["ssp2_4_5", "ssp5_8_5"]
HIST_BASELINE_WINDOW = (2003, 2014)


def compute_delta_lookups(cmip: pd.DataFrame, scenario: str,
                          future_window: tuple[int, int]
                          ) -> dict[str, pd.DataFrame]:
    """Return per-variable delta tables indexed by (landkreis, month).
    Additive for tas*/multiplicative for pr/rsds (percent form).

    When the input parquet contains multiple CMIP6 models (a ``model``
    column), this groupby collapses them into a per-district monthly
    ensemble mean — model averaging happens implicitly here.
    """
    if "model" in cmip.columns:
        n_models = cmip["model"].nunique()
        print(f"  ensemble over {n_models} CMIP6 model(s): "
              f"{sorted(cmip['model'].unique())}")
    hist = (cmip[(cmip["experiment"] == "historical")
                 & (cmip["year"].between(*HIST_BASELINE_WINDOW))]
            .groupby(["landkreis", "month", "variable"])["value"].mean())
    fut = (cmip[(cmip["experiment"] == scenario)
                & (cmip["year"].between(*future_window))]
           .groupby(["landkreis", "month", "variable"])["value"].mean())

    out: dict[str, pd.DataFrame] = {}
    for v in ["tasmax", "tasmin"]:
        d = (fut - hist).xs(v, level="variable")
        out[v] = d.rename(f"delta_{v}").reset_index()
    for v in ["pr", "rsds"]:
        # percent change guarded against zero-baseline
        base = hist.xs(v, level="variable")
        change = fut.xs(v, level="variable") - base
        pct = np.where(base.to_numpy() > 1e-6, change.to_numpy() / base.to_numpy(), 0.0)
        pct_df = pd.DataFrame({
            "landkreis": base.index.get_level_values("landkreis"),
            "month": base.index.get_level_values("month"),
            f"pct_{v}": pct,
        })
        out[v] = pct_df
    return out


def perturb_observed_daily(obs: pd.DataFrame, deltas: dict[str, pd.DataFrame]
                           ) -> pd.DataFrame:
    """Apply monthly deltas to daily rows. Vectorized merges — fast."""
    df = obs.copy()
    df["month"] = pd.to_datetime(df["date"]).dt.month
    # tasmax → temperature_air_max_2m (additive)
    d = deltas["tasmax"]
    m = df[["landkreis", "month"]].merge(d, on=["landkreis", "month"], how="left")
    df["temperature_air_max_2m"] = df["temperature_air_max_2m"] + m["delta_tasmax"].fillna(0).to_numpy()
    # tasmin → temperature_air_min_2m (additive)
    d = deltas["tasmin"]
    m = df[["landkreis", "month"]].merge(d, on=["landkreis", "month"], how="left")
    df["temperature_air_min_2m"] = df["temperature_air_min_2m"] + m["delta_tasmin"].fillna(0).to_numpy()
    # pr → precipitation_height (multiplicative)
    d = deltas["pr"]
    m = df[["landkreis", "month"]].merge(d, on=["landkreis", "month"], how="left")
    df["precipitation_height"] = df["precipitation_height"] * (1.0 + m["pct_pr"].fillna(0).to_numpy())
    # rsds → radiation_global (multiplicative)
    d = deltas["rsds"]
    m = df[["landkreis", "month"]].merge(d, on=["landkreis", "month"], how="left")
    df["radiation_global"] = df["radiation_global"] * (1.0 + m["pct_rsds"].fillna(0).to_numpy())
    return df.drop(columns=["month"])


def eval_model(model: str, crop: str, df_source: pd.DataFrame) -> pd.DataFrame | None:
    """Run one model on one crop's test-set rows, using ``df_source`` as
    the underlying dataset (baseline vs counterfactual).

    We temporarily monkeypatch mt.load_unified to return ``df_source`` for
    the duration of one prediction call.
    """
    original = mt.load_unified
    mt.load_unified = lambda: df_source
    try:
        pred_df, err = mt._predict_test_set(model, crop)
        if err or pred_df.empty:
            return None
        return pred_df[["landkreis", "year", "y_true", "y_pred"]]
    finally:
        mt.load_unified = original


def summarize(model: str, crop: str, scenario: str,
              base_pred: pd.DataFrame, cf_pred: pd.DataFrame,
              tasmax_delta: float, pr_pct: float) -> tuple[dict, list[dict]]:
    """Aggregate to (model, crop, scenario) row + per-district long rows."""
    both = base_pred.rename(columns={"y_pred": "pred_base"}).merge(
        cf_pred.rename(columns={"y_pred": "pred_cf"}),
        on=["landkreis", "year", "y_true"], how="inner")
    if both.empty:
        return {}, []
    base_mean = float(both["pred_base"].mean())
    cf_mean = float(both["pred_cf"].mean())
    summary = {
        "model": model, "crop": crop, "scenario": scenario,
        "baseline_mean_pred": base_mean,
        "counterfactual_mean_pred": cf_mean,
        "delta_dt_ha": cf_mean - base_mean,
        "delta_pct": (cf_mean - base_mean) / base_mean * 100 if base_mean > 0 else float("nan"),
        "n_districts_years": int(len(both)),
        "tasmax_delta_C": tasmax_delta,
        "pr_pct_delta": pr_pct,
    }
    district = [{
        "model": model, "crop": crop, "scenario": scenario,
        "landkreis": row["landkreis"], "year": int(row["year"]),
        "pred_baseline": float(row["pred_base"]),
        "pred_counterfactual": float(row["pred_cf"]),
        "delta_dt_ha": float(row["pred_cf"] - row["pred_base"]),
    } for _, row in both.iterrows()]
    return summary, district


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS),
                    help="Comma-separated models to run.")
    ap.add_argument("--future-window", nargs=2, type=int, default=[2046, 2050])
    args = ap.parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    fw = (args.future_window[0], args.future_window[1])
    cfg = load_data_config()
    crops = cfg["crops"]

    print(f"Models: {models}")
    print(f"Future window: {fw[0]}-{fw[1]} vs historical {HIST_BASELINE_WINDOW[0]}-{HIST_BASELINE_WINDOW[1]}")
    print(f"Crops: {crops}\n")

    cmip = pd.read_parquet(CMIP6_PATH)
    obs = pd.read_parquet(OBS_PATH)

    # Pre-filter observed to include everything the models need for their
    # phenology window on the test years (models may look back into 2021
    # for winter crops planted in autumn). Cutting to years >= 2020 preserves
    # any encoder look-back while dropping ~14 M rows of ancient data.
    obs_recent = obs[pd.to_datetime(obs["date"]).dt.year >= 2020].copy()
    print(f"Observed (post-filter): {len(obs_recent):,} rows "
          f"(from {len(obs):,})")

    summary_rows: list[dict] = []
    district_rows: list[dict] = []

    # Cache baseline predictions once per (model, crop) — same across scenarios
    print("\n=== phase 1: baseline predictions (once per model × crop) ===")
    baselines: dict[tuple[str, str], pd.DataFrame] = {}
    for model in models:
        for crop in crops:
            base = eval_model(model, crop, obs_recent)
            if base is not None:
                baselines[(model, crop)] = base
                print(f"  {model:13s} {crop:16s} -> {len(base)} rows")
            else:
                print(f"  {model:13s} {crop:16s} -> SKIP")

    for scenario in SCENARIOS:
        print(f"\n=== phase 2: {scenario} counterfactual ===")
        deltas = compute_delta_lookups(cmip, scenario, fw)
        obs_perturbed = perturb_observed_daily(obs_recent, deltas)
        tasmax_avg = float(deltas["tasmax"]["delta_tasmax"].mean())
        pr_pct_avg = float(deltas["pr"]["pct_pr"].mean() * 100)
        print(f"  mean tasmax delta: {tasmax_avg:+.2f} °C")
        print(f"  mean pr pct delta: {pr_pct_avg:+.2f} %")

        for model in models:
            for crop in crops:
                key = (model, crop)
                if key not in baselines:
                    continue
                cf = eval_model(model, crop, obs_perturbed)
                if cf is None:
                    print(f"    {model:13s} {crop:16s} -> counterfact SKIP")
                    continue
                s, d = summarize(model, crop, scenario,
                                 baselines[key], cf, tasmax_avg, pr_pct_avg)
                if s:
                    summary_rows.append(s)
                    district_rows.extend(d)
                    print(f"    {model:13s} {crop:16s} -> baseline {s['baseline_mean_pred']:.2f}, "
                          f"cf {s['counterfactual_mean_pred']:.2f}, Δ {s['delta_pct']:+.2f}%")

    OUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(OUT_SUMMARY, index=False)
    pd.DataFrame(district_rows).to_csv(OUT_DISTRICT, index=False)
    print(f"\nwrote {OUT_SUMMARY.name} ({len(summary_rows)} rows)")
    print(f"wrote {OUT_DISTRICT.name} ({len(district_rows)} rows)")

    if summary_rows:
        print("\n=== headline: yield delta % per (crop, scenario), mean over models ===")
        s = pd.DataFrame(summary_rows)
        lb = s.pivot_table(index="crop", columns="scenario", values="delta_pct", aggfunc="mean")
        print(lb.round(2).to_string())


if __name__ == "__main__":
    main()

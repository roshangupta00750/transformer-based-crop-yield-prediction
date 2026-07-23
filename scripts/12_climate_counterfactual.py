"""Train monthly-featured XGBoost + run climate counterfactual experiment.

Method (delta approach, standard in agricultural climate-impact literature):

1. Build one row per (landkreis, crop, year), with features aggregated over
   the growing season months. Weather features come from the monthly
   observed table (``dataset_monthly.parquet``).
2. Train XGBoost on 2003-2020, validate on 2021, test on 2022-2023.
3. Compute per-(district, month, variable) climate deltas from CMIP6:
       delta_var = mean_over_years(SSP <year window>) - mean_over_years(historical <baseline window>)
4. Apply delta to the observed 2022-2023 features to build counterfactual
   feature sets for each SSP scenario × future window (currently 2046-2050).
5. Score the same trained XGBoost on baseline (2022-23 observed) vs
   counterfactuals. Report per-crop yield anomaly.

Outputs:
    results/metrics/climate_counterfactual.csv
        columns: crop, scenario, mean_pred_yield_baseline,
                 mean_pred_yield_counterfactual, mean_yield_delta_dt_ha,
                 mean_yield_delta_pct, mean_tas_delta_C, mean_pr_delta_mm

    results/metrics/climate_counterfactual_district.csv
        long: per (crop, scenario, landkreis) baseline vs counterfactual pred

The XGBoost model itself is saved to results/checkpoints/xgboost_monthly.ubj
alongside its FeatureSpec JSON.

Usage:
    python scripts/12_climate_counterfactual.py
    python scripts/12_climate_counterfactual.py --future-window 2046 2050
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent.parent
OBS_MONTHLY = ROOT / "data" / "processed" / "dataset_monthly.parquet"
CMIP6 = ROOT / "data" / "processed" / "cmip6_nuts3.parquet"
CKPT_PATH = ROOT / "results" / "checkpoints" / "xgboost_monthly.ubj"
FEATURES_META_PATH = ROOT / "results" / "checkpoints" / "xgboost_monthly.features.json"
OUT_SUMMARY = ROOT / "results" / "metrics" / "climate_counterfactual.csv"
OUT_DISTRICT = ROOT / "results" / "metrics" / "climate_counterfactual_district.csv"

# Growing-season month windows per crop family. Winter crops planted in
# autumn, summer crops in spring; using a conservative March-August window
# captures the main phenology period for all 10 crops.
GROWING_MONTHS = list(range(3, 9))     # March-August
SEASON_TAG = "MarAug"

# Deliberately exclude rsds/radiation: the observed DWD radiation channel is
# absent for most model districts, so a learned radiation response would
# largely be a response to imputation artefacts.
WEATHER_VARS = ["tas", "tasmax", "tasmin", "pr"]
SOIL_VARS = ["bulk_density", "clay", "ph", "sand", "soc"]

TRAIN_YEARS = list(range(2003, 2021))
VAL_YEARS = [2021]
TEST_YEARS = [2022, 2023]

HIST_BASELINE_WINDOW = (2003, 2014)  # matches CMIP6 historical


def build_feature_frame(monthly: pd.DataFrame) -> pd.DataFrame:
    """Collapse (landkreis, crop, year, month) into (landkreis, crop, year)
    rows via one pivot per weather variable — vectorized, seconds not minutes."""
    keep = monthly[monthly["month"].isin(GROWING_MONTHS)].copy()
    print(f"  building feature rows from {len(keep):,} monthly rows (vectorized)...")

    # Pivot each weather variable to columns per month
    pivots = []
    for v in WEATHER_VARS:
        p = keep.pivot_table(
            index=["landkreis", "crop", "year"], columns="month", values=v,
            aggfunc="mean",
        )
        p.columns = [f"{v}_m{int(m):02d}" for m in p.columns]
        pivots.append(p)
    df = pd.concat(pivots, axis=1).reset_index()

    # Bring in soil + yield (constant per district-crop-year)
    static = (keep.drop_duplicates(subset=["landkreis", "crop", "year"])
                  [["landkreis", "crop", "year"] + SOIL_VARS + ["yield_dt_per_ha"]])
    df = df.merge(static, on=["landkreis", "crop", "year"], how="left")

    print(f"  {len(df):,} annual rows built ({df['landkreis'].nunique()} districts × "
          f"{df['crop'].nunique()} crops × {df['year'].nunique()} years)")
    return df


def train_xgb(features: pd.DataFrame) -> tuple[xgb.XGBRegressor, list[str], dict]:
    """Train XGBoost on TRAIN_YEARS, validate on VAL_YEARS."""
    feat_cols = [c for c in features.columns
                 if c not in ("landkreis", "crop", "year", "yield_dt_per_ha")]

    train = features[features["year"].isin(TRAIN_YEARS)].dropna(subset=feat_cols)
    val = features[features["year"].isin(VAL_YEARS)].dropna(subset=feat_cols)
    test = features[features["year"].isin(TEST_YEARS)].dropna(subset=feat_cols)
    print(f"  splits: train {len(train):,}, val {len(val):,}, test {len(test):,}")

    # Encode crop as a category feature — XGBoost handles it well
    for split in (train, val, test):
        split["crop_id"] = split["crop"].astype("category").cat.codes
    feat_cols = feat_cols + ["crop_id"]

    model = xgb.XGBRegressor(
        n_estimators=800, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        tree_method="hist", n_jobs=8, early_stopping_rounds=30,
        eval_metric="rmse",
    )
    model.fit(
        train[feat_cols], train["yield_dt_per_ha"],
        eval_set=[(val[feat_cols], val["yield_dt_per_ha"])],
        verbose=False,
    )

    # Metrics
    metrics = {}
    for name, df in [("train", train), ("val", val), ("test", test)]:
        y = df["yield_dt_per_ha"].to_numpy()
        p = model.predict(df[feat_cols])
        metrics[name] = {
            "r2": float(r2_score(y, p)),
            "rmse": float(np.sqrt(mean_squared_error(y, p))),
            "mae": float(mean_absolute_error(y, p)),
            "n": int(len(df)),
        }
    print(f"  metrics: train R²={metrics['train']['r2']:.3f}, "
          f"val R²={metrics['val']['r2']:.3f}, "
          f"test R²={metrics['test']['r2']:.3f}")
    return model, feat_cols, metrics


def cmip6_delta(cmip: pd.DataFrame, scenario: str,
                future_window: tuple[int, int]) -> pd.DataFrame:
    """Compute per-(landkreis, month, variable) delta:
    mean(scenario future window) - mean(historical baseline window).
    Returns wide table: (landkreis, month) × variable delta."""
    hist_mean = (cmip[
        (cmip["experiment"] == "historical")
        & (cmip["year"].between(*HIST_BASELINE_WINDOW))
    ].groupby(["landkreis", "month", "variable"])["value"].mean())

    fut_mean = (cmip[
        (cmip["experiment"] == scenario)
        & (cmip["year"].between(*future_window))
    ].groupby(["landkreis", "month", "variable"])["value"].mean())

    delta = (fut_mean - hist_mean).reset_index()
    return delta.pivot_table(index=["landkreis", "month"], columns="variable",
                             values="value").reset_index()


def apply_counterfactual(features_test: pd.DataFrame,
                         delta_wide: pd.DataFrame) -> pd.DataFrame:
    """Apply CMIP6 delta to the test-year feature rows.

    Vectorized: for each month m, one merge on landkreis brings in the
    delta columns; add to <var>_m{m:02d}. O(n) not O(n·m·v).
    """
    counterfact = features_test.copy()
    for m in GROWING_MONTHS:
        deltas_m = delta_wide[delta_wide["month"] == m].drop(columns=["month"])
        if deltas_m.empty:
            continue
        merged = counterfact[["landkreis"]].merge(
            deltas_m, on="landkreis", how="left")
        for v in WEATHER_VARS:
            col = f"{v}_m{m:02d}"
            if col not in counterfact.columns or v not in merged.columns:
                continue
            delta_series = merged[v].fillna(0.0).to_numpy()
            counterfact[col] = counterfact[col].to_numpy() + delta_series
    return counterfact


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--future-window", nargs=2, type=int, default=[2046, 2050],
                    metavar=("START", "END"))
    args = ap.parse_args()
    fw = (args.future_window[0], args.future_window[1])

    print(f"Growing-season window: months {GROWING_MONTHS[0]}-{GROWING_MONTHS[-1]} ({SEASON_TAG})")
    print(f"Historical baseline for delta: {HIST_BASELINE_WINDOW[0]}-{HIST_BASELINE_WINDOW[1]}")
    print(f"Future counterfactual window: {fw[0]}-{fw[1]}")

    print("\nreading monthly observed...")
    monthly = pd.read_parquet(OBS_MONTHLY)
    print(f"  {len(monthly):,} monthly rows, {monthly['landkreis'].nunique()} districts")

    print("\nbuilding annual features...")
    features = build_feature_frame(monthly)

    print("\ntraining monthly-featured XGBoost...")
    model, feat_cols, metrics = train_xgb(features)

    CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(CKPT_PATH))
    FEATURES_META_PATH.write_text(json.dumps({
        "feat_cols": feat_cols,
        "metrics": metrics,
        "growing_months": GROWING_MONTHS,
        "train_years": TRAIN_YEARS,
        "val_years": VAL_YEARS,
        "test_years": TEST_YEARS,
    }, indent=2))
    print(f"  saved {CKPT_PATH.name}")

    print("\nreading CMIP6 processed...")
    cmip = pd.read_parquet(CMIP6)

    test = features[features["year"].isin(TEST_YEARS)].dropna(subset=[c for c in feat_cols if c != "crop_id"]).copy()
    test["crop_id"] = test["crop"].astype("category").cat.codes
    baseline_pred = model.predict(test[feat_cols])
    test["pred_baseline"] = baseline_pred

    summary_rows, district_rows = [], []
    for scenario in ["ssp2_4_5", "ssp5_8_5"]:
        print(f"\ncounterfactual: {scenario} × {fw[0]}-{fw[1]}")
        delta = cmip6_delta(cmip, scenario, fw)
        cfact = apply_counterfactual(test.drop(columns=["pred_baseline"]), delta)
        cfact["crop_id"] = cfact["crop"].astype("category").cat.codes
        cfact_pred = model.predict(cfact[feat_cols])
        test[f"pred_{scenario}"] = cfact_pred

        # Per-crop summary
        for crop, gsub in test.groupby("crop"):
            base = gsub["pred_baseline"].mean()
            cf = gsub[f"pred_{scenario}"].mean()
            # Also the mean tas + pr delta for the crop
            crop_delta = delta.groupby("month").mean(numeric_only=True)
            tas_delta = crop_delta.loc[GROWING_MONTHS, "tas"].mean() if "tas" in crop_delta.columns else np.nan
            pr_delta = crop_delta.loc[GROWING_MONTHS, "pr"].mean() if "pr" in crop_delta.columns else np.nan
            summary_rows.append({
                "crop": crop, "scenario": scenario,
                "future_window": f"{fw[0]}-{fw[1]}",
                "baseline_pred_mean": float(base),
                "counterfactual_pred_mean": float(cf),
                "yield_delta_dt_ha": float(cf - base),
                "yield_delta_pct": float(100 * (cf - base) / base) if base > 0 else np.nan,
                "growing_season_tas_delta_C": float(tas_delta),
                "growing_season_pr_delta_mm": float(pr_delta),
                "n_districts_years": int(len(gsub)),
            })
        # District-level
        for _, row in test.iterrows():
            district_rows.append({
                "landkreis": row["landkreis"], "crop": row["crop"],
                "year": int(row["year"]), "scenario": scenario,
                "baseline_pred": float(row["pred_baseline"]),
                "counterfactual_pred": float(row[f"pred_{scenario}"]),
                "delta_dt_ha": float(row[f"pred_{scenario}"] - row["pred_baseline"]),
            })

    OUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(OUT_SUMMARY, index=False)
    pd.DataFrame(district_rows).to_csv(OUT_DISTRICT, index=False)
    print(f"\nwrote {OUT_SUMMARY.name} ({len(summary_rows)} rows)")
    print(f"wrote {OUT_DISTRICT.name} ({len(district_rows)} rows)")

    print("\n=== headline: mean yield delta per (crop, scenario) ===")
    lb = pd.DataFrame(summary_rows).pivot_table(
        index="crop", columns="scenario", values="yield_delta_pct")
    print(lb.round(2).to_string())


if __name__ == "__main__":
    main()

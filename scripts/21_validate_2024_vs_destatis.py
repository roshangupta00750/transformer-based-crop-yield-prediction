"""Validate the 2024 forecast against manually-downloaded Destatis 2024 truth.

Input:
  data/raw/destatis/destatis_2024_manual.csv   (user-downloaded 41241-01-03-4)
  results/metrics/forecast_2024.csv            (forecast from 20_forecast_2024.py)

Output:
  results/metrics/forecast_2024_metrics.csv    (per-model, per-crop R²/RMSE/MAE)
  results/metrics/forecast_2024_leaderboard.csv (per-model mean-R² leaderboard)

Ingestion path:
  * The Destatis CSV uses German column headers + `;` separator, `,` decimal
  * We reuse `destatis_yield.clean(...)` if compatible, else parse inline
  * Map Destatis crop labels → our internal crop names (winter_wheat, etc.)
  * Map Destatis Kreis names → NUTS-3 landkreis codes (reuses
    `src/data/acquisition/destatis_yield.py`'s KREIS_TO_NUTS3 helper if
    available)
  * Join to forecast_2024.csv on (crop, landkreis), compute metrics
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# Read from the main destatis file (which now includes 2024 rows after the
# credentialed pull); manual placeholder is a fallback if the API path fails.
DESTATIS_CSV = REPO / "data" / "raw" / "destatis" / "destatis_41241-01-03-4.csv"
MANUAL_CSV = REPO / "data" / "raw" / "destatis" / "destatis_2024_manual.csv"
FORECAST_CSV = REPO / "results" / "metrics" / "forecast_2024.csv"
METRICS_CSV = REPO / "results" / "metrics" / "forecast_2024_metrics.csv"
LEADERBOARD_CSV = REPO / "results" / "metrics" / "forecast_2024_leaderboard.csv"

# Destatis crop labels → our internal names (matches configs/data.yaml crops)
CROP_MAP = {
    "Winterweizen": "winter_wheat",
    "Wintergerste": "winter_barley",
    "Roggen": "rye",
    "Roggen und Wintermenggetreide": "rye",  # older label
    "Triticale": "triticale",
    "Winterraps": "winter_rapeseed",
    "Sommergerste": "summer_barley",
    "Hafer": "oats",
    "Silomais": "silage_maize",
    "Silomais/Grünmais": "silage_maize",
    "Kartoffeln": "potatoes",
    "Zuckerrüben": "sugar_beet",
}


def load_destatis_2024() -> pd.DataFrame:
    """Ingest Destatis 2024 truth by reusing the existing clean() step
    (handles AGS→NUTS3 mapping via GISCO shapefile) instead of ad-hoc parsing."""
    src = DESTATIS_CSV if DESTATIS_CSV.exists() else MANUAL_CSV
    if not src.exists():
        raise SystemExit(
            f"[abort] neither {DESTATIS_CSV.relative_to(REPO)} nor "
            f"{MANUAL_CSV.relative_to(REPO)} found."
        )
    print(f"  reading Destatis from: {src.relative_to(REPO)}")

    from src.data.acquisition.destatis_yield import clean as destatis_clean
    # Materialize a cleaned yield parquet (this rebuilds interim/yield.parquet
    # with 2024 rows now included, since the raw CSV has 2024)
    interim_out = REPO / "data" / "interim" / "yield.parquet"
    destatis_clean(raw_path=src, out_path=interim_out)
    yld = pd.read_parquet(interim_out)
    print(f"  cleaned yield: {len(yld):,} rows across "
          f"{yld['crop'].nunique()} crops, {yld['landkreis'].nunique()} districts, "
          f"years {yld['year'].min()}-{yld['year'].max()}")

    yld_2024 = yld[yld["year"] == 2024]
    print(f"  2024 rows: {len(yld_2024):,} across "
          f"{yld_2024['crop'].nunique()} crops, {yld_2024['landkreis'].nunique()} districts")
    if yld_2024.empty:
        raise SystemExit("[abort] no 2024 rows in cleaned Destatis output.")

    return yld_2024[["landkreis", "crop", "yield_dt_per_ha"]]
    print(f"  Destatis CSV: {df.shape} — columns: {list(df.columns)[:5]}...")

    # Try to detect the schema flexibly. The typical fields are:
    #   time, 1_variable_attribute_code (Kreis code), 1_variable_attribute_label,
    #   2_variable_attribute_label (crop label), value (yield dt/ha)
    keep_cols = {
        "time": ["time", "Jahr"],
        "kreis_code": ["1_variable_attribute_code", "Kreise (2019) Schlüssel",
                       "Regional (2019) Schlüssel", "Kreis-code"],
        "kreis_label": ["1_variable_attribute_label", "Kreise (2019)",
                        "Regional (2019)", "Regionale Einheit"],
        "crop_label": ["2_variable_attribute_label", "Feldfruchtarten",
                       "Feldfrucht"],
        "value": ["value", "Hektarerträge (dt/ha)", "Wert"],
    }
    resolved = {}
    for k, cands in keep_cols.items():
        for c in cands:
            if c in df.columns:
                resolved[k] = c
                break
        if k not in resolved:
            print(f"  [warn] no column matched for {k} (candidates: {cands})")
    if len(resolved) < 4:
        raise SystemExit(f"[abort] destatis CSV schema not recognized. Resolved: {resolved}")

    df = df.rename(columns={v: k for k, v in resolved.items()})[list(resolved)]
    df = df[df["time"] == 2024]
    print(f"  after year=2024 filter: {len(df)} rows")

    # Map crop labels
    df["crop"] = df["crop_label"].map(CROP_MAP)
    unmapped = df[df["crop"].isna()]["crop_label"].unique()
    if len(unmapped):
        print(f"  [warn] unmapped crop labels: {unmapped}")
    df = df[df["crop"].notna()]

    # Value → float; may have "-" or "." placeholders → NaN
    df["yield_dt_per_ha"] = pd.to_numeric(df["value"], errors="coerce")

    # Map Kreis code → NUTS-3 landkreis id (Destatis code = 5-digit AGS,
    # NUTS-3 is DEXXX). Try the destatis_yield helper first.
    try:
        from src.data.acquisition.destatis_yield import KREIS_TO_NUTS3
        df["landkreis"] = df["kreis_code"].astype(str).str.zfill(5).map(KREIS_TO_NUTS3)
    except Exception:
        # Fallback: use kreis_label lookup via NUTS-3 shapefile
        print("  [warn] KREIS_TO_NUTS3 helper unavailable; trying label match")
        from src.data.landkreise import load_landkreise
        de = load_landkreise()
        name_to_nuts = dict(zip(de["landkreis_name"], de["landkreis"]))
        df["landkreis"] = df["kreis_label"].map(name_to_nuts)

    unmapped_k = df[df["landkreis"].isna()]["kreis_label"].nunique()
    if unmapped_k:
        print(f"  [warn] {unmapped_k} unmapped Kreis names (dropped)")
    df = df[df["landkreis"].notna()]
    print(f"  final 2024 Destatis rows: {len(df):,} across {df['crop'].nunique()} crops "
          f"and {df['landkreis'].nunique()} districts")

    return df[["landkreis", "crop", "yield_dt_per_ha"]]


def main() -> None:
    if not FORECAST_CSV.exists():
        raise SystemExit(f"[abort] {FORECAST_CSV} not found — run 20_forecast_2024.py first.")

    truth = load_destatis_2024()
    fc = pd.read_csv(FORECAST_CSV)
    print(f"  forecast_2024.csv: {len(fc):,} rows, "
          f"{fc['model'].nunique()} models, {fc['crop'].nunique()} crops")

    # Rename truth column so it doesn't collide with fc.y_true (which was NaN
    # in forecast-only mode). We use the Destatis-cleaned truth as ground truth.
    truth = truth.rename(columns={"yield_dt_per_ha": "y_truth"})
    joined = fc.merge(truth, on=["landkreis", "crop"], how="inner")
    print(f"  joined rows (both forecast + truth): {len(joined):,}")
    if joined.empty:
        raise SystemExit("[abort] No overlap between forecast + Destatis truth.")

    rows = []
    for (m, c), sub in joined.groupby(["model", "crop"]):
        sub = sub.dropna(subset=["y_pred", "y_truth"])
        if len(sub) < 3:
            continue
        yt = sub["y_truth"].to_numpy()
        yp = sub["y_pred"].to_numpy()
        rows.append({
            "model": m, "crop": c,
            "r2": float(r2_score(yt, yp)),
            "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
            "mae": float(mean_absolute_error(yt, yp)),
            "n": int(len(sub)),
        })

    metrics = pd.DataFrame(rows)
    metrics.to_csv(METRICS_CSV, index=False)
    print(f"  wrote {METRICS_CSV.relative_to(REPO)}  ({len(metrics)} rows)")

    lb = (metrics.groupby("model")
          .agg(mean_r2=("r2", "mean"),
               median_r2=("r2", "median"),
               min_r2=("r2", "min"),
               max_r2=("r2", "max"),
               n_crops=("crop", "nunique"))
          .sort_values("mean_r2", ascending=False)
          .round(3))
    lb.to_csv(LEADERBOARD_CSV)
    print(f"  wrote {LEADERBOARD_CSV.relative_to(REPO)}")
    print()
    print("=== 2024 FORECAST vs DESTATIS TRUTH — LEADERBOARD ===")
    print(lb.to_string())
    print()
    print("=== PER-CROP MATRIX ===")
    pivot = metrics.pivot(index="crop", columns="model", values="r2").round(3)
    print(pivot.to_string())


if __name__ == "__main__":
    main()

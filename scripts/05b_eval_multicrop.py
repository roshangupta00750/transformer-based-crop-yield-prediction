"""Multi-crop evaluation: per-(model, crop) R²/RMSE + mean-per-crop headline.

Uses the dashboard prediction helpers (which handle TFT's pf path too) to
get one test prediction per (landkreis, year) for each crop, then computes
metrics in absolute dt/ha space. Writes:

  results/metrics/multicrop_summary.csv   (model, crop, r2, rmse, mae, n)
  results/metrics/multicrop_leaderboard.csv (model, mean_r2, median_r2, n_crops)

Pooled-raw R² across crops is intentionally NOT used as the headline — it's
dominated by between-crop scale differences. Mean-per-crop R² weights each
crop equally and reflects within-(district, crop) skill.
"""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dashboard"))

# Stub streamlit so the dashboard helper imports cleanly outside a run.
st = types.ModuleType("streamlit")
st.cache_data = lambda **kw: (lambda f: f)
st.set_page_config = lambda **kw: None
sys.modules["streamlit"] = st

import importlib.util
import json
spec = importlib.util.spec_from_file_location("mt", str(ROOT / "dashboard" / "_model_template.py"))
mt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mt)

from src.utils import load_data_config
from src.data.dataset import FEATURE_POLICY_VERSION

MODELS = ["xgboost", "dlinear", "itransformer", "lstm", "patchtst", "tft"]


def main() -> None:
    cfg = load_data_config()
    crops = cfg["crops"]

    # Incremental write: append after each (model, crop) so a mid-eval crash
    # (e.g. TFT loading OOM) still leaves a partial multicrop_summary.csv.
    summary_path = ROOT / "results" / "metrics" / "multicrop_summary.csv"
    policy_path = ROOT / "results" / "metrics" / "feature_policy.json"
    if policy_path.exists():
        policy_path.unlink()
    if summary_path.exists():
        summary_path.unlink()
    header_written = False

    rows = []
    for model in MODELS:
        for crop in crops:
            try:
                df, err = mt._predict_test_set(model, crop)
            except Exception as e:  # noqa: BLE001
                err = {"error": str(e)}
                df = pd.DataFrame()
            if err or df.empty:
                print(f"  {model:13s} {crop:16s} -> SKIP ({(err or {}).get('error','empty')[:50]})")
                continue
            yt, yp = df["y_true"].to_numpy(), df["y_pred"].to_numpy()
            r2 = r2_score(yt, yp)
            rmse = float(np.sqrt(mean_squared_error(yt, yp)))
            mae = float(mean_absolute_error(yt, yp))
            row = {"model": model, "crop": crop, "r2": r2,
                   "rmse": rmse, "mae": mae, "n": len(df)}
            rows.append(row)
            pd.DataFrame([row]).to_csv(summary_path, mode="a", index=False,
                                        header=not header_written)
            header_written = True
            print(f"  {model:13s} {crop:16s} -> R²={r2:6.3f}  RMSE={rmse:8.2f}  n={len(df)}", flush=True)

    summary = pd.DataFrame(rows)

    lead = (summary.groupby("model")
                   .agg(mean_r2=("r2", "mean"),
                        median_r2=("r2", "median"),
                        min_r2=("r2", "min"),
                        max_r2=("r2", "max"),
                        n_crops=("crop", "nunique"))
                   .sort_values("mean_r2", ascending=False)
                   .round(3))
    lead.to_csv(ROOT / "results" / "metrics" / "multicrop_leaderboard.csv")

    expected_pairs = len(MODELS) * len(crops)
    completed_pairs = summary[["model", "crop"]].drop_duplicates().shape[0]
    if completed_pairs == expected_pairs:
        policy_path.write_text(json.dumps({
            "feature_policy_version": FEATURE_POLICY_VERSION,
            "models": MODELS,
            "crops": crops,
            "completed_pairs": completed_pairs,
        }, indent=2))
        print(f"Wrote current-results marker: {policy_path}")
    else:
        print(
            f"Results remain unmarked: completed {completed_pairs}/"
            f"{expected_pairs} model-crop pairs."
        )

    print("\n=== Mean-per-crop leaderboard ===")
    print(lead.to_string())
    print("\n=== Per-crop R² matrix (model × crop) ===")
    pivot = summary.pivot(index="crop", columns="model", values="r2").round(3)
    # order models by mean
    pivot = pivot[lead.index.tolist()]
    print(pivot.to_string())


if __name__ == "__main__":
    main()

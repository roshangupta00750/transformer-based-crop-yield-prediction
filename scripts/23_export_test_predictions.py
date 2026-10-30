"""Export per-district test predictions from the CURRENT checkpoints.

The dashboard's Geographic and Predictions pages need one row per
(model, crop, district, year). The older files that used to feed them
(forecast_2024.csv, spatial_per_district.csv) predate the current checkpoints,
so they describe models that no longer exist. This regenerates the rows from
whatever is in results/checkpoints/ now. Inference only, nothing is trained.

Each row also carries ``y_hist``: the district's historical average for that
crop (the training baseline, with its train-only trend where one applies).
That is exactly what the model is asked to improve on, so every map and chart
can show model error next to the error of simply using the historical average.

Writes results/metrics/test_predictions.csv:
    model, crop, landkreis, year, y_true, y_pred, y_hist

Usage:
    python scripts/23_export_test_predictions.py
    python scripts/23_export_test_predictions.py --models itransformer,lstm
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

# Stub streamlit so the dashboard helper imports cleanly outside a run
# (same approach as scripts/05b_eval_multicrop.py).
st = types.ModuleType("streamlit")
st.cache_data = lambda **kw: (lambda f: f)
st.cache_resource = lambda **kw: (lambda f: f)
st.set_page_config = lambda **kw: None
sys.modules["streamlit"] = st

import importlib.util
spec = importlib.util.spec_from_file_location(
    "mt", str(ROOT / "dashboard" / "_model_template.py"))
mt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mt)

from src.utils import load_data_config

# With streamlit stubbed out there is no cache, so _predict_test_set would
# re-read the full daily table (16.5M rows) for every (model, crop) pair.
# Read it once and hand the same frame back; callers only filter it.
_UNIFIED = mt.load_unified()
mt.load_unified = lambda: _UNIFIED

MODELS =["xgboost", "dlinear", "itransformer", "lstm", "patchtst", "tft"]
OUT = ROOT / "results" / "metrics" / "test_predictions.csv"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", default=",".join(MODELS))
    args = ap.parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    crops = load_data_config()["crops"]
    frames, skipped = [], []

    for model in models:
        pipeline = mt.load_pipeline(model) if model != "tft" else None
        for crop in crops:
            try:
                df, err = mt._predict_test_set(model, crop)
            except Exception as e:  # noqa: BLE001
                df, err = pd.DataFrame(), {"error": str(e)}
            if err or df.empty:
                msg = (err or {}).get("error", "empty")[:70]
                skipped.append((model, crop, msg))
                print(f"  {model:13s} {crop:16s} SKIP  {msg}", flush=True)
                continue

            df = df[["landkreis", "year", "y_true", "y_pred"]].copy()
            df.insert(0, "crop", crop)
            df.insert(0, "model", model)

            # Historical average for the same rows. TFT has no shared
            # pipeline JSON here, so its rows reuse another model's baseline
            # below; the baseline depends only on training yields, not on
            # the model.
            if pipeline is not None:
                df["y_hist"] = pipeline.inverse_transform_array(
                    df["landkreis"].to_numpy(),
                    np.full(len(df), crop),
                    np.zeros(len(df)),
                    df["year"].to_numpy(),
                )
            frames.append(df)
            rmse = float(np.sqrt(((df.y_true - df.y_pred) ** 2).mean()))
            print(f"  {model:13s} {crop:16s} n={len(df):4d}  RMSE={rmse:7.2f}",
                  flush=True)

    if not frames:
        print("nothing exported")
        sys.exit(1)

    out = pd.concat(frames, ignore_index=True)

    # Fill y_hist for any model that lacked a pipeline (TFT) from the rows of
    # a model that has it, keyed on (crop, district, year).
    if "y_hist" in out and out["y_hist"].isna().any():
        ref = (out.dropna(subset=["y_hist"])
                  .drop_duplicates(["crop", "landkreis", "year"])
                  .set_index(["crop", "landkreis", "year"])["y_hist"])
        miss = out["y_hist"].isna()
        keys = list(zip(out.loc[miss, "crop"], out.loc[miss, "landkreis"],
                        out.loc[miss, "year"]))
        out.loc[miss, "y_hist"] = ref.reindex(keys).to_numpy()

    out.to_csv(OUT, index=False)
    print(f"\nWrote {OUT} ({len(out)} rows, "
          f"{out['model'].nunique()} models, {out['crop'].nunique()} crops)")
    if skipped:
        print(f"Skipped {len(skipped)} (model, crop) pairs:")
        for m, c, msg in skipped:
            print(f"  {m} / {c}: {msg}")


if __name__ == "__main__":
    main()

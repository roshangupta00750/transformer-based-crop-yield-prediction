"""Per-(model, crop, horizon) evaluation matrix.

Produces ``results/metrics/multicrop_horizons.csv`` with one row per
(model, crop, horizon). Used by the dashboard's In-Season page.

Horizons are taken from ``configs/data.yaml`` (in_season_horizons_days) plus
a 'full' row that represents the full-window prediction (no cutoff).
"""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
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
spec = importlib.util.spec_from_file_location("mt", str(ROOT / "dashboard" / "_model_template.py"))
mt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mt)

from src.utils import load_data_config

MODELS = ["xgboost", "dlinear", "itransformer", "lstm", "patchtst", "tft"]


FIELDS = ["model", "crop", "horizon", "horizon_days", "harvest_doy",
          "cutoff_doy", "r2", "rmse", "mae", "n"]


def _existing_keys(out_path: Path) -> set:
    """Set of (model, crop, horizon) tuples already in the CSV — to skip on resume."""
    if not out_path.exists():
        return set()
    try:
        prev = pd.read_csv(out_path)
        return set(zip(prev["model"], prev["crop"], prev["horizon"]))
    except Exception:  # noqa: BLE001
        return set()


def main() -> None:
    cfg = load_data_config()
    crops = cfg["crops"]
    horizons_days = cfg["in_season_horizons_days"]
    phenology = cfg["phenology"]

    out_path = ROOT / "results" / "metrics" / "multicrop_horizons.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = _existing_keys(out_path)
    if done:
        print(f"Resuming: {len(done)} (model, crop, horizon) tuples already done.")

    # Write header now if file is missing — every subsequent eval appends one row.
    if not out_path.exists():
        pd.DataFrame(columns=FIELDS).to_csv(out_path, index=False)

    n_written = 0
    for model in MODELS:
        for crop in crops:
            harvest_doy = phenology.get(crop, {}).get("harvest_doy", 220)
            for h in [None] + list(horizons_days):
                label = "full" if h is None else f"h-{h}d"
                if (model, crop, label) in done:
                    continue
                cutoff = None if h is None else max(harvest_doy - int(h), 1)
                try:
                    df, err = mt._predict_test_set(model, crop, cutoff_doy=cutoff)
                except Exception as e:  # noqa: BLE001
                    err = {"error": f"{type(e).__name__}: {e}"}
                    df = pd.DataFrame()
                if err or df.empty:
                    print(f"  {model:13s} {crop:16s} {label:7s} -> SKIP "
                          f"({(err or {}).get('error', 'empty')[:50]})")
                    continue
                yt, yp = df["y_true"].to_numpy(), df["y_pred"].to_numpy()
                row = {
                    "model": model, "crop": crop, "horizon": label,
                    "horizon_days": (None if h is None else int(h)),
                    "harvest_doy": int(harvest_doy),
                    "cutoff_doy": (None if cutoff is None else int(cutoff)),
                    "r2": float(r2_score(yt, yp)),
                    "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
                    "mae": float(mean_absolute_error(yt, yp)),
                    "n": len(df),
                }
                pd.DataFrame([row])[FIELDS].to_csv(
                    out_path, mode="a", header=False, index=False)
                n_written += 1
                print(f"  {model:13s} {crop:16s} {label:7s} -> R²={row['r2']:6.3f}  "
                      f"RMSE={row['rmse']:8.2f}  n={row['n']}")

    print(f"\nWrote {n_written} new rows to {out_path}")


if __name__ == "__main__":
    main()

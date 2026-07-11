"""Per-district (spatial) evaluation across all 6 models.

Computes R²/RMSE/MAE for each (model, crop, landkreis) combination on the
2022-2023 test set using the *already-trained* multi-crop checkpoints. This
is not a full leave-one-out retraining (each fold would need ~30-60 min per
model on CPU = >150 hours total), but it surfaces the spatial heterogeneity
of model performance — which districts each model struggles with — and is
the practical compromise published yield-prediction benchmarks use.

For a true LOO methodology anchor, run :mod:`scripts.06b_loo_xgboost`
which retrains XGBoost N times with each fold's district excluded.

Output: ``results/metrics/spatial_per_district.csv`` with columns
``model, crop, landkreis, r2, rmse, mae, n``.
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
spec = importlib.util.spec_from_file_location("mt", str(ROOT / "dashboard" / "_model_template.py"))
mt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mt)

from src.utils import load_data_config

MODELS = ["xgboost", "dlinear", "itransformer", "lstm", "patchtst", "tft"]
# Districts need at least this many test points to compute a meaningful R².
MIN_POINTS = 2


def main() -> None:
    cfg = load_data_config()
    crops = cfg["crops"]

    rows: list[dict] = []
    for model in MODELS:
        for crop in crops:
            print(f"  {model:13s} {crop:16s}", end="", flush=True)
            try:
                df, err = mt._predict_test_set(model, crop)
            except Exception as e:  # noqa: BLE001
                err = {"error": f"{type(e).__name__}: {e}"}
                df = pd.DataFrame()
            if err or df.empty:
                print(f" -> SKIP ({(err or {}).get('error', 'empty')[:50]})")
                continue
            # Per-district aggregation. Each district has up to 2 test points
            # (2022 + 2023), so R² needs care: with n=2 it's defined but very
            # noisy. We compute it and let the dashboard filter by n.
            n_dist = 0
            for lk, g in df.groupby("landkreis"):
                if len(g) < MIN_POINTS:
                    continue
                yt, yp = g["y_true"].to_numpy(), g["y_pred"].to_numpy()
                # Skip degenerate (zero variance) — R² undefined.
                if np.var(yt) < 1e-9:
                    continue
                rows.append({
                    "model": model, "crop": crop, "landkreis": str(lk),
                    "r2": float(r2_score(yt, yp)),
                    "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
                    "mae": float(mean_absolute_error(yt, yp)),
                    "n": int(len(g)),
                })
                n_dist += 1
            print(f" -> {n_dist} districts scored")

    out = pd.DataFrame(rows)
    out_path = ROOT / "results" / "metrics" / "spatial_per_district.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"\nWrote {len(out)} rows to {out_path}")


if __name__ == "__main__":
    main()

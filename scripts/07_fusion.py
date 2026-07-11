"""Cross-model fusion — try multiple methods on cached base-model predictions.

Workflow
--------
1.  Generate (or load from parquet cache) per-(model, crop) predictions on
    validation (2021) and test (2022-23) for the 5 base models. The
    predictions are deterministic, so on subsequent runs they're loaded
    from ``results/metrics/fusion_predictions_{val,test}.parquet`` and
    no re-prediction happens (seconds vs ~50-80 min).

2.  Apply five fusion methods per crop:
        ridge_alpha1   — Ridge(alpha=1.0) on val, predict test
        nnls           — non-negative least squares (no sign flips)
        simple_avg     — uniform mean of base predictions
        weighted_avg   — mean weighted by each base model's val R²
        median         — per-sample median across base models
    Each is compared against ``best_single``: the val-best single model
    for that crop (textbook baseline — proper holdout selection).

3.  Write ``results/metrics/fusion_summary.csv`` with one row per
    (crop, method). Also write ``fusion_coefficients.csv`` (ridge weights
    + NNLS weights) for interpretability.

TFT excluded — broken checkpoint (R² ≈ -23 mean).
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
from scipy.optimize import nnls
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dashboard"))

# Stub streamlit so importing the dashboard helper works headless.
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

DEFAULT_BASE_MODELS = ["xgboost", "dlinear", "itransformer", "lstm", "patchtst", "tft"]
BASE_MODELS = list(DEFAULT_BASE_MODELS)  # can be reduced via CLI
RIDGE_ALPHA = 1.0
CACHE_DIR = ROOT / "results" / "metrics"


def _predict_or_skip(model: str, crop: str,
                     years: list[int]) -> pd.DataFrame | None:
    try:
        df, err = mt._predict_test_set(model, crop, years=years)
    except Exception as e:  # noqa: BLE001
        print(f"    {model:13s} -> EXCEPTION ({type(e).__name__}): "
              f"{str(e)[:80]}")
        return None
    if err or df.empty:
        print(f"    {model:13s} -> SKIP "
              f"({(err or {}).get('error', 'empty')[:60]})")
        return None
    out = df[["landkreis", "year", "y_true", "y_pred"]].copy()
    out["model"] = model
    out["crop"] = crop
    return out


def collect_predictions(split_name: str, years: list[int],
                        crops: list[str]) -> pd.DataFrame:
    """Predict (or load from cache) every (model, crop) on the given split.

    Incremental cache: if the cache has some (model, crop) combos but not
    others, only the missing combos are generated and appended.
    """
    cache_path = CACHE_DIR / f"fusion_predictions_{split_name}.parquet"
    cached = pd.DataFrame()
    if cache_path.exists():
        cached = pd.read_parquet(cache_path)
        existing = set(zip(cached["model"], cached["crop"]))
        needed = {(m, c) for m in BASE_MODELS for c in crops}
        missing = needed - existing
        if not missing:
            print(f"  ✓ loaded {len(cached)} {split_name} predictions from cache")
            return cached
        print(f"  cache has {len(existing)} combos, {len(missing)} missing "
              f"({split_name}) — generating just those")
    else:
        missing = {(m, c) for m in BASE_MODELS for c in crops}
        print(f"  no cache — generating {len(missing)} {split_name} predictions")

    rows: list[pd.DataFrame] = [cached] if not cached.empty else []
    for m, crop in sorted(missing):
        df = _predict_or_skip(m, crop, years)
        if df is not None:
            rows.append(df)
            print(f"    {m:13s} {crop:16s} -> {len(df)} rows")
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cache_path, index=False)
    print(f"  cached → {cache_path} ({len(out)} rows total)")
    return out


def to_wide(long: pd.DataFrame) -> pd.DataFrame:
    """Pivot long → wide so each row is one (landkreis, year, crop) and
    each model is a column."""
    if long.empty:
        return long
    wide = long.pivot_table(
        index=["crop", "landkreis", "year", "y_true"],
        columns="model", values="y_pred",
    ).reset_index()
    wide.columns.name = None
    return wide


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    if np.var(y_true) < 1e-9 or len(y_true) < 2:
        return {"r2": float("nan"),
                "rmse": float(np.sqrt(((y_true - y_pred) ** 2).mean())),
                "mae": float(np.abs(y_true - y_pred).mean()),
                "n": int(len(y_true))}
    return {"r2": float(r2_score(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "n": int(len(y_true))}


def fuse_per_crop(val_wide: pd.DataFrame,
                  test_wide: pd.DataFrame,
                  crop: str) -> tuple[list[dict], list[dict]]:
    """Apply all fusion methods to one crop. Returns (summary_rows, coef_rows)."""
    v = val_wide[val_wide["crop"] == crop].dropna()
    t = test_wide[test_wide["crop"] == crop].dropna()
    common_models = sorted(set(BASE_MODELS) & set(v.columns) & set(t.columns))
    if len(common_models) < 2:
        print(f"  [{crop}] skip — only {len(common_models)} models with both splits")
        return [], []

    Xval = v[common_models].to_numpy()
    yval = v["y_true"].to_numpy()
    Xtest = t[common_models].to_numpy()
    ytest = t["y_true"].to_numpy()
    print(f"  [{crop}] val={len(v)}, test={len(t)}, models={common_models}")

    summary: list[dict] = []
    coefs: list[dict] = []

    # ── Method 1: Ridge ───────────────────────────────────────────────────
    ridge = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True)
    ridge.fit(Xval, yval)
    yhat = ridge.predict(Xtest)
    summary.append({"crop": crop, "method": "ridge_alpha1",
                    "n_base": len(common_models), **metrics(ytest, yhat)})
    for c, w in zip(common_models, ridge.coef_):
        coefs.append({"crop": crop, "method": "ridge_alpha1",
                      "model": c, "coefficient": float(w)})
    coefs.append({"crop": crop, "method": "ridge_alpha1",
                  "model": "_intercept_", "coefficient": float(ridge.intercept_)})

    # ── Method 2: NNLS (non-negative least squares) ──────────────────────
    # No intercept by default — augment X with a column of ones to fit one.
    Xv_nn = np.hstack([Xval, np.ones((len(Xval), 1))])
    Xt_nn = np.hstack([Xtest, np.ones((len(Xtest), 1))])
    w_nn, _ = nnls(Xv_nn, yval)
    yhat = Xt_nn @ w_nn
    summary.append({"crop": crop, "method": "nnls",
                    "n_base": len(common_models), **metrics(ytest, yhat)})
    for c, w in zip(common_models, w_nn[:-1]):
        coefs.append({"crop": crop, "method": "nnls",
                      "model": c, "coefficient": float(w)})
    coefs.append({"crop": crop, "method": "nnls",
                  "model": "_intercept_", "coefficient": float(w_nn[-1])})

    # ── Method 3: Simple average ──────────────────────────────────────────
    yhat = Xtest.mean(axis=1)
    summary.append({"crop": crop, "method": "simple_avg",
                    "n_base": len(common_models), **metrics(ytest, yhat)})

    # ── Method 4: Weighted average by validation R² ──────────────────────
    val_r2 = []
    for j in range(Xval.shape[1]):
        if np.var(yval) < 1e-9:
            val_r2.append(0.0)
        else:
            val_r2.append(max(0.0, r2_score(yval, Xval[:, j])))
    val_r2 = np.array(val_r2)
    if val_r2.sum() > 0:
        weights = val_r2 / val_r2.sum()
        yhat = Xtest @ weights
    else:
        yhat = Xtest.mean(axis=1)
        weights = np.ones(len(common_models)) / len(common_models)
    summary.append({"crop": crop, "method": "weighted_avg_valr2",
                    "n_base": len(common_models), **metrics(ytest, yhat)})
    for c, w in zip(common_models, weights):
        coefs.append({"crop": crop, "method": "weighted_avg_valr2",
                      "model": c, "coefficient": float(w)})

    # ── Method 5: Median ──────────────────────────────────────────────────
    yhat = np.median(Xtest, axis=1)
    summary.append({"crop": crop, "method": "median",
                    "n_base": len(common_models), **metrics(ytest, yhat)})

    # ── Baseline: best single by validation R² (proper holdout selection)
    best_idx = int(np.argmax(val_r2))
    yhat = Xtest[:, best_idx]
    summary.append({"crop": crop, "method": "best_single_by_val",
                    "n_base": 1,
                    **metrics(ytest, yhat),
                    "best_model": common_models[best_idx]})

    return summary, coefs


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exclude", default="",
                    help="Comma-separated models to exclude from BASE_MODELS "
                         "(e.g. 'tft' for the 5-model comparison run).")
    ap.add_argument("--suffix", default="",
                    help="Filename suffix on outputs (e.g. '_5models').")
    args = ap.parse_args()

    global BASE_MODELS
    excluded = {m.strip() for m in args.exclude.split(",") if m.strip()}
    if excluded:
        BASE_MODELS = [m for m in DEFAULT_BASE_MODELS if m not in excluded]

    cfg = load_data_config()
    val_years = cfg["splits"]["temporal"]["val_years"]
    test_years = cfg["splits"]["temporal"]["test_years"]
    crops = cfg["crops"]
    print(f"Val years:  {val_years}")
    print(f"Test years: {test_years}")
    print(f"Crops:      {crops}")
    print(f"Base models: {BASE_MODELS} (excluded: {sorted(excluded) or '-'})\n")

    print("=== Collecting val predictions ===")
    val_long = collect_predictions("val", val_years, crops)
    print("\n=== Collecting test predictions ===")
    test_long = collect_predictions("test", test_years, crops)
    if val_long.empty or test_long.empty:
        print("No predictions; aborting.")
        return

    val_wide = to_wide(val_long)
    test_wide = to_wide(test_long)

    print("\n=== Fusion ===")
    summary_rows: list[dict] = []
    coef_rows: list[dict] = []
    for crop in crops:
        s, c = fuse_per_crop(val_wide, test_wide, crop)
        summary_rows.extend(s)
        coef_rows.extend(c)

    if not summary_rows:
        print("No fusion results.")
        return

    summary_df = pd.DataFrame(summary_rows)
    out_dir = ROOT / "results" / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_dir / f"fusion_summary{args.suffix}.csv", index=False)
    coef_df = pd.DataFrame(coef_rows)
    coef_df.to_csv(out_dir / f"fusion_coefficients{args.suffix}.csv", index=False)
    print(f"\nWrote fusion_summary{args.suffix}.csv ({len(summary_df)} rows)")
    print(f"Wrote fusion_coefficients{args.suffix}.csv ({len(coef_df)} rows)")

    # ── Method leaderboard ───────────────────────────────────────────────
    method_lb = summary_df.groupby("method").agg(
        mean_r2=("r2", "mean"), median_r2=("r2", "median"),
        wins_above_0=("r2", lambda s: int((s > 0).sum())),
    ).sort_values("mean_r2", ascending=False)
    print("\n=== Method leaderboard (mean across 10 crops) ===")
    print(method_lb.round(3).to_string())


if __name__ == "__main__":
    main()

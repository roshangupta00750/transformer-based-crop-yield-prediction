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
from scipy.optimize import minimize, nnls
from sklearn.linear_model import ElasticNet, Ridge, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

try:
    import xgboost as xgb
    _HAVE_XGB = True
except ImportError:
    _HAVE_XGB = False

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
RIDGE_ALPHA_GRID = np.logspace(-3, 3, 13)  # 0.001 → 1000, 13 values
VRF_THRESHOLD = 0.0  # drop per-crop bases with val R² below this
TOP_K = 3            # top-K per crop for top_k_avg method
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


def _val_r2_per_model(Xval: np.ndarray, yval: np.ndarray) -> np.ndarray:
    if np.var(yval) < 1e-9:
        return np.zeros(Xval.shape[1])
    return np.array([r2_score(yval, Xval[:, j]) for j in range(Xval.shape[1])])


def _cv_ridge(Xval: np.ndarray, yval: np.ndarray,
              alphas: np.ndarray) -> tuple[float, Ridge]:
    """K-fold CV on val to pick alpha, refit on full val."""
    n = len(yval)
    k = min(5, max(2, n // 5))
    kf = KFold(n_splits=k, shuffle=True, random_state=0)
    best_alpha, best_mse = alphas[0], np.inf
    for a in alphas:
        mses = []
        for tr, te in kf.split(Xval):
            r = Ridge(alpha=a, fit_intercept=True)
            r.fit(Xval[tr], yval[tr])
            mses.append(mean_squared_error(yval[te], r.predict(Xval[te])))
        m = float(np.mean(mses))
        if m < best_mse:
            best_mse, best_alpha = m, float(a)
    ridge = Ridge(alpha=best_alpha, fit_intercept=True).fit(Xval, yval)
    return best_alpha, ridge


def _nnls_convex(Xval: np.ndarray, yval: np.ndarray) -> np.ndarray:
    """Non-negative weights summing to 1 (convex combination) — no intercept."""
    n_m = Xval.shape[1]
    x0 = np.ones(n_m) / n_m
    def loss(w):
        return float(np.mean((yval - Xval @ w) ** 2))
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    bnds = [(0.0, 1.0)] * n_m
    r = minimize(loss, x0, method="SLSQP", bounds=bnds, constraints=cons,
                 options={"maxiter": 200, "ftol": 1e-9})
    if r.success:
        return r.x
    # Fallback: uniform
    return x0


def _pooled_stack(val_wide: pd.DataFrame, test_wide: pd.DataFrame,
                  crops: list[str]) -> list[dict]:
    """Pooled cross-crop stacking. One meta-learner trained on all crops,
    reported per-crop for fair comparison."""
    v_all = val_wide.dropna(subset=BASE_MODELS + ["y_true"], how="any")
    t_all = test_wide.dropna(subset=BASE_MODELS + ["y_true"], how="any")
    v_all = v_all[v_all["crop"].isin(crops)]
    t_all = t_all[t_all["crop"].isin(crops)]
    if v_all.empty or t_all.empty:
        return []
    common = [m for m in BASE_MODELS if m in v_all.columns and m in t_all.columns]
    if len(common) < 2:
        return []

    def to_features(df: pd.DataFrame) -> pd.DataFrame:
        X = df[common].copy()
        for c in crops:
            X[f"crop_{c}"] = (df["crop"] == c).astype(int)
        return X

    Xv = to_features(v_all).to_numpy(dtype=np.float32)
    yv = v_all["y_true"].to_numpy()
    Xt = to_features(t_all).to_numpy(dtype=np.float32)

    # Two variants: crop-aware GBT (small depth, sees interactions) and
    # crop-aware Ridge (linear, safer for small val)
    rows: list[dict] = []
    try:
        meta = xgb.XGBRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
            reg_lambda=1.0, random_state=0, n_jobs=1, verbosity=0,
        )
        meta.fit(Xv, yv)
        preds = meta.predict(Xt)
        for crop in crops:
            mask = (t_all["crop"] == crop).to_numpy()
            if not mask.any():
                continue
            yt = t_all.loc[mask, "y_true"].to_numpy()
            yp = preds[mask]
            rows.append({"crop": crop, "method": "stack_xgb_pooled",
                         "n_base": len(common), **metrics(yt, yp)})
    except Exception as e:  # noqa: BLE001
        print(f"    stack_xgb_pooled failed: {e}")

    try:
        _, ridge_p = _cv_ridge(Xv, yv, RIDGE_ALPHA_GRID)
        preds = ridge_p.predict(Xt)
        for crop in crops:
            mask = (t_all["crop"] == crop).to_numpy()
            if not mask.any():
                continue
            yt = t_all.loc[mask, "y_true"].to_numpy()
            yp = preds[mask]
            rows.append({"crop": crop, "method": "stack_ridge_pooled",
                         "n_base": len(common), **metrics(yt, yp)})
    except Exception as e:  # noqa: BLE001
        print(f"    stack_ridge_pooled failed: {e}")

    return rows


def fuse_per_crop(val_wide: pd.DataFrame,
                  test_wide: pd.DataFrame,
                  crop: str) -> tuple[list[dict], list[dict]]:
    """Apply all fusion methods to one crop. Returns (summary_rows, coef_rows)."""
    v_all = val_wide[val_wide["crop"] == crop]
    t_all = test_wide[test_wide["crop"] == crop]
    # Only require completeness on the models we actually USE, not on every
    # column present in the cache (else a stale extra column like TFT would
    # silently drop rows in a 5-model run).
    used_cols_all = sorted(set(BASE_MODELS) & set(v_all.columns) & set(t_all.columns))
    if len(used_cols_all) < 2:
        print(f"  [{crop}] skip — only {len(used_cols_all)} models with both splits")
        return [], []
    v = v_all.dropna(subset=used_cols_all + ["y_true"])
    t = t_all.dropna(subset=used_cols_all + ["y_true"])
    common_models = used_cols_all

    Xval_full = v[common_models].to_numpy()
    yval = v["y_true"].to_numpy()
    Xtest_full = t[common_models].to_numpy()
    ytest = t["y_true"].to_numpy()

    # Per-crop val R² → used for weighting AND filtering
    val_r2_full = _val_r2_per_model(Xval_full, yval)

    # Adaptive filter: drop models with val R² < VRF_THRESHOLD.
    # If fewer than 2 models survive, fall back to the top-2 by val R².
    keep = np.where(val_r2_full > VRF_THRESHOLD)[0]
    if len(keep) < 2:
        keep = np.argsort(-val_r2_full)[:2]
    filt_models = [common_models[i] for i in keep]
    Xval = Xval_full[:, keep]
    Xtest = Xtest_full[:, keep]
    val_r2 = val_r2_full[keep]

    print(f"  [{crop}] val={len(v)}, test={len(t)}, "
          f"models_full={common_models}, models_filt={filt_models}")

    summary: list[dict] = []
    coefs: list[dict] = []

    def add(method: str, yhat: np.ndarray, n_base: int, extra: dict | None = None):
        row = {"crop": crop, "method": method, "n_base": n_base,
               **metrics(ytest, yhat)}
        if extra:
            row.update(extra)
        summary.append(row)

    def add_coefs(method: str, models: list[str], weights: np.ndarray,
                  intercept: float | None = None):
        for c, w in zip(models, weights):
            coefs.append({"crop": crop, "method": method,
                          "model": c, "coefficient": float(w)})
        if intercept is not None:
            coefs.append({"crop": crop, "method": method,
                          "model": "_intercept_", "coefficient": float(intercept)})

    # =====================================================================
    # BASELINE METHODS (unchanged — computed on ALL models, no VRF filter)
    # =====================================================================

    # Ridge alpha=1 (legacy baseline)
    ridge = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True).fit(Xval_full, yval)
    add("ridge_alpha1", ridge.predict(Xtest_full), len(common_models))
    add_coefs("ridge_alpha1", common_models, ridge.coef_, float(ridge.intercept_))

    # NNLS with intercept (legacy)
    Xv_nn = np.hstack([Xval_full, np.ones((len(Xval_full), 1))])
    Xt_nn = np.hstack([Xtest_full, np.ones((len(Xtest_full), 1))])
    w_nn, _ = nnls(Xv_nn, yval)
    add("nnls", Xt_nn @ w_nn, len(common_models))
    add_coefs("nnls", common_models, w_nn[:-1], float(w_nn[-1]))

    # Simple average (legacy)
    add("simple_avg", Xtest_full.mean(axis=1), len(common_models))

    # Weighted average by val R² (legacy: clipped to ≥0)
    wval = np.maximum(0.0, val_r2_full)
    if wval.sum() > 0:
        weights = wval / wval.sum()
        yhat = Xtest_full @ weights
    else:
        yhat = Xtest_full.mean(axis=1)
        weights = np.ones(len(common_models)) / len(common_models)
    add("weighted_avg_valr2", yhat, len(common_models))
    add_coefs("weighted_avg_valr2", common_models, weights)

    # Median (legacy)
    add("median", np.median(Xtest_full, axis=1), len(common_models))

    # Best single by val R² (baseline)
    best_idx = int(np.argmax(val_r2_full))
    add("best_single_by_val", Xtest_full[:, best_idx], 1,
        {"best_model": common_models[best_idx]})

    # =====================================================================
    # ENHANCED METHODS
    # =====================================================================

    # Ridge CV — grid-search alpha per crop
    alpha_best, ridge_cv = _cv_ridge(Xval_full, yval, RIDGE_ALPHA_GRID)
    add("ridge_cv", ridge_cv.predict(Xtest_full), len(common_models),
        {"best_model": f"alpha={alpha_best:.4g}"})
    add_coefs("ridge_cv", common_models, ridge_cv.coef_, float(ridge_cv.intercept_))

    # ElasticNet CV — L1+L2 combo, sparse selection
    try:
        from sklearn.linear_model import ElasticNetCV
        en = ElasticNetCV(l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
                          alphas=np.logspace(-3, 1, 20),
                          cv=min(5, max(2, len(yval)//5)),
                          fit_intercept=True, max_iter=5000, random_state=0)
        en.fit(Xval_full, yval)
        add("elasticnet_cv", en.predict(Xtest_full), len(common_models),
            {"best_model": f"alpha={en.alpha_:.3g},l1={en.l1_ratio_:.2f}"})
        add_coefs("elasticnet_cv", common_models, en.coef_, float(en.intercept_))
    except Exception as e:  # noqa: BLE001
        print(f"    elasticnet failed: {e}")

    # VRF + simple average — mean over positively-scoring models only
    add("vrf_simple_avg", Xtest.mean(axis=1), len(filt_models))

    # VRF + NNLS
    Xv_f = np.hstack([Xval, np.ones((len(Xval), 1))])
    Xt_f = np.hstack([Xtest, np.ones((len(Xtest), 1))])
    w_f, _ = nnls(Xv_f, yval)
    add("vrf_nnls", Xt_f @ w_f, len(filt_models))
    add_coefs("vrf_nnls", filt_models, w_f[:-1], float(w_f[-1]))

    # VRF + Ridge CV
    if len(filt_models) >= 2:
        _, ridge_vrf = _cv_ridge(Xval, yval, RIDGE_ALPHA_GRID)
        add("vrf_ridge_cv", ridge_vrf.predict(Xtest), len(filt_models))
        add_coefs("vrf_ridge_cv", filt_models, ridge_vrf.coef_,
                  float(ridge_vrf.intercept_))

    # Convex combination (NNLS with sum=1, no intercept) — well-calibrated
    w_cvx = _nnls_convex(Xval_full, yval)
    add("convex_nnls", Xtest_full @ w_cvx, len(common_models))
    add_coefs("convex_nnls", common_models, w_cvx)

    # Trimmed mean — drop per-sample min AND max before averaging (robust to
    # a single catastrophic model like TFT on summer_barley)
    if len(common_models) >= 4:
        sorted_test = np.sort(Xtest_full, axis=1)
        yhat = sorted_test[:, 1:-1].mean(axis=1)
        add("trimmed_mean_1", yhat, len(common_models))

    # Top-K by val R² average
    if len(common_models) >= TOP_K:
        top_idx = np.argsort(-val_r2_full)[:TOP_K]
        yhat = Xtest_full[:, top_idx].mean(axis=1)
        add(f"top{TOP_K}_avg", yhat, TOP_K,
            {"best_model": ",".join(common_models[i] for i in top_idx)})

    # Stacking meta-learner — small XGBoost on val→test (linear + non-linear)
    if _HAVE_XGB:
        try:
            meta = xgb.XGBRegressor(
                n_estimators=200, max_depth=3, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
                reg_lambda=1.0, random_state=0, n_jobs=1, verbosity=0,
            )
            meta.fit(Xval_full, yval)
            add("stack_xgb", meta.predict(Xtest_full), len(common_models))
            add_coefs("stack_xgb", common_models,
                      np.array(meta.feature_importances_))
        except Exception as e:  # noqa: BLE001
            print(f"    stack_xgb failed: {e}")

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

    # ── Cross-crop pooled stacking: single XGBoost meta-learner on ALL crops
    # simultaneously, with per-crop bias correction + one-hot crop features.
    # Effective val n = ~300+ samples (vs ~30-50 per-crop), so the meta model
    # can actually learn.
    if _HAVE_XGB and not val_wide.empty and not test_wide.empty:
        summary_rows.extend(_pooled_stack(val_wide, test_wide, crops))

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

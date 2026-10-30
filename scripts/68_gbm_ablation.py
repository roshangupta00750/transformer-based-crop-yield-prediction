"""Pre-registration v16 §2 arm G16: gradient boosting on YieldFormer-DE's inputs, one rolling fold.

Uses the fitted Q12 feature pipeline of fold Y (seed 42), so features, scaling, district
anomalies, the target framing (z relative to y_hist) and y_hist are the network's. Each
past-observed feature is averaged over 8 equal segments of the crop-year sequence; static
features and the crop code are added. Train <= Y-2 (reliable series), early stop on Y-1, test Y.
Hyperparameters fixed in v16 §2 (no tuning). Seeds 42/43/44, forecast = their mean.

Output: results/metrics/v16_gbm/G16_<Y>.parquet (year, crop, landkreis, y_true, y_hist, G16, n)

Usage:  python scripts/68_gbm_ablation.py --test-year 2015 [--device cuda]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.data.dataset import FeaturePipeline, load_modeling_frame, reliable_series  # noqa: E402
from src.utils import load_data_config, load_model_config  # noqa: E402

SEGMENTS = 8
SEEDS = (42, 43, 44)
PARAMS = dict(tree_method="hist", max_depth=6, learning_rate=0.03, n_estimators=3000,
              early_stopping_rounds=100, subsample=0.8, colsample_bytree=0.8,
              min_child_weight=5, reg_lambda=1.0, objective="reg:squarederror")
KEY = ["landkreis", "crop", "year"]


def tabular(raw: pd.DataFrame, pipe: FeaturePipeline, crops: list[str]) -> pd.DataFrame:
    """One row per district-crop-year: 8 segment means per past feature + statics + crop code + z."""
    t = pipe.transform_features(raw).sort_values(KEY + ["date"])
    spec = pipe.spec
    past = list(spec.past_observed)
    pos = t.groupby(KEY, sort=False).cumcount()
    size = t.groupby(KEY, sort=False)["date"].transform("size")
    t["_seg"] = np.minimum((pos * SEGMENTS // size).to_numpy(), SEGMENTS - 1)
    seg = t.groupby(KEY + ["_seg"], sort=False)[past].mean().unstack("_seg")
    seg.columns = [f"{c}__s{s}" for c, s in seg.columns]
    first = t.groupby(KEY, sort=False)[list(spec.static) + [spec.target]].first()
    X = seg.join(first).reset_index()
    X["crop_code"] = X["crop"].map({c: i for i, c in enumerate(crops)}).astype("category")
    X["z"] = [pipe.transform_target(lk, cr, y, yr) for lk, cr, y, yr
              in zip(X["landkreis"], X["crop"], X[spec.target], X["year"])]
    return X


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--test-year", type=int, required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    import xgboost as xgb

    Y = args.test_year
    cfg = load_model_config("itransformer_v12Q")
    dc = load_data_config()
    frame = load_modeling_frame(ROOT / cfg["dataset"], dc, window="crop_year", common_sample=True,
                                cache_dir=ROOT / "data/processed/cache", phenology=cfg["phenology"],
                                phenology_version=cfg["phenology_version"],
                                stage_features=bool(cfg["stage_features"]))
    pipe = FeaturePipeline.load(ROOT / f"results/checkpoints/itransformer_v12Q_roll{Y}_s42.pipeline.json")
    pipe.require_current_feature_policy()
    crops = sorted(frame["crop"].unique())

    tr = reliable_series(frame[frame["year"] <= Y - 2])
    parts = {k: tabular(f, pipe, crops) for k, f in
             (("train", tr), ("val", frame[frame["year"] == Y - 1]), ("test", frame[frame["year"] == Y]))}
    feats = [c for c in parts["train"].columns if c not in KEY + ["z", pipe.spec.target]]
    print(f"fold {Y}: train {len(parts['train']):,}, val {len(parts['val']):,}, test {len(parts['test']):,}, "
          f"{len(feats)} features", flush=True)

    zs = []
    for s in SEEDS:
        m = xgb.XGBRegressor(**PARAMS, random_state=s, device=args.device, enable_categorical=True)
        m.fit(parts["train"][feats], parts["train"]["z"],
              eval_set=[(parts["val"][feats], parts["val"]["z"])], verbose=False)
        zs.append(m.predict(parts["test"][feats], iteration_range=(0, m.best_iteration + 1)))
        print(f"  seed {s}: best iteration {m.best_iteration}, val rmse(z) {m.best_score:.4f}", flush=True)

    T = parts["test"]
    lk, cr, yr = T["landkreis"].to_numpy(), T["crop"].to_numpy(), T["year"].to_numpy()
    out = pd.DataFrame({"year": yr, "crop": cr, "landkreis": lk,
                        "y_true": T[pipe.spec.target].to_numpy(),
                        "y_hist": pipe.inverse_transform_array(lk, cr, np.zeros(len(T)), yr),
                        "G16": pipe.inverse_transform_array(lk, cr, np.mean(zs, axis=0), yr),
                        "n": len(SEEDS)})
    d = ROOT / "results/metrics/v16_gbm"
    d.mkdir(parents=True, exist_ok=True)
    out.to_parquet(d / f"G16_{Y}.parquet", index=False)
    acc = 100 - 100 * np.mean(np.abs(out.G16 - out.y_true) / out.y_true)
    acc_h = 100 - 100 * np.mean(np.abs(out.y_hist - out.y_true) / out.y_true)
    print(f"fold {Y}: G16 accuracy {acc:.2f}%  (normal {acc_h:.2f}%), {len(out):,} rows")


if __name__ == "__main__":
    main()

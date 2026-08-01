"""XGBoost on engineered monthly aggregates.

For each (Landkreis, crop, year) sample, builds a flat feature vector:
  - mean / std / min / max of each weather variable over each month
  - mean of each satellite index over each month
  - static soil properties
That vector goes straight into XGBoost. No deep learning needed.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

from .base import BaseModel
from .registry import register


def _flatten_batch(batch, n_months: int = 12) -> tuple[np.ndarray, np.ndarray]:
    """Convert a batch of variable-length time-series tensors into tabular form."""
    B = batch["x_past"].shape[0]
    mask = batch.get("mask")
    feats = []
    for i in range(B):
        past = batch["x_past"][i].cpu().numpy()
        static = batch["x_static"][i].cpu().numpy()
        # Bucket the VALID span, not the padded one. collate_pad pads every
        # sample to the longest phenology window (220 days), so bucketing the
        # padded axis at fixed absolute indices left the trailing buckets
        # identically zero for short-season crops — 320 of 974 features dead
        # for summer_barley and oats, which could reach only 55.7 % of the
        # booster's leaves. It also made bucket k mean a different calendar
        # day per crop, inside a pooled model with no crop identifier.
        # Slicing first makes bucket k the same relative phenological phase
        # for every crop.
        if mask is not None:
            valid = int(mask[i].sum().item())
            if valid > 0:
                past = past[:valid]
        # Aggregate into 12 equal-length phenological buckets
        chunks = np.array_split(past, n_months, axis=0)
        agg = np.concatenate([
            np.concatenate([c.mean(0), c.std(0), c.min(0), c.max(0)])
            for c in chunks
        ])
        feats.append(np.concatenate([agg, static]))
    X = np.stack(feats)
    y = batch["y"].cpu().numpy()
    return X, y


def loader_to_xy(loader) -> tuple[np.ndarray, np.ndarray]:
    Xs, ys = [], []
    for batch in loader:
        X, y = _flatten_batch(batch)
        Xs.append(X); ys.append(y)
    return np.concatenate(Xs), np.concatenate(ys)


@register
class XGBoostModel(BaseModel):
    name = "xgboost"

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self.model: xgb.XGBRegressor | None = None

    def fit(self, train_loader, val_loader=None, **_):
        Xtr, ytr = loader_to_xy(train_loader)
        eval_set = None
        if val_loader is not None:
            Xv, yv = loader_to_xy(val_loader)
            eval_set = [(Xv, yv)]

        self.model = xgb.XGBRegressor(
            n_estimators=self.config["n_estimators"],
            max_depth=self.config["max_depth"],
            learning_rate=self.config["learning_rate"],
            subsample=self.config["subsample"],
            colsample_bytree=self.config["colsample_bytree"],
            objective=self.config["objective"],
            early_stopping_rounds=self.config.get("early_stopping_rounds"),
            tree_method="hist",
            n_jobs=self.config.get("n_jobs", 4),
        )
        self.model.fit(Xtr, ytr, eval_set=eval_set, verbose=False)
        return self

    def predict(self, loader) -> np.ndarray:
        X, _ = loader_to_xy(loader)
        return self.model.predict(X)

    def save(self, path):
        # XGBoost picks the on-disk format from the extension; force .ubj to
        # avoid the "Unknown file format" warning and (worse) loader segfaults.
        from pathlib import Path
        p = Path(path)
        if p.suffix.lower() not in (".ubj", ".json"):
            p = p.with_suffix(".ubj")
        self.model.save_model(str(p))

    def load(self, path):
        from pathlib import Path
        p = Path(path)
        if p.suffix.lower() not in (".ubj", ".json"):
            cand = p.with_suffix(".ubj")
            if cand.exists():
                p = cand
        # Force a fixed thread count on load — n_jobs=-1 segfaults predict()
        # in libomp on some macOS builds.
        self.model = xgb.XGBRegressor(n_jobs=self.config.get("n_jobs", 4))
        self.model.load_model(str(p))
        return self

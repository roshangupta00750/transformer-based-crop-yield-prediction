"""Run the same model across temporal/spatial holdouts and in-season horizons.

Metrics are always computed in absolute-yield space (dt/ha) — the model
output (z-scored anomaly) is inverse-transformed via the FeaturePipeline
before scoring, so R²/RMSE/MAE are directly interpretable.
"""
from __future__ import annotations

import json
from functools import partial
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

from ..data import CropYieldDataset
from ..data.dataset import FeaturePipeline, FeatureSpec, collate_pad
from ..models import BaseModel
from .metrics import all_metrics


def evaluate_at_horizons(
    model: BaseModel,
    df_test,
    feature_spec: FeatureSpec,
    pipeline: FeaturePipeline,
    harvest_doy: int,
    horizons_days: list[int],
    batch_size: int = 128,
    sequence_length: int | None = None,
) -> dict[str, dict]:
    """Truncate test sequences at (harvest_doy − h) and score in dt/ha space."""
    df_test_t = pipeline.transform_features(df_test)
    results: dict[str, dict] = {}

    for h in horizons_days:
        cutoff = max(harvest_doy - h, 1)
        ds = CropYieldDataset(df_test_t, feature_spec, pipeline, cutoff_doy=cutoff)
        if len(ds) == 0:
            continue
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            collate_fn=partial(collate_pad, pad_to=sequence_length),
        )
        z_pred = np.asarray(model.predict(loader)).ravel()

        lk = np.array([k[0] for k in ds.keys])
        cr = np.array([k[1] for k in ds.keys])
        y_pred_abs = pipeline.inverse_transform_array(lk, cr, z_pred)
        y_true_abs = pipeline.inverse_transform_observed_array(
            lk, cr, np.asarray(ds._targets_z)
        )

        results[f"h-{h}d"] = all_metrics(y_true_abs, y_pred_abs)
    return results


def save_results(results: dict, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    return out_path

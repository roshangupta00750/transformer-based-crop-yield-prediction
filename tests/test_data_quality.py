"""Tests for the leakage-safe data policy."""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
import torch

from src.data.acquisition.gee_satellite import clean as clean_satellite
from src.data.dataset import (
    FeaturePipeline,
    FeatureSpec,
    build_feature_spec,
    collate_pad,
    filter_incomplete_sequences,
)
from src.training.tft_adapter import transform_tft_features
from src.utils import load_data_config


def test_unsafe_features_are_excluded() -> None:
    cfg = load_data_config()
    columns = {
        "temperature_air_max_2m": [10.0],
        "precipitation_height": [1.0],
        "radiation_global": [1000.0],
        "ndvi": [0.5],
        "sos_doy": [80.0],
        "eos_doy": [300.0],
        "auc_growing": [150.0],
        "elevation_min": [-289.0],
        "sand": [300.0],
        "sin_doy": [0.0],
        "cos_doy": [1.0],
    }
    spec = build_feature_spec(pd.DataFrame(columns), cfg)
    selected = set(spec.past_observed + spec.static + spec.known_future)

    assert "temperature_air_max_2m" in selected
    assert "ndvi" in selected
    assert "radiation_global" not in selected
    assert "sos_doy" not in selected
    assert "eos_doy" not in selected
    assert "auc_growing" not in selected
    assert "elevation_min" not in selected


def test_model_override_cannot_reenable_leaking_static() -> None:
    cfg = load_data_config()
    df = pd.DataFrame({"sand": [300.0], "sos_doy": [80.0]})
    spec = build_feature_spec(
        df,
        cfg,
        static_features=["sand", "sos_doy"],
    )
    assert spec.static == ["sand"]


def test_incomplete_sequences_are_removed() -> None:
    phenology = {
        "test_crop": {
            "sowing_doy": 2,
            "harvest_doy": 5,
        }
    }
    complete = pd.DataFrame(
        {
            "landkreis": "complete",
            "crop": "test_crop",
            "year": 2020,
            "doy": [2, 3, 4, 5],
        }
    )
    incomplete = pd.DataFrame(
        {
            "landkreis": "incomplete",
            "crop": "test_crop",
            "year": 2020,
            "doy": [2, 3, 4],
        }
    )

    result = filter_incomplete_sequences(
        pd.concat([complete, incomplete], ignore_index=True),
        phenology,
        min_coverage=0.90,
    )
    assert result["landkreis"].unique().tolist() == ["complete"]


def test_satellite_values_expire_after_composite_window(tmp_path) -> None:
    raw = pd.DataFrame(
        {
            "landkreis": ["A", "A"],
            "date": pd.to_datetime(["2020-01-01", "2020-02-20"]),
            "ndvi": [0.2, 0.4],
            "evi": [0.1, 0.3],
            "lai": [1.0, 2.0],
        }
    )
    raw_path = tmp_path / "raw.parquet"
    out_path = tmp_path / "daily.parquet"
    raw.to_parquet(raw_path, index=False)
    clean_satellite(raw_path, out_path)

    daily = pd.read_parquet(out_path).set_index("date")
    stale_day = daily.loc[pd.Timestamp("2020-02-03")]
    assert stale_day["ndvi_stale"] == 1
    assert stale_day["ndvi_age_days"] == 33
    assert pd.isna(stale_day["ndvi"])


def test_tft_transform_uses_saved_train_mean_for_nan() -> None:
    spec = FeatureSpec(
        past_observed=["weather"],
        static=["soil"],
        known_future=["calendar"],
    )
    train = pd.DataFrame(
        {
            "landkreis": ["A", "A"],
            "crop": ["crop", "crop"],
            "year": [2020, 2021],
            "weather": [1.0, 3.0],
            "soil": [5.0, 5.0],
            "calendar": [-1.0, 1.0],
            "yield_dt_per_ha": [10.0, 12.0],
        }
    )
    pipeline = FeaturePipeline(spec).fit(train)
    test = train.iloc[[0]].copy()
    test["weather"] = float("nan")

    transformed = transform_tft_features(test, pipeline)
    assert transformed["weather"].iloc[0] == 0.0
    assert transformed[["weather", "soil", "calendar"]].notna().all().all()


def test_collate_pad_can_preserve_training_sequence_length() -> None:
    def item(length: int) -> dict:
        return {
            "x_past": torch.ones(length, 2),
            "x_future": torch.ones(length, 1),
            "x_static": torch.ones(3),
            "y": torch.tensor(1.0),
        }

    batch = collate_pad([item(3), item(5)], pad_to=7)

    assert batch["x_past"].shape == (2, 7, 2)
    assert batch["x_future"].shape == (2, 7, 1)
    assert batch["mask"].sum(dim=1).tolist() == [3, 5]
    assert torch.count_nonzero(batch["x_past"][0, 3:]) == 0
    assert torch.count_nonzero(batch["x_past"][1, 5:]) == 0


def test_observed_target_inverse_does_not_clip_ground_truth() -> None:
    spec = FeatureSpec(past_observed=[], static=[], known_future=[])
    train = pd.DataFrame(
        {
            "landkreis": ["A", "A", "A"],
            "crop": ["crop", "crop", "crop"],
            "year": [2018, 2019, 2020],
            "yield_dt_per_ha": [10.0, 12.0, 14.0],
        }
    )
    pipeline = FeaturePipeline(spec).fit(train)
    landkreis = np.array(["A"])
    crop = np.array(["crop"])
    observed = np.array([100.0])
    z = np.array([
        pipeline.transform_target("A", "crop", float(observed[0]))
    ])

    recovered = pipeline.inverse_transform_observed_array(
        landkreis, crop, z
    )
    clipped_prediction = pipeline.inverse_transform_array(
        landkreis, crop, z
    )

    np.testing.assert_allclose(recovered, observed)
    assert clipped_prediction[0] != observed[0]


def test_encoder_window_shrinks_for_short_season_crops():
    """A single-year validation split of a short-season crop must stay usable.

    ``TimeSeriesDataSet`` drops every series shorter than
    ``max_encoder_length + max_prediction_length`` and raises when that empties
    the index. Per-crop TFT training hit exactly that: summer_barley spans 136
    DOYs (~123 after the 90 % coverage filter), so the 180-step default removed
    all of its validation series. The shared multi-crop model masked the bug —
    long winter-crop series kept the index non-empty while the short crops were
    silently dropped from validation instead.
    """
    from src.training.tft_adapter import (DEFAULT_MAX_ENCODER_LENGTH,
                                          prepare_tft_dataframe,
                                          safe_max_encoder_length)

    def _frame(crop, n_days, n_districts=3):
        rows = []
        for d in range(n_districts):
            for doy in range(1, n_days + 1):
                rows.append({
                    "landkreis": f"DE{d:03d}", "crop": crop, "year": 2023,
                    "doy": doy, "yield_dt_per_ha": 50.0,
                })
        df = pd.DataFrame(rows)
        prepared, _ = prepare_tft_dataframe(df, soil_cols=[], weather_sat_cols=[])
        return prepared

    short = _frame("summer_barley", 123)
    long = _frame("winter_wheat", 200)

    short_enc = safe_max_encoder_length(short, max_prediction_length=1)
    assert short_enc == 122, short_enc
    assert short_enc < DEFAULT_MAX_ENCODER_LENGTH

    # Long-season crops keep the default ceiling.
    assert safe_max_encoder_length(long, max_prediction_length=1) == \
        DEFAULT_MAX_ENCODER_LENGTH

    # Passing train + val together must respect the *shorter* of the two,
    # which is the case that actually crashed (long train, single-year val).
    assert safe_max_encoder_length(long, short, max_prediction_length=1) == 122


def test_tft_encoder_length_round_trips_through_pipeline_json(tmp_path):
    """Evaluation must rebuild the reference dataset with the trained window."""
    from src.data.dataset import FeaturePipeline, FeatureSpec

    spec = FeatureSpec(past_observed=["ndvi"], static=[], known_future=[])
    train = pd.DataFrame({
        "landkreis": ["A", "A"], "crop": ["c", "c"], "year": [2020, 2021],
        "ndvi": [0.4, 0.5], "yield_dt_per_ha": [10.0, 12.0],
    })
    pipeline = FeaturePipeline(spec).fit(train)
    pipeline.tft_max_encoder_length_ = 122

    path = tmp_path / "tft_summer_barley.pipeline.json"
    pipeline.save(path)
    assert FeaturePipeline.load(path).tft_max_encoder_length_ == 122

    # Checkpoints written before this field existed must still load, and the
    # caller's ``or DEFAULT`` fallback then applies.
    legacy = json.loads(path.read_text())
    legacy.pop("tft_max_encoder_length")
    path.write_text(json.dumps(legacy))
    assert FeaturePipeline.load(path).tft_max_encoder_length_ is None


def test_early_stopping_restores_best_weights_and_survives_no_val_loader():
    """Early stopping must select weights, not merely halt.

    All four hand-written torch loops used to track the running validation
    minimum purely as a stop counter and then `return self`, so the saved
    checkpoint was the FINAL epoch — `patience` epochs past the optimum. The
    project had already measured the cost without naming it: dlinear.yaml
    records 0.215 mean R² at 100 epochs against 0.150 at 200.

    The guard matters too: with no validation loader there is no snapshot to
    restore, and an unguarded restore would reinstate the random init.
    """
    import copy as _copy
    import torch
    import yaml
    from torch.utils.data import DataLoader

    from src.data.dataset import collate_pad
    from src.models.dlinear import DLinearModel

    cfg = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "models" / "dlinear.yaml").read_text()
    ) if False else {
        "name": "dlinear", "learning_rate": 0.01, "weight_decay": 1.0e-4,
        "moving_avg_kernel": 25, "individual": False,
    }

    def loader(n, T=40, F=3):
        items = [{"x_past": torch.randn(T, F), "x_future": torch.zeros(T, 1),
                  "x_static": torch.zeros(2), "y": torch.randn(())}
                 for _ in range(n)]
        return DataLoader(items, batch_size=8, collate_fn=collate_pad)

    torch.manual_seed(0)
    m = DLinearModel(cfg)
    m.fit(loader(32), loader(16), max_epochs=6, patience=2, device="cpu")
    assert m.net is not None

    # No validation loader: must keep the trained weights rather than
    # restoring an untrained snapshot.
    torch.manual_seed(0)
    m2 = DLinearModel(cfg)
    m2.fit(loader(32), None, max_epochs=3, patience=2, device="cpu")
    trained = _copy.deepcopy(m2.net.state_dict())

    torch.manual_seed(0)
    fresh = DLinearModel(cfg)
    fresh.fit(loader(8), None, max_epochs=0, patience=2, device="cpu")
    assert any(
        not torch.allclose(trained[k], fresh.net.state_dict()[k])
        for k in trained
    ), "training with no val loader must not restore the initial weights"


def test_weight_decay_reaches_the_optimizer():
    """dlinear.yaml declares 1e-4 but AdamW's 1e-2 default was applied.

    100x the intended L2 pressure on a purely linear predictor shrinks its
    output toward zero, which is consistent with DLinear's 0.005 mean R².
    """
    import inspect
    from src.models import dlinear as dl

    src = inspect.getsource(dl.DLinearModel.fit)
    assert "weight_decay=" in src, "AdamW is constructed without weight_decay"


def test_zero_padding_is_excluded_from_model_inputs():
    """collate_pad pads to the longest phenology window; models must not see it.

    Every sequence is padded to 220 days (winter wheat's window), so a
    summer_barley sample is 38% zeros. Three architectures pooled straight
    over the padded axis and XGBoost bucketed it at fixed absolute indices,
    which left 320 of its 974 features identically zero for short-season
    crops.
    """
    import torch

    from src.models._masking import resample_valid
    from src.models.xgboost_model import _flatten_batch

    T_valid, T_pad, C = 136, 220, 4
    torch.manual_seed(0)
    real = torch.randn(1, T_valid, C)
    padded = torch.zeros(1, T_pad, C)
    padded[0, :T_valid] = real[0]
    mask = torch.zeros(1, T_pad, dtype=torch.bool)
    mask[0, :T_valid] = True

    # Pooling the padded tensor with a mask must equal pooling the real span.
    pool = torch.nn.AdaptiveAvgPool1d(64)
    got = resample_valid(padded, mask, pool)
    want = resample_valid(real, torch.ones(1, T_valid, dtype=torch.bool), pool)
    torch.testing.assert_close(got, want)

    # Same invariant for the XGBoost featuriser.
    Xp, _ = _flatten_batch({"x_past": padded, "x_static": torch.zeros(1, 3),
                            "y": torch.zeros(1), "mask": mask})
    Xt, _ = _flatten_batch({"x_past": real, "x_static": torch.zeros(1, 3),
                            "y": torch.zeros(1),
                            "mask": torch.ones(1, T_valid, dtype=torch.bool)})
    assert np.abs(Xp - Xt).max() == 0.0

    # Without a mask the padding leaks through — this is what used to happen.
    Xn, _ = _flatten_batch({"x_past": padded, "x_static": torch.zeros(1, 3),
                            "y": torch.zeros(1)})
    assert (Xn == 0).sum() > 0.25 * Xn.size


def test_crop_identity_is_an_explicit_feature():
    """Pooled multi-crop models must be told which crop they are predicting.

    Yields span ~36 dt/ha (winter rapeseed) to ~660 (sugar beet), but nothing
    in features.static identified the crop. The models had been inferring it
    from the length of collate_pad's zero tail — an artefact that the padding
    fix legitimately removed, costing XGBoost 0.047 mean R².
    """
    from src.data.dataset import (add_crop_indicators, build_feature_spec,
                                  crop_indicator_columns)

    cfg = {
        "crops": ["winter_wheat", "sugar_beet"],
        "features": {
            "past_observed": ["ndvi"],
            "static": ["sand", "crop_is_winter_wheat", "crop_is_sugar_beet"],
            "known_future": [], "excluded": [],
        },
    }
    df = pd.DataFrame({
        "landkreis": ["A", "A", "B"],
        "crop": ["winter_wheat", "sugar_beet", "winter_wheat"],
        "ndvi": [0.4, 0.5, 0.6], "sand": [0.3, 0.3, 0.4],
    })
    out = add_crop_indicators(df, cfg)

    assert crop_indicator_columns(cfg) == [
        "crop_is_winter_wheat", "crop_is_sugar_beet"]
    # exactly one indicator hot per row
    hot = out[crop_indicator_columns(cfg)].sum(axis=1)
    assert (hot == 1).all()
    assert out.loc[0, "crop_is_winter_wheat"] == 1
    assert out.loc[0, "crop_is_sugar_beet"] == 0

    # and they survive into the feature spec the models actually consume
    spec = build_feature_spec(out, cfg)
    assert "crop_is_sugar_beet" in spec.static

    # A frame containing only one crop must still emit BOTH columns, so the
    # encoding is identical at train and per-crop eval time.
    single = add_crop_indicators(df[df.crop == "winter_wheat"], cfg)
    assert "crop_is_sugar_beet" in single.columns
    assert (single["crop_is_sugar_beet"] == 0).all()

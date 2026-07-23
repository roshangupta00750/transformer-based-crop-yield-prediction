"""Temporal Fusion Transformer (Lim et al., 2021).

Wraps the `pytorch-forecasting` implementation behind our unified
`BaseModel` interface. Because TFT requires its own DataLoader format,
we accept either:

  (a) a standard DataLoader producing dicts (the common case) — we
      reconstruct the underlying DataFrame from `loader.dataset._groups`
      and build a TimeSeriesDataSet on the fly, **or**

  (b) a raw `pd.DataFrame` directly passed in via `fit_from_df(...)` —
      this is the recommended path for production runs.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .base import BaseModel
from .registry import register


@register
class TFTModel(BaseModel):
    name = "tft"

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self.model = None
        self.trainer = None
        self.training_dataset = None
        self.device = "cpu"

    # ----------------------------------------------------------------- #
    # Recommended entry point — pass DataFrames directly
    # ----------------------------------------------------------------- #
    def fit_from_df(
        self,
        df_train,
        df_val=None,
        *,
        soil_cols: list[str],
        weather_sat_cols: list[str],
        target_col: str = "yield_dt_per_ha",
        max_encoder_length: int = 180,
        max_epochs: int = 50,
        device: str = "cpu",
        max_minutes: float | None = None,
    ):
        try:
            import torch
            # pytorch-forecasting inherits from `lightning.pytorch`, not the
            # legacy `pytorch_lightning` namespace. Using the wrong one makes
            # Trainer.fit() reject the model with a confusing isinstance error.
            import lightning.pytorch as pl
            from lightning.pytorch.loggers import CSVLogger
            from pytorch_forecasting import TemporalFusionTransformer
            from pytorch_forecasting.metrics import MAE, QuantileLoss
            from torch.utils.data import DataLoader
        except ImportError as e:
            raise ImportError(
                "Install pytorch-forecasting + lightning"
            ) from e

        # Tensor-core matmul precision — Lightning warns without this on Ampere+
        torch.set_float32_matmul_precision("high")

        from ..training.tft_adapter import (
            build_timeseries_dataset, prepare_tft_dataframe,
        )

        df_tr, cols = prepare_tft_dataframe(
            df_train, soil_cols=soil_cols,
            weather_sat_cols=weather_sat_cols, target_col=target_col,
        )
        ds_tr = build_timeseries_dataset(
            df_tr, cols, max_encoder_length=max_encoder_length,
            max_prediction_length=1,
        )
        self.training_dataset = ds_tr
        train_loader = ds_tr.to_dataloader(
            train=True, batch_size=self.config.get("batch_size", 128),
            num_workers=4, persistent_workers=True, pin_memory=True,
        )

        val_loader = None
        if df_val is not None:
            df_v, _ = prepare_tft_dataframe(
                df_val, soil_cols=soil_cols,
                weather_sat_cols=weather_sat_cols, target_col=target_col,
            )
            ds_v = build_timeseries_dataset(
                df_v, cols, max_encoder_length=max_encoder_length,
                max_prediction_length=1,
            )
            val_loader = ds_v.to_dataloader(
                train=False, batch_size=self.config.get("batch_size", 128),
                num_workers=4, persistent_workers=True, pin_memory=True,
            )

        loss_name = str(self.config.get("loss", "mae")).lower()
        if loss_name == "quantile":
            quantiles = [0.1, 0.5, 0.9]
            loss_fn = QuantileLoss(quantiles=quantiles)
            output_size = self.config.get("output_size", len(quantiles))
        else:
            loss_fn = MAE()
            output_size = int(self.config.get("output_size", 1))
        self.model = TemporalFusionTransformer.from_dataset(
            ds_tr,
            hidden_size=self.config["hidden_size"],
            attention_head_size=self.config["attention_head_size"],
            dropout=self.config["dropout"],
            hidden_continuous_size=self.config["hidden_continuous_size"],
            output_size=output_size,
            loss=loss_fn,
            learning_rate=self.config["learning_rate"],
            log_interval=-1,  # disable per-step plotting (errorbar bug on
                              # non-monotonic early predictions)
            reduce_on_plateau_patience=4,
        )

        accelerator = "gpu" if device == "cuda" and torch.cuda.is_available() else "cpu"
        # Wall-clock cap comes from the config (experiment.yaml:max_minutes).
        # Default 55 min preserves the laptop-safe fallback when unset.
        cap_minutes = float(max_minutes) if max_minutes else 55.0
        log_dir = Path("results/logs/tft")
        log_dir.mkdir(parents=True, exist_ok=True)
        self.trainer = pl.Trainer(
            max_epochs=max_epochs,
            max_time={"minutes": cap_minutes},
            accelerator=accelerator,
            gradient_clip_val=0.1,
            enable_progress_bar=True,
            enable_checkpointing=True,
            logger=CSVLogger(save_dir=str(log_dir.parent), name=log_dir.name),
            log_every_n_steps=50,
        )
        self.trainer.fit(self.model, train_dataloaders=train_loader,
                         val_dataloaders=val_loader)
        return self

    # ----------------------------------------------------------------- #
    # BaseModel interface
    # ----------------------------------------------------------------- #
    def fit(self, train_loader, val_loader=None, **kwargs):
        """
        Fallback path: reconstruct the DataFrame from a CropYieldDataset.
        Prefer `fit_from_df` for full control.
        """
        ds = getattr(train_loader, "dataset", None)
        if ds is None or not hasattr(ds, "_groups"):
            raise RuntimeError(
                "TFTModel.fit needs a DataLoader wrapping CropYieldDataset, "
                "or call fit_from_df(...) directly."
            )

        import pandas as pd

        def reconstruct(dataset):
            return pd.concat(dataset._groups, ignore_index=True)

        df_tr = reconstruct(ds)
        df_v = reconstruct(val_loader.dataset) if val_loader is not None else None

        # Use the already validated FeatureSpec from the common training path.
        # Prefix-based discovery previously re-enabled radiation accidentally.
        soil_cols = list(ds.spec.static)
        weather_sat_cols = list(ds.spec.past_observed)

        return self.fit_from_df(
            df_train=df_tr, df_val=df_v,
            soil_cols=soil_cols, weather_sat_cols=weather_sat_cols,
            max_epochs=kwargs.get("max_epochs", 50),
            device=kwargs.get("device", "cpu"),
            max_minutes=kwargs.get("max_minutes"),
        )

    def predict(self, loader) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Call fit() / fit_from_df() before predict().")
        out = self.model.predict(loader, mode="prediction", return_y=False)
        return out.cpu().numpy().squeeze()

    def explain(self, loader):
        """Return TFT's native variable-selection + attention scores."""
        if self.model is None:
            raise RuntimeError("Call fit() first.")
        raw_predictions = self.model.predict(loader, mode="raw", return_x=True)
        interpretation = self.model.interpret_output(
            raw_predictions.output, reduction="sum"
        )
        return interpretation

    def save(self, path: str | Path) -> None:
        if self.model is None:
            raise RuntimeError("Nothing to save.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.trainer.save_checkpoint(str(path))

    def load(self, path: str | Path):
        from pytorch_forecasting import TemporalFusionTransformer
        self.model = TemporalFusionTransformer.load_from_checkpoint(str(path))
        return self

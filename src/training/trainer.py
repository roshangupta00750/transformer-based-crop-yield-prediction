"""Thin trainer that delegates to each model's `fit()`.

Keeps the comparison fair by giving every model the same:
  - DataLoader splits
  - wall-clock budget
  - random seed
  - device
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..models import BaseModel
from ..utils import get_logger

log = get_logger("trainer")


@dataclass
class TrainResult:
    model_name: str
    fit_seconds: float
    best_epoch: int | None
    config: dict
    notes: str = ""
    stop_reason: str | None = None          # early_stop | time_cap | max_epochs (models that record it)
    best_val_loss: float | None = None
    history: list = field(default_factory=list)   # per epoch: train_loss, val_loss, seconds


class Trainer:
    def __init__(
        self,
        model: BaseModel,
        device: str = "cpu",
        max_epochs: int = 100,
        early_stopping_patience: int = 15,
        max_minutes: int | None = None,
    ) -> None:
        self.model = model
        self.device = device
        self.max_epochs = max_epochs
        self.patience = early_stopping_patience
        self.max_minutes = max_minutes

    def fit(self, train_loader, val_loader=None) -> TrainResult:
        log.info("Training model=%s on device=%s for ≤%d epochs (cap %s min)",
                 self.model.name, self.device, self.max_epochs, self.max_minutes)
        t0 = time.time()

        self.model.fit(
            train_loader, val_loader,
            max_epochs=self.max_epochs,
            patience=self.patience,
            device=self.device,
            max_minutes=self.max_minutes,
        )

        elapsed = time.time() - t0
        hist = list(getattr(self.model, "history_", []) or [])
        best_epoch = getattr(self.model, "best_epoch_", None)
        vals = [h["val_loss"] for h in hist if h.get("val_loss") is not None]
        stop = getattr(self.model, "stop_reason_", None)
        log.info("Done in %.1fs: %d epochs, best epoch %s, stop: %s", elapsed, len(hist), best_epoch, stop)
        return TrainResult(
            model_name=self.model.name,
            fit_seconds=elapsed,
            best_epoch=best_epoch,
            config=self.model.config,
            stop_reason=stop,
            best_val_loss=min(vals) if vals else None,
            history=hist,
        )

    def save_metadata(self, result: TrainResult, out_dir: str | Path,
                      name: str | None = None) -> Path:
        """Write run metadata. ``name`` should match the checkpoint stem.

        Keyed on model name alone, a ten-task per-crop array had all ten tasks
        overwrite one <model>_meta.json, leaving metadata for whichever task
        finished last and losing the other nine.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{name or result.model_name}_meta.json"
        with open(path, "w") as f:
            json.dump(asdict(result), f, indent=2, default=str)
        return path

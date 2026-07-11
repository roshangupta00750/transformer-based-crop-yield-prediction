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
from dataclasses import asdict, dataclass
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
        log.info("Done in %.1fs", elapsed)
        return TrainResult(
            model_name=self.model.name,
            fit_seconds=elapsed,
            best_epoch=None,
            config=self.model.config,
        )

    def save_metadata(self, result: TrainResult, out_dir: str | Path) -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{result.model_name}_meta.json"
        with open(path, "w") as f:
            json.dump(asdict(result), f, indent=2, default=str)
        return path

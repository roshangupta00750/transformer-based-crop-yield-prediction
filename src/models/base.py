"""Abstract base interface that every benchmark model implements."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseModel(ABC):
    """
    Unified interface so the trainer can swap models with a config switch.

    Subclasses must implement:
      - `fit(train_loader, val_loader, ...)`
      - `predict(loader) -> np.ndarray`

    Subclasses should also expose:
      - `name`: short identifier (matches config key)
      - `save(path)` / `load(path)`
      - `explain(...)` (optional, used by the interpretation runner)
    """

    name: str = "base"

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    @abstractmethod
    def fit(self, train_loader, val_loader=None, **kwargs):
        """Train the model. Return self for chaining."""

    @abstractmethod
    def predict(self, loader):
        """Predict point yield (or median quantile) for each sample."""

    def save(self, path) -> None:
        raise NotImplementedError

    def load(self, path) -> "BaseModel":
        raise NotImplementedError

    def explain(self, loader):
        """Return interpretability artefacts (attention, variable importance).

        Default: not implemented. TFT overrides this with its native
        attention + variable-selection outputs.
        """
        return None

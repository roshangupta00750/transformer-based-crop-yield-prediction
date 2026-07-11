"""Model registry — central place to look models up by name."""
from __future__ import annotations

from .base import BaseModel

MODEL_REGISTRY: dict[str, type[BaseModel]] = {}


def register(cls: type[BaseModel]) -> type[BaseModel]:
    """Decorator: register a model class under its `name` attribute."""
    if not getattr(cls, "name", None):
        raise ValueError(f"{cls.__name__} must define a class-level `name`.")
    MODEL_REGISTRY[cls.name] = cls
    return cls


def build_model(name: str, config: dict) -> BaseModel:
    if name not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model '{name}'. Registered: {sorted(MODEL_REGISTRY)}"
        )
    return MODEL_REGISTRY[name](config)


# Side-effect imports so registration runs at package load
from . import lstm, dlinear, xgboost_model, tft, patchtst, itransformer  # noqa: F401,E402

"""Model implementations under comparison.

All models implement the `BaseModel` interface so the unified trainer
and evaluator can treat them identically.
"""
from .base import BaseModel
from .registry import MODEL_REGISTRY, build_model

__all__ = ["BaseModel", "MODEL_REGISTRY", "build_model"]

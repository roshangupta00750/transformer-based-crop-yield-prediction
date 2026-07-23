"""Project-wide utilities: config loading, seeding, logging."""
from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "configs"


# --------------------------------------------------------------------------- #
# Config loading
# --------------------------------------------------------------------------- #
def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file."""
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with open(path) as f:
        return yaml.safe_load(f)


def load_data_config() -> dict[str, Any]:
    return load_yaml(CONFIG_DIR / "data.yaml")


def load_experiment_config() -> dict[str, Any]:
    return load_yaml(CONFIG_DIR / "experiment.yaml")


def load_model_config(model_name: str) -> dict[str, Any]:
    return load_yaml(CONFIG_DIR / "models" / f"{model_name}.yaml")


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy, and PyTorch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    if _HAS_TORCH:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def pick_device(preference: str = "auto") -> str:
    env_override = os.environ.get("CROP_YIELD_DEVICE")
    if env_override:
        preference = env_override
    if preference == "cpu":
        return "cpu"
    if _HAS_TORCH and torch.cuda.is_available() and preference in ("auto", "cuda"):
        return "cuda"
    if _HAS_TORCH and hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and preference == "auto":
        return "mps"
    return "cpu"


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
def get_logger(name: str = "crop_yield", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                          datefmt="%H:%M:%S")
    )
    logger.addHandler(handler)
    return logger

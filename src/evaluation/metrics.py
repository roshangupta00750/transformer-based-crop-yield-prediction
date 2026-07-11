"""Regression metrics used across all benchmark models."""
from __future__ import annotations

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1.0 - ss_res / max(ss_tot, 1e-12))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> float:
    return float(np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), eps, None))) * 100)


def nse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Nash–Sutcliffe Efficiency — same as R² when reference is the mean."""
    return r2(y_true, y_pred)


def all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": rmse(y_true, y_pred),
        "mae":  mae(y_true, y_pred),
        "r2":   r2(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "nse":  nse(y_true, y_pred),
        "n":    int(len(y_true)),
    }

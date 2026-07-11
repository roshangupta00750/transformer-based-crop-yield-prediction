"""Smoke tests — verify the scaffold imports cleanly with no real data."""
from __future__ import annotations


def test_imports():
    import src                  # noqa: F401
    import src.utils            # noqa: F401
    import src.data             # noqa: F401
    import src.models           # noqa: F401
    import src.training         # noqa: F401
    import src.evaluation       # noqa: F401


def test_model_registry_lists_six():
    from src.models import MODEL_REGISTRY
    expected = {"tft", "patchtst", "itransformer", "lstm", "xgboost", "dlinear"}
    assert expected.issubset(MODEL_REGISTRY), \
        f"Missing: {expected - set(MODEL_REGISTRY)}"


def test_configs_load():
    from src.utils import load_data_config, load_experiment_config, load_model_config
    assert load_data_config()["years"]["start"] == 2003
    assert load_experiment_config()["seed"] == 42
    for name in ["tft", "patchtst", "itransformer", "lstm", "xgboost", "dlinear"]:
        assert load_model_config(name)["name"] == name


def test_metrics():
    import numpy as np
    from src.evaluation import all_metrics
    y = np.array([1.0, 2.0, 3.0, 4.0])
    yhat = np.array([1.1, 2.1, 2.9, 4.2])
    m = all_metrics(y, yhat)
    assert m["rmse"] < 0.3 and m["r2"] > 0.9

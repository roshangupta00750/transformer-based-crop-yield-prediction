"""Aggregate all per-model JSONs and emit one comparison CSV + plot.

Usage:
    python scripts/05_compare_results.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import get_logger, load_experiment_config, PROJECT_ROOT

log = get_logger("compare")


def main() -> None:
    exp_cfg = load_experiment_config()
    metrics_dir = PROJECT_ROOT / exp_cfg["log"]["metrics_dir"]

    rows = []
    for fp in metrics_dir.glob("*_horizons.json"):
        with open(fp) as f:
            data = json.load(f)
        # filename pattern: <model>_<crop>_horizons.json
        parts = fp.stem.split("_")
        model = parts[0]
        crop = "_".join(parts[1:-1])
        for horizon, metrics in data.items():
            rows.append({"model": model, "crop": crop, "horizon": horizon, **metrics})

    if not rows:
        log.warning("No results in %s — run 04_evaluate.py first.", metrics_dir)
        return

    df = pd.DataFrame(rows)
    out_csv = metrics_dir / "comparison.csv"
    df.to_csv(out_csv, index=False)
    log.info("Wrote %s", out_csv)
    print(df.pivot_table(index=["crop", "model"], columns="horizon",
                         values="rmse", aggfunc="mean").round(2))

    # Optional plot
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        for crop, sub in df.groupby("crop"):
            for model, ss in sub.groupby("model"):
                ax.plot(ss["horizon"], ss["rmse"], marker="o", label=f"{model}—{crop}")
        ax.set_xlabel("Horizon (days pre-harvest)")
        ax.set_ylabel("RMSE (dt/ha)")
        ax.legend(fontsize=8)
        ax.set_title("In-season RMSE by horizon and model")
        fig_path = PROJECT_ROOT / exp_cfg["log"]["figures_dir"] / "horizon_rmse.png"
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout(); fig.savefig(fig_path, dpi=120)
        log.info("Wrote %s", fig_path)
    except ImportError:
        log.info("matplotlib not installed; skipping plot.")


if __name__ == "__main__":
    main()

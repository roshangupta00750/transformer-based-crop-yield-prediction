"""Inputs for the v5 national gate (paper/preregistration_2026-09-30_v5.md §2.1).

1. National seasonal indicators per year from dataset_v4's daily weather
   -> data/processed/static/national_indicators_v4.parquet
2. Gate decision per crop for every rolling fold (training years <= Y-2,
   Y = 2016..2025) -> data/processed/static/gate_decisions_v5.csv
   Uses training years only; printed for the record before any evaluation.

Usage: python scripts/49_national_gate_inputs.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.models.national_gate import gate_decisions, national_indicators  # noqa: E402

DATA = ROOT / "data/processed/dataset_v4.parquet"
OUT_IND = ROOT / "data/processed/static/national_indicators_v4.parquet"
OUT_GATE = ROOT / "data/processed/static/gate_decisions_v5.csv"


def load_yields() -> pd.DataFrame:
    y = pd.read_parquet(DATA, columns=["landkreis", "crop", "year", "yield_dt_per_ha"])
    return (y.dropna().drop_duplicates(["landkreis", "crop", "year"])
             .rename(columns={"yield_dt_per_ha": "yield"}))


def main() -> None:
    cols = ["landkreis", "date", "precipitation_height", "cwb", "temperature_air_max_2m", "swroot7"]
    daily = pd.read_parquet(DATA, columns=cols)
    ind = national_indicators(daily)
    del daily
    OUT_IND.parent.mkdir(parents=True, exist_ok=True)
    ind.to_parquet(OUT_IND)
    print(ind.round(1).to_string())

    y = load_yields()
    rows = [gate_decisions(y, ind, train_end=Y - 2).assign(test_year=Y, train_end=Y - 2)
            for Y in range(2016, 2026)]
    G = pd.concat(rows, ignore_index=True)
    G.to_csv(OUT_GATE, index=False)
    print("\nLOYO R² on training years (gate on if < 0.20):")
    print(G.pivot_table(index="crop", columns="test_year", values="loyo_r2").round(2).to_string())
    print("\ngate on (folds of 10):", G.groupby("crop").gate.sum().astype(int).to_dict())


if __name__ == "__main__":
    main()

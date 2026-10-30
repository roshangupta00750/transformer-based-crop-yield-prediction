"""Merge the Earth Engine flower_<Y>.csv exports into data/processed/static/ndyi_crop.parquet
(preregistration v8 §2.1). Input: data/raw/prereg_v1/flower_*.csv (scripts/26 download)."""
from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
files = sorted(glob.glob(str(ROOT / "data/raw/prereg_v1/flower_*.csv")))
d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True).dropna(subset=["ndyi_crop"])
d["date"] = pd.to_datetime(d["date"])
d = d[(d["ndyi_crop"] > -1) & (d["ndyi_crop"] < 1)].drop_duplicates(["landkreis", "date"])
out = ROOT / "data/processed/static/ndyi_crop.parquet"
d[["landkreis", "date", "ndyi_crop"]].sort_values(["landkreis", "date"]).to_parquet(out, index=False)
print(f"{len(files)} files, {len(d):,} district-days, {d.landkreis.nunique()} districts, "
      f"{d.date.dt.year.min()}-{d.date.dt.year.max()} -> {out.relative_to(ROOT)}")
print(d.groupby(d.date.dt.year).ndyi_crop.agg(["count", "mean"]).round(3).to_string())

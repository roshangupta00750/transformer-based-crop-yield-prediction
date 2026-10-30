"""Merge the crop-specific MODIS exports (scripts/32 --sources vics,laics,cropshare) into
data/processed/static/cs_sat.parquet and cs_share.csv (preregistration v13 §2).

Long table: landkreis, date (end of the composite period: VI +16 d, LAI/fPAR +8 d), mask,
ndvi, evi, lai, fpar. The modelling frame joins it per crop (src/data/dataset.add_crop_specific_sat).
"""
import glob
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data/raw/prereg_v1"
MASKS = ["wheat", "barley", "rye", "oats", "maize", "triticale", "potato", "sugarbeet", "rapeseed"]


def long(pattern, bands, shift):
    d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(RAW / pattern)))], ignore_index=True)
    d["date"] = pd.to_datetime(d["date"]) + pd.Timedelta(days=shift)
    parts = []
    for m in MASKS:
        cols = {f"{b}_{m}": b for b in bands}
        parts.append(d[["landkreis", "date"] + list(cols)].rename(columns=cols).assign(mask=m))
    return pd.concat(parts, ignore_index=True)


vi = long("vics_*.csv", ["ndvi", "evi"], 16)
la = long("laics_*.csv", ["lai", "fpar"], 8)
out = pd.concat([vi, la], ignore_index=True)
out = out.groupby(["landkreis", "mask", "date"], as_index=False).first()
p = ROOT / "data/processed/static/cs_sat.parquet"
out.to_parquet(p, index=False)
share = pd.read_csv(RAW / "cropshare.csv")
share.to_csv(ROOT / "data/processed/static/cs_share.csv", index=False)
print(f"{p.name}: {len(out):,} rows, {out.landkreis.nunique()} districts, {out.date.min().date()}..{out.date.max().date()}")
print(out.groupby("mask")[["ndvi", "lai"]].agg(["count", "mean"]).round(3).to_string())

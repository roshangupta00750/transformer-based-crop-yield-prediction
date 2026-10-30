"""District x crop x harvest-year dates of observed crop development (DWD).

Implements paper/preregistration_2026-09-28_v2.md §2.2:
    networks   full = annual + immediate reporters; rt = immediate reporters only
    slots      sow, emerge, stem, head, ripe (phase ids per crop below); harvest unused
    quality    DWD quality byte 5/7/8 (doubtful / invalid date / wrong) dropped
    district   median of station dates inside the NUTS-3 polygon; else median of
               the 5 nearest stations within 75 km of the centroid; else missing
    harvest yr winter crops: dates in Aug-Dec belong to the next harvest year
               (v2 rule, all slots). --with-harvest (v3) fixes it: only sow and
               emerge move to the next year; stem..harvest stay in their calendar
               year (an August harvest or ripeness belongs to that year's crop)
    proxies    triticale = mean of winter wheat and rye district dates per slot;
               summer barley = oats in years without summer-barley observations

Output: data/processed/static/phenology_obs.parquet
        columns landkreis, crop, year (harvest), network, slot, date
        --with-harvest (preregistration v3 §2.1): adds slot "harvest" (phase 24;
        silage maize 24 or 39) and writes phenology_obs_v3.parquet instead.
Usage:  python scripts/43_phenology_dates.py [--with-harvest]
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PH = ROOT / "data/raw/phenology_dwd"
OUT = ROOT / "data/processed/static/phenology_obs.parquet"
YEARS = (2000, 2026)
BAD_QB = {5, 7, 8}
SLOTS = ["sow", "emerge", "stem", "head", "ripe"]
HARVEST = {"silage_maize": [24, 39]}   # every other crop: [24]
# crop -> (DWD file name, winter?, {slot: phase_id})
CROPS = {
    "winter_wheat":    ("Winterweizen", True,  dict(sow=10, emerge=12, stem=15, head=18, ripe=21)),
    "winter_barley":   ("Wintergerste", True,  dict(sow=10, emerge=12, stem=15, head=18, ripe=21)),
    "rye":             ("Winterroggen", True,  dict(sow=10, emerge=12, stem=15, head=18, ripe=21)),
    "winter_rapeseed": ("Winterraps",   True,  dict(sow=10, emerge=12, stem=17, head=5, ripe=22)),
    "summer_barley":   ("Sommergerste", False, dict(sow=10, emerge=12, stem=15, head=18, ripe=21)),
    "oats":            ("Hafer",        False, dict(sow=10, emerge=12, stem=15, head=66, ripe=21)),
    "silage_maize":    ("Mais",         False, dict(sow=10, emerge=12, stem=67, head=65, ripe=19)),
    "potatoes":        ("Kartoffel",    False, dict(sow=10, emerge=12, stem=13, head=5)),
    "sugar_beet":      ("Zucker-Ruebe", False, dict(sow=10, emerge=12, stem=13)),
}


def read_obs(pattern: str) -> pd.DataFrame:
    parts = []
    for f in sorted(glob.glob(str(PH / pattern))):
        d = pd.read_csv(f, sep=";", skipinitialspace=True, dtype=str, encoding="latin1")
        d.columns = [c.strip() for c in d.columns]
        d = d[d["Stations_id"].str.strip().str.isdigit()]
        parts.append(d)
    if not parts:
        return pd.DataFrame(columns=["station", "phase", "date"])
    d = pd.concat(parts, ignore_index=True)
    d = d[~d["Eintrittsdatum_QB"].str.strip().astype(int).isin(BAD_QB)]
    out = pd.DataFrame({"station": d["Stations_id"].str.strip().astype(int),
                        "phase": d["Phase_id"].str.strip().astype(int),
                        "date": pd.to_datetime(d["Eintrittsdatum"].str.strip(), format="%Y%m%d", errors="coerce")})
    return out.dropna().drop_duplicates()


def stations() -> gpd.GeoDataFrame:
    parts = []
    for f in ("Jahresmelder", "Sofortmelder"):
        s = pd.read_csv(PH / f"PH_Beschreibung_Phaenologie_Stationen_{f}.txt", sep=";",
                        encoding="latin1", dtype=str)
        s.columns = [c.strip() for c in s.columns]
        s = s[s["Stations_id"].str.strip().str.isdigit()]
        parts.append(pd.DataFrame({"station": s["Stations_id"].str.strip().astype(int),
                                   "lat": s["geograph.Breite"].str.strip().astype(float),
                                   "lon": s["geograph.Laenge"].str.strip().astype(float)}))
    s = pd.concat(parts).drop_duplicates("station")
    return gpd.GeoDataFrame(s, geometry=gpd.points_from_xy(s.lon, s.lat), crs=4326)


def district_dates(obs: pd.DataFrame, st: gpd.GeoDataFrame, nuts: gpd.GeoDataFrame) -> pd.DataFrame:
    """Median date per (district, year, slot); inside first, else 5 nearest <= 75 km."""
    inside = gpd.sjoin(st, nuts[["landkreis", "geometry"]], predicate="within")[["station", "landkreis"]]
    o = obs.merge(inside, on="station", how="left")
    have = o.dropna(subset=["landkreis"]).groupby(["landkreis", "year", "slot"])["date"].median()
    # nearest-station fallback, metric distances in EPSG:3035
    st_m = st.to_crs(3035).set_index("station")
    cen = nuts.to_crs(3035).set_index("landkreis").geometry.centroid
    rows = []
    for (year, slot), g in obs.groupby(["year", "slot"]):
        pts = st_m.loc[st_m.index.intersection(g["station"].unique())]
        if pts.empty:
            continue
        date_by_st = g.groupby("station")["date"].median()
        for lk, c in cen.items():
            if (lk, year, slot) in have.index:
                continue
            d = pts.geometry.distance(c)
            near = d[d <= 75_000].nsmallest(5)
            if len(near):
                rows.append((lk, year, slot, date_by_st.loc[near.index].median()))
    fb = pd.Series({(a, b, c): d for a, b, c, d in rows}, dtype="datetime64[ns]")
    fb.index.names = ["landkreis", "year", "slot"]
    return pd.concat([have, fb]).rename("date").reset_index()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--with-harvest", action="store_true")
    args = ap.parse_args()
    out_path = OUT.with_name("phenology_obs_v3.parquet") if args.with_harvest else OUT
    nuts = gpd.read_file(ROOT / "data/raw/geo/NUTS_RG_01M_2021_4326_LEVL_3.shp")
    nuts = nuts[nuts["CNTR_CODE"] == "DE"].rename(columns={"NUTS_ID": "landkreis"})
    st = stations()
    out = []
    for crop, (name, winter, slots) in CROPS.items():
        for net, files in (("full", ["Jahresmelder", "Sofortmelder"]), ("rt", ["Sofortmelder"])):
            obs = pd.concat([o for f in files
                             if not (o := read_obs(f"PH_{f}_Landwirtschaft_Kulturpflanze_{name}_*.txt")).empty]
                            or [pd.DataFrame(columns=["station", "phase", "date"])])
            if obs.empty:
                print(f"{crop:16s} {net:4s} no observations (left missing)")
                continue
            inv = {v: k for k, v in slots.items()}
            if args.with_harvest:
                inv.update({p: "harvest" for p in HARVEST.get(crop, [24])})
            obs = obs[obs["phase"].isin(inv)].assign(slot=lambda d: d["phase"].map(inv))
            autumn = (obs["date"].dt.month >= 8) & winter
            if args.with_harvest:
                autumn &= obs["slot"].isin(["sow", "emerge"])
            obs["year"] = obs["date"].dt.year + autumn.astype(int)
            obs = obs[obs["year"].between(*YEARS)]
            if obs.empty:
                continue
            dd = district_dates(obs, st, nuts)
            out.append(dd.assign(crop=crop, network=net))
            print(f"{crop:16s} {net:4s} stations {obs.station.nunique():5d}  district-years "
                  f"{dd.groupby(['landkreis', 'year']).ngroups:6d}", flush=True)
    D = pd.concat(out, ignore_index=True)

    # proxies (fixed in the plan)
    key = ["landkreis", "year", "network", "slot"]
    ww = D[D.crop == "winter_wheat"].set_index(key)["date"]
    ry = D[D.crop == "rye"].set_index(key)["date"]
    both = pd.concat([ww.rename("a"), ry.rename("b")], axis=1).dropna()
    tri = (both["a"] + (both["b"] - both["a"]) / 2).dt.normalize().rename("date").reset_index()
    D = pd.concat([D, tri.assign(crop="triticale")], ignore_index=True)
    sb_years = D[D.crop == "summer_barley"].groupby("network")["year"].min().to_dict()
    oats = D[D.crop == "oats"]
    fill = oats[oats.apply(lambda r: r["year"] < sb_years.get(r["network"], 9999), axis=1)]
    D = pd.concat([D, fill.assign(crop="summer_barley")], ignore_index=True)

    D = D[["landkreis", "crop", "year", "network", "slot", "date"]].sort_values(
        ["crop", "network", "landkreis", "year", "slot"]).reset_index(drop=True)
    assert not D.duplicated(["landkreis", "crop", "year", "network", "slot"]).any()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    D.to_parquet(out_path, index=False)
    cov = D.groupby(["crop", "network"]).apply(lambda g: g.groupby(["landkreis", "year"]).ngroups)
    print(f"\nwrote {out_path.relative_to(ROOT)}: {len(D):,} district-slot dates\n{cov.unstack().to_string()}")


if __name__ == "__main__":
    main()

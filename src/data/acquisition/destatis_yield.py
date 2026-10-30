"""Crop yield labels at Landkreis (NUTS-3) level via Regionaldatenbank Deutschland.

Pulls annual crop yield (Ertrag, dt/ha) per Kreis from the Regionaldatenbank
(`regionalstatistik.de`), which is the federation-of-state-offices database.
The federal Destatis Genesis (`www-genesis.destatis.de`) only exposes country-
and state-level aggregates for these tables.

Source table: 41241-01-03-4 — "Erträge ausgewählter landwirtschaftlicher
Feldfrüchte - Jahressumme - regionale Tiefe: Kreise".

Credentials: requires a (free) Regionaldatenbank account, separate from federal
Destatis. Set `REGIO_USER` and `REGIO_PASS` env vars (`DESTATIS_USER`/`DESTATIS_PASS`
are accepted as fallback for backward compatibility).
"""
from __future__ import annotations

import io
import os
import re
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests

from ...utils import get_logger

log = get_logger("destatis")

REGIO_URL = "https://www.regionalstatistik.de/genesisws/rest/2020"
TABLE_ID = "41241-01-03-4"


def download_via_api(
    out_dir: str | Path,
    table_id: str = TABLE_ID,
    username: str | None = None,
    password: str | None = None,
    startyear: int = 2003,
    endyear: int = 2023,
) -> Path:
    """Download a Regionaldatenbank table via REST API.

    The API requires POST with credentials in HTTP headers and returns a ZIP
    containing one flat CSV. Without an explicit year range the API returns
    only the most recent year, so pass ``startyear``/``endyear``.
    """
    username = username or os.environ.get("REGIO_USER") or os.environ.get("DESTATIS_USER")
    password = password or os.environ.get("REGIO_PASS") or os.environ.get("DESTATIS_PASS")
    if not (username and password):
        raise RuntimeError(
            "Set REGIO_USER + REGIO_PASS env vars (register at regionalstatistik.de)."
        )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"destatis_{table_id}.csv"

    log.info("Fetching Regionaldatenbank table %s (%d-%d)", table_id, startyear, endyear)
    auth = {"username": username, "password": password}
    payload = {
        "name": table_id, "area": "all", "format": "ffcsv", "language": "en",
        "startyear": str(startyear), "endyear": str(endyear),
    }

    # Try synchronous first; fall back to async job-poll when table is too big.
    r = requests.post(f"{REGIO_URL}/data/tablefile",
                      headers=auth, data=payload, timeout=180)
    r.raise_for_status()

    if r.headers.get("content-type", "").startswith("application/json"):
        msg = r.json()
        if msg.get("Status", {}).get("Code") == 98:
            # "too big for dialogue-processing" → submit as background job.
            content = _fetch_via_job(table_id, auth, payload)
        else:
            raise RuntimeError(f"Regionaldatenbank API error: {r.text}")
    else:
        content = r.content

    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
        if not csv_names:
            raise RuntimeError(f"No CSV in ZIP. Members: {zf.namelist()}")
        with zf.open(csv_names[0]) as src:
            out_file.write_bytes(src.read())
    log.info("Wrote %s", out_file)
    return out_file


def _fetch_via_job(table_id: str, auth: dict, payload: dict,
                   poll_every: int = 5, timeout: int = 600) -> bytes:
    """Submit table request as a background job, poll until ready, fetch result."""
    log.info("Table too large for sync request — submitting background job.")

    job_payload = {**payload, "job": "true"}
    r = requests.post(f"{REGIO_URL}/data/tablefile",
                      headers=auth, data=job_payload, timeout=60)
    r.raise_for_status()
    submit = r.json()
    status = submit.get("Status", {})
    if status.get("Code") not in (0, 99):
        raise RuntimeError(f"Job submission failed: {submit}")

    # Job ID is the table name suffixed with the job's internal code; the API
    # returns it in the Status content.
    content_text = status.get("Content", "")
    m = re.search(r"\b([A-Za-z0-9_-]+_\d+)\b", content_text)
    job_id = m.group(1) if m else f"{table_id}_0"
    log.info("Job submitted: %s — polling every %ds", job_id, poll_every)

    deadline = time.time() + timeout
    while time.time() < deadline:
        time.sleep(poll_every)
        # Look up the job; status text tells us if it's still running.
        list_r = requests.post(f"{REGIO_URL}/catalogue/jobs",
                               headers=auth,
                               data={"selection": job_id, "language": "en"},
                               timeout=30)
        list_r.raise_for_status()
        jobs = (list_r.json().get("List") or [])
        if jobs:
            state = jobs[0].get("State", "")
            log.info("Job state: %s", state)
            if state.lower().startswith(("fertig", "finished", "complete")):
                break

    # Fetch the prepared result.
    result_r = requests.post(f"{REGIO_URL}/data/resultfile",
                             headers=auth,
                             data={"name": job_id, "area": "all",
                                   "format": "ffcsv", "language": "en",
                                   "compress": "false"},
                             timeout=180)
    result_r.raise_for_status()
    if result_r.headers.get("content-type", "").startswith("application/json"):
        raise RuntimeError(f"Result fetch failed: {result_r.text}")
    return result_r.content


# German crop name → snake_case used in configs / training
CROP_MAP = {
    "Winterweizen": "winter_wheat",
    "Winterraps": "winter_rapeseed",
    "Wintergerste": "winter_barley",
    "Sommergerste": "summer_barley",
    "Roggen und Wintermenggetreide": "rye",
    "Hafer": "oats",
    "Triticale": "triticale",
    "Silomais": "silage_maize",
    "Kartoffeln": "potatoes",
    "Zuckerrüben": "sugar_beet",
}


def _normalize_name(s: str) -> str:
    """Strip Kreis-type suffixes and lowercase for fuzzy matching."""
    s = s.lower()
    s = re.sub(r",?\s*(kreisfreie\s+stadt|landeshauptstadt|landkreis|kreis|stadt)\b", "", s)
    s = re.sub(r"[^\w]+", " ", s, flags=re.UNICODE).strip()
    return s


def _build_ags_to_nuts3(geo_dir: str | Path) -> dict[str, str]:
    """Build an AGS (5-digit) → NUTS-3 mapping by matching Kreis names."""
    import geopandas as gpd
    shp = gpd.read_file(Path(geo_dir) / "NUTS_RG_01M_2021_4326_LEVL_3.shp")
    de = shp[shp["CNTR_CODE"] == "DE"][["NUTS_ID", "NUTS_NAME"]]
    return {_normalize_name(n): nid for nid, n in zip(de["NUTS_ID"], de["NUTS_NAME"])}


# --------------------------------------------------------------------------- #
# AGS -> NUTS-3 by code (preregistration v10 §2.1). The name-only mapping above
# collides city/Landkreis pairs (e.g. "Hof, Kreisfreie Stadt" and "Hof,
# Landkreis" both -> "hof") and drops renamed districts; kept only for old builds.
# --------------------------------------------------------------------------- #
_STATE = {"01": "DEF", "02": "DE6", "03": "DE9", "04": "DE5", "05": "DEA", "06": "DE7", "07": "DEB",
          "08": "DE1", "09": "DE2", "10": "DEC", "11": "DE3", "12": "DE4", "13": "DE8", "14": "DED",
          "15": "DEE", "16": "DEG"}
# official successors of district reforms (old AGS -> 2021 NUTS-3)
_OLD_TO_NUTS = {
    "03152": "DE91C", "03156": "DE91C", "05313": "DEA2D", "05354": "DEA2D", "10041": "DEC01",
    "13001": "DE80N", "13002": "DE80J", "13005": "DE80L", "13006": "DE80M", "13051": "DE80K",
    "13052": "DE80J", "13053": "DE80K", "13054": "DE80O", "13055": "DE80J", "13056": "DE80J",
    "13057": "DE80L", "13058": "DE80M", "13059": "DE80N", "13060": "DE80O", "13061": "DE80L", "13062": "DE80N",
    "14161": "DED41", "14166": "DED44", "14167": "DED45", "14171": "DED42", "14173": "DED45",
    "14177": "DED43", "14178": "DED44", "14181": "DED42", "14182": "DED43", "14188": "DED42",
    "14191": "DED42", "14193": "DED45", "14262": "DED21", "14263": "DED2D", "14264": "DED2C",
    "14272": "DED2C", "14280": "DED2E", "14284": "DED2D", "14285": "DED2E", "14286": "DED2D",
    "14287": "DED2F", "14290": "DED2F", "14292": "DED2C", "14365": "DED51", "14374": "DED53",
    "14375": "DED43", "14379": "DED52", "14383": "DED52", "14389": "DED53",
    "15101": "DEE01", "15151": "DEE05", "15153": "DEE0C", "15154": "DEE05", "15159": "DEE05",
    "15171": "DEE0E", "15202": "DEE02", "15256": "DEE08", "15260": "DEE0A", "15261": "DEE0B",
    "15265": "DEE0B", "15266": "DEE0A", "15268": "DEE08", "15303": "DEE03", "15352": "DEE0C",
    "15355": "DEE07", "15357": "DEE09", "15358": "DEE06", "15362": "DEE07", "15363": "DEE0D",
    "15364": "DEE09", "15367": "DEE0C", "15369": "DEE09", "15370": "DEE04"}
# current codes whose names do not match GISCO's spelling
_CUR_TO_NUTS = {"13003": "DE803", "13004": "DE804", "11000": "DE300", "02000": "DE600",
                "16056": "DEG0N", "03241": "DE929", "05334": "DEA2D", "13072": "DE80K",
                "14729": "DED52", "14713": "DED51", "14511": "DED41", "14612": "DED21"}


def _base_name(s: str) -> str:
    s = s.lower()
    for pat in (r"kreisfreie stadt", r"stadtkreis", r"krsfr\. stadt", r"krfr\. stadt", r"landeshauptstadt",
                r"hansestadt", r"landkreis", r"\bkreis\b", r", stadt\b", r"\(de\)", r"regionalverband",
                r"\bstadt\b"):
        s = re.sub(pat, " ", s)
    return re.sub(r"[^\w]+", "", s)


def _raw_kind(code: str, label: str) -> str:
    l = label.lower()
    if re.search(r"kreisfreie stadt|stadtkreis|krsfr|krfr|, stadt\b|landeshauptstadt", l) and "region hannover" not in l:
        return "city"
    if code.startswith("09") and code[3] == "6":        # Bavarian kreisfreie Städte
        return "city"
    if code in ("02000", "11000", "04011", "04012"):
        return "city"
    return "lk"


def build_ags_to_nuts3_by_code(codes_labels, geo_dir: str | Path) -> pd.DataFrame:
    """AGS -> NUTS-3 (2021) by state + city/Landkreis type + name, plus reform successors.

    Returns columns ags, label, nuts, kind ('current' | 'old'). Every current
    NUTS-3 code is the target of at most one current AGS (asserted).
    """
    import difflib
    import geopandas as gpd
    shp = gpd.read_file(Path(geo_dir) / "NUTS_RG_01M_2021_4326_LEVL_3.shp")
    de = shp[shp["CNTR_CODE"] == "DE"][["NUTS_ID", "NUTS_NAME"]].copy()
    de["st"] = de["NUTS_ID"].str[:3]
    de["b"] = de["NUTS_NAME"].map(_base_name)
    l = de["NUTS_NAME"].str.lower()
    de["t"] = l.map(lambda x: "city" if ("kreisfreie stadt" in x or "stadtkreis" in x)
                    else ("lk" if "landkreis" in x else "any"))
    rows = []
    for code, label in codes_labels:
        label = str(label)
        if code in _CUR_TO_NUTS:
            rows.append((code, label, _CUR_TO_NUTS[code], "current")); continue
        if code in _OLD_TO_NUTS:
            rows.append((code, label, _OLD_TO_NUTS[code], "old")); continue
        cand = de[(de["st"] == _STATE[code[:2]]) & de["t"].isin([_raw_kind(code, label), "any"])]
        b = _base_name(label)
        hit = cand[cand["b"] == b]
        if len(hit) != 1:
            hit = cand[cand["b"].isin(difflib.get_close_matches(b, cand["b"].tolist(), n=1, cutoff=0.75))]
        rows.append((code, label, hit["NUTS_ID"].iloc[0] if len(hit) == 1 else None, "current"))
    mp = pd.DataFrame(rows, columns=["ags", "label", "nuts", "kind"])
    cur = mp[(mp["kind"] == "current") & mp["nuts"].notna()]
    dup = cur[cur["nuts"].duplicated(keep=False)]
    assert dup.empty, f"two current AGS map to one NUTS-3:\n{dup}"
    return mp


def clean_by_code(raw_paths, out_path: str | Path, geo_dir: str | Path,
                  mapping_out: str | Path | None = None) -> Path:
    """(landkreis, crop, year, yield) from one or more Regionaldatenbank CSVs, matched by AGS.

    A current-code value always wins. A year without one is filled only from a
    single old predecessor that maps to that NUTS-3 alone (preregistration v10 §2.1).
    """
    raw = pd.concat([pd.read_csv(p, sep=";", encoding="utf-8-sig", dtype=str) for p in raw_paths],
                    ignore_index=True)
    raw = raw.rename(columns={"time": "year", "1_variable_attribute_code": "ags",
                              "1_variable_attribute_label": "kreis_name",
                              "2_variable_attribute_label": "crop_de", "value": "yield_dt_per_ha"})
    raw = raw[raw["ags"].str.fullmatch(r"\d{5}", na=False)]
    codes = raw[["ags", "kreis_name"]].drop_duplicates("ags").values
    mp = build_ags_to_nuts3_by_code(codes, geo_dir)
    if mapping_out:
        mp.to_csv(mapping_out, index=False)
    df = raw.merge(mp[["ags", "nuts", "kind"]], on="ags", how="left").rename(columns={"nuts": "landkreis"})
    df["crop"] = df["crop_de"].map(CROP_MAP)
    df["yield_dt_per_ha"] = pd.to_numeric(df["yield_dt_per_ha"], errors="coerce")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["landkreis", "crop", "yield_dt_per_ha", "year"])
    df["year"] = df["year"].astype(int)
    df = df.drop_duplicates(["ags", "crop", "year"], keep="last")
    cur = df[df["kind"] == "current"]
    assert not cur.duplicated(["landkreis", "crop", "year"]).any(), "duplicate current values"
    old = df[df["kind"] == "old"]
    n_pred = old.groupby(["landkreis", "crop", "year"])["ags"].transform("nunique")
    single = old[n_pred == 1]
    have = pd.MultiIndex.from_frame(cur[["landkreis", "crop", "year"]])
    fill = single[~pd.MultiIndex.from_frame(single[["landkreis", "crop", "year"]]).isin(have)]
    out = pd.concat([cur.assign(source="current"), fill.assign(source="predecessor")], ignore_index=True)
    out = out[["landkreis", "crop", "year", "yield_dt_per_ha", "source"]].sort_values(
        ["landkreis", "crop", "year"]).reset_index(drop=True)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    log.info("Yields by AGS -> %s: %d rows, %d districts (%d predecessor fills)",
             out_path, len(out), out["landkreis"].nunique(), int((out["source"] == "predecessor").sum()))
    return Path(out_path)


def clean(
    raw_path: str | Path,
    out_path: str | Path,
    geo_dir: str | Path | None = None,
) -> Path:
    """Parse the Regionaldatenbank flat CSV into (landkreis, crop, year, yield).

    The flat CSV format (ffcsv) has columns:
      time, 1_variable_code (area type), 1_variable_attribute_code (AGS),
      1_variable_attribute_label (Kreis name), 2_variable_attribute_label
      (crop name), value (dt/ha).

    We filter to 5-digit AGS codes (Kreis rows only), then resolve AGS →
    NUTS-3 by matching normalized Kreis names against the GISCO shapefile.
    """
    raw_path, out_path = Path(raw_path), Path(out_path)
    if geo_dir is None:
        geo_dir = raw_path.parent.parent / "geo"

    df = pd.read_csv(raw_path, sep=";", encoding="utf-8-sig", low_memory=False)

    df = df.rename(columns={
        "time": "year",
        "1_variable_attribute_code": "ags",
        "1_variable_attribute_label": "kreis_name",
        "2_variable_attribute_label": "crop_de",
        "value": "yield_dt_per_ha",
    })

    # Keep only Kreis-level rows (5-digit AGS).
    df = df[df["ags"].astype(str).str.fullmatch(r"\d{5}")]

    # Resolve AGS → NUTS-3 by matching Kreis names against the GISCO shapefile.
    nuts_by_name = _build_ags_to_nuts3(geo_dir)
    df["landkreis"] = df["kreis_name"].map(lambda s: nuts_by_name.get(_normalize_name(str(s))))
    unmatched = df.loc[df["landkreis"].isna(), "kreis_name"].unique()
    if len(unmatched):
        log.warning("Could not map %d Kreis names to NUTS-3: %s",
                    len(unmatched), list(unmatched)[:5])

    df["crop"] = df["crop_de"].map(CROP_MAP)
    df["yield_dt_per_ha"] = pd.to_numeric(df["yield_dt_per_ha"], errors="coerce")
    df = df.dropna(subset=["landkreis", "crop", "yield_dt_per_ha"])
    df["year"] = df["year"].astype(int)
    df = df[["landkreis", "crop", "year", "yield_dt_per_ha"]]
    # Destatis reports several rows per (landkreis, crop, year) when the same
    # crop has multiple "intended use" categories (e.g. main vs. fodder).
    # Average them so we end up with one yield value per sample.
    df = (df.groupby(["landkreis", "crop", "year"], as_index=False)
            ["yield_dt_per_ha"].mean())

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    log.info("Cleaned yield → %s (%d rows, %d Kreise, %d crops)",
             out_path, len(df), df["landkreis"].nunique(), df["crop"].nunique())
    return out_path

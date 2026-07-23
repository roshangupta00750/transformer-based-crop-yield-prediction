"""SoilGrids 2.0 — static soil properties per Landkreis.

Pulls aggregated values for sand %, clay %, soil organic carbon, pH, and
bulk density (0-30 cm) at the Landkreis centroid from the ISRIC SoilGrids
REST API.

ISRIC heavily rate-limits this endpoint (HTTP 429 after ~5 fast requests),
so we use a session-pooled client, throttle to ~1 req/sec, and
exponential-back-off any 429/5xx response. Partial progress is persisted to
the output file after every Landkreis so a crash mid-run is recoverable.

Reference: https://rest.isric.org
"""
from __future__ import annotations

import random
import time
from pathlib import Path

import pandas as pd
import requests

from ...utils import get_logger

log = get_logger("soilgrids")

API = "https://rest.isric.org/soilgrids/v2.0/properties/query"
PROPS = ("sand", "clay", "soc", "phh2o", "bdod")
# The 2.0 API no longer exposes a "0-30cm" aggregate — it returns native
# 0-5/5-15/15-30/30-60/100/200 slices. We pull the three topsoil slices and
# depth-weighted average them client-side to approximate root-zone properties.
TOPSOIL_LABELS = ("0-5cm", "5-15cm", "15-30cm")
TOPSOIL_WEIGHTS = (5.0, 10.0, 15.0)

# Rate-limit knobs. ISRIC's docs say <=5 req/min for unauthenticated use; we
# keep ourselves well under that, with jitter so concurrent users don't sync up.
BASE_DELAY = 1.5   # seconds between successful requests
MAX_RETRIES = 6
BACKOFF_BASE = 5.0


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": "crop-yield-benchmark/0.1 (research)",
                      "Accept": "application/json"})
    return s


def fetch_point(session: requests.Session, lon: float, lat: float) -> dict:
    """Query SoilGrids for a single point, with retry/backoff on 429/5xx."""
    params = [("lon", lon), ("lat", lat), ("value", "mean")]
    params += [("property", p) for p in PROPS]
    params += [("depth", d) for d in TOPSOIL_LABELS]

    for attempt in range(MAX_RETRIES):
        try:
            r = session.get(API, params=params, timeout=30)
        except requests.exceptions.RequestException as e:
            wait = BACKOFF_BASE * (2 ** attempt) + random.uniform(0, 1)
            log.warning("Network error (%s); backing off %.1fs", e, wait)
            time.sleep(wait)
            continue

        if r.status_code == 200:
            return r.json()
        if r.status_code in (429, 502, 503, 504):
            wait = BACKOFF_BASE * (2 ** attempt) + random.uniform(0, 1)
            log.warning("HTTP %d, backing off %.1fs (attempt %d/%d)",
                        r.status_code, wait, attempt + 1, MAX_RETRIES)
            time.sleep(wait)
            continue
        r.raise_for_status()

    raise RuntimeError(f"SoilGrids exhausted retries for ({lon}, {lat})")


def _parse_response(data: dict) -> dict[str, float | None]:
    """Pull mean values for each requested property, depth-averaged to 0-30 cm.

    Each property's `depths` list contains the requested slices in unspecified
    order. We weight 0-5/5-15/15-30 cm by their thicknesses (5/10/15 cm) so
    the topsoil average is biased toward the deeper, more agronomically
    relevant horizons.
    """
    out: dict[str, float | None] = {}
    for layer in data.get("properties", {}).get("layers", []):
        name = layer["name"]
        slices = {d.get("label"): d.get("values", {}).get("mean")
                  for d in layer.get("depths", [])}
        nums, weights = [], []
        for label, w in zip(TOPSOIL_LABELS, TOPSOIL_WEIGHTS):
            v = slices.get(label)
            if v is not None:
                nums.append(v * w)
                weights.append(w)
        out[name] = sum(nums) / sum(weights) if weights else None
    return out


def download(
    out_dir: str | Path,
    landkreis_centroids: pd.DataFrame,
) -> Path:
    """Pull soil properties for each Landkreis centroid.

    Resumes from any partial run by reading the existing parquet and skipping
    districts that already have non-null values.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "soilgrids.parquet"

    # Resume: pick up any districts already fetched.
    existing: dict[str, dict] = {}
    if out_file.exists():
        try:
            prev = pd.read_parquet(out_file)
            for _, r in prev.iterrows():
                if all(pd.notna(r.get(p)) for p in PROPS):
                    existing[r["landkreis"]] = r.to_dict()
            log.info("Resuming: %d Landkreise already have data.", len(existing))
        except Exception:
            pass

    rows: list[dict] = list(existing.values())
    todo = landkreis_centroids[~landkreis_centroids["landkreis"].isin(existing)]
    log.info("Downloading SoilGrids for %d remaining Landkreise (throttle %.1fs)",
             len(todo), BASE_DELAY)

    sess = _session()
    for i, row in enumerate(todo.itertuples(index=False), 1):
        try:
            data = fetch_point(sess, row.lon, row.lat)
            props = _parse_response(data)
            props["landkreis"] = row.landkreis
            rows.append(props)
        except Exception as e:
            log.warning("SoilGrids gave up on %s: %s", row.landkreis, e)
            rows.append({"landkreis": row.landkreis,
                         **{p: None for p in PROPS}})

        # Persist progress every 10 districts so we never lose more than that.
        if i % 10 == 0 or i == len(todo):
            pd.DataFrame(rows).to_parquet(out_file, index=False)
            log.info("  progress %d/%d", i, len(todo))

        time.sleep(BASE_DELAY + random.uniform(0, 0.5))

    df = pd.DataFrame(rows)
    # Ensure all property columns exist even if some districts returned partial.
    for p in PROPS:
        if p not in df.columns:
            df[p] = None
    df.to_parquet(out_file, index=False)
    full = int(df[list(PROPS)].notna().all(axis=1).sum())
    log.info("Wrote %s (%d Landkreise, %d with full data)", out_file, len(df), full)
    return out_file


def clean(raw_path: str | Path, out_path: str | Path) -> Path:
    raw_path, out_path = Path(raw_path), Path(out_path)
    df = pd.read_parquet(raw_path)

    rename = {"phh2o": "ph", "bdod": "bulk_density"}
    df = df.rename(columns=rename)
    if "ph" in df:
        df["ph"] = df["ph"] / 10.0   # SoilGrids stores pH × 10
    if "soc" in df:
        df["soc"] = df["soc"] / 10.0  # dg/kg → g/kg

    # Preserve the fact that a value was unavailable before median imputation.
    # The aggregate flag is a safe static model feature and enables sensitivity
    # analysis on districts whose SoilGrids request failed.
    feat_cols = [c for c in ("sand", "clay", "soc", "ph", "bulk_density")
                 if c in df.columns]
    imputed_cols = []
    for c in feat_cols:
        indicator = f"{c}_imputed"
        df[indicator] = df[c].isna().astype("int8")
        imputed_cols.append(indicator)
        df[c] = df[c].fillna(df[c].median())
    df["soil_imputed"] = df[imputed_cols].max(axis=1).astype("int8")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    log.info("Cleaned soil → %s (%d Landkreise)", out_path, len(df))
    return out_path

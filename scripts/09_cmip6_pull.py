"""Download CMIP6 monthly climate projections from Copernicus CDS.

Fires N requests: {experiments} × {variables} using EC-Earth3-CC output.
Each request goes into the CDS queue server-side; the cdsapi client polls
until the request is ready then streams the file to
``data/raw/cmip6/{experiment}_{variable}.zip``.

Wall-clock: ~30 min to a few hours depending on CDS queue load.
Re-runs skip files that already exist, so the script is safely resumable
after any interruption.

Credentials: expects ``~/.cdsapirc`` (URL + key). If missing, cdsapi will
print an actionable error message.

Usage:
    python scripts/09_cmip6_pull.py
    python scripts/09_cmip6_pull.py --experiment historical
    python scripts/09_cmip6_pull.py --variable near_surface_air_temperature
    python scripts/09_cmip6_pull.py --dry-run   # list what would be pulled, do not submit
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cdsapi

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "raw" / "cmip6"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_MODEL = "ec_earth3_cc"
MODEL = DEFAULT_MODEL  # overridden via --model
TEMPORAL_RESOLUTION = "monthly"

MONTHS_ALL = [f"{m:02d}" for m in range(1, 13)]

# Historical covers our training window; SSP scenarios cover 2015-2050.
EXPERIMENTS = {
    "historical": [str(y) for y in range(2003, 2015)],   # 2003-2014
    "ssp2_4_5":   [str(y) for y in range(2015, 2051)],   # 2015-2050
    "ssp5_8_5":   [str(y) for y in range(2015, 2051)],   # 2015-2050
}

# Five variables — matches the training features (tmax/tmin/precip/radiation)
# plus a mean-temperature backup. Names use CDS's exact snake_case.
VARIABLES = [
    "near_surface_air_temperature",
    "daily_maximum_near_surface_air_temperature",
    "daily_minimum_near_surface_air_temperature",
    "precipitation",
    "surface_downwelling_shortwave_radiation",
]


def build_request(experiment: str, variable: str) -> dict:
    """Return the CDS request dict for one (experiment, variable) combo."""
    return {
        "temporal_resolution": TEMPORAL_RESOLUTION,
        "experiment": experiment,
        "variable": variable,
        "model": MODEL,
        "year": EXPERIMENTS[experiment],
        "month": MONTHS_ALL,
    }


def _fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def pull_one(client: cdsapi.Client, experiment: str, variable: str) -> Path:
    """Submit one request, wait for CDS to prepare it, download. Cached.
    Filenames include the model to keep multi-model pulls tidy; single-
    model pulls with the legacy default model preserve the old naming."""
    if MODEL == DEFAULT_MODEL:
        out_path = OUT_DIR / f"{experiment}_{variable}.zip"
    else:
        out_path = OUT_DIR / f"{MODEL}_{experiment}_{variable}.zip"
    if out_path.exists() and out_path.stat().st_size > 1024:
        print(f"  ✓ {out_path.name} exists ({_fmt_bytes(out_path.stat().st_size)}) — skip")
        return out_path

    print(f"  → submitting {experiment} / {variable}...")
    t0 = time.time()
    request = build_request(experiment, variable)
    client.retrieve("projections-cmip6", request).download(str(out_path))
    dt = time.time() - t0
    size = out_path.stat().st_size
    print(f"  ✓ wrote {out_path.name} ({_fmt_bytes(size)}) in {dt:.0f}s")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiment", choices=list(EXPERIMENTS),
                    help="Restrict to one experiment.")
    ap.add_argument("--variable", choices=VARIABLES,
                    help="Restrict to one variable.")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"CMIP6 model. Default: {DEFAULT_MODEL}. "
                         "Good alternatives for European agriculture: "
                         "mpi_esm1_2_hr, mpi_esm1_2_lr, mri_esm2_0.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the planned requests without submitting.")
    args = ap.parse_args()

    global MODEL
    MODEL = args.model

    exps = [args.experiment] if args.experiment else list(EXPERIMENTS)
    vars_ = [args.variable] if args.variable else VARIABLES
    combos = [(e, v) for e in exps for v in vars_]

    print(f"CMIP6 pull — {len(combos)} request(s)")
    print(f"model:       {MODEL}")
    print(f"temporal:    {TEMPORAL_RESOLUTION}")
    print(f"output dir:  {OUT_DIR}")
    print()

    if args.dry_run:
        for e, v in combos:
            print(f"  {e:12s}  {v}")
        return

    try:
        client = cdsapi.Client()
    except Exception as e:  # noqa: BLE001
        print(f"cdsapi init failed — is ~/.cdsapirc present + valid? {e}")
        sys.exit(1)

    ok, failed = 0, []
    for i, (exp, var) in enumerate(combos, 1):
        print(f"[{i}/{len(combos)}] {exp} × {var}")
        try:
            pull_one(client, exp, var)
            ok += 1
        except Exception as e:  # noqa: BLE001
            print(f"  ✗ FAILED: {type(e).__name__}: {str(e)[:200]}")
            failed.append((exp, var, str(e)[:120]))

    print(f"\nDone. {ok}/{len(combos)} downloaded, {len(failed)} failed.")
    if failed:
        print("Failures (re-run this script to retry — successful pulls are cached):")
        for e, v, msg in failed:
            print(f"  {e} × {v}: {msg}")


if __name__ == "__main__":
    main()

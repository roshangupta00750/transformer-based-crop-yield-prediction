"""Per-(landkreis, year) phenology proxy from MODIS NDVI daily series.

Fits a **double-logistic** curve to each district-year NDVI time series (from
``data/processed/gee_satellite.parquet`` after the LAI fix) and derives 7
phenology parameters usable as static-per-year features by TFT / XGBoost /
etc. This is a "proxy" for HR-VPP — cheap, self-contained, consistent across
all years including the pre-LAI-fix ones because it uses ONLY NDVI.

Model
-----
Double-logistic (Beck 2006, Fisher 2006):

    y(t) = base + amp * (1/(1+exp(-r1*(t-s1))) - 1/(1+exp(-r2*(t-s2))))

    base : baseline NDVI (dormant / bare soil)
    amp  : rise from baseline to the difference-of-sigmoids peak
    s1   : DOY of spring greenup midpoint  (rising-sigmoid inflection)
    s2   : DOY of autumn senescence midpoint (falling-sigmoid inflection)
    r1   : greenup rate
    r2   : senescence rate

Phenology derivation
--------------------
The 7 emitted features are computed by densely evaluating the *fitted* curve
at 1-day resolution over DOY 1..366, then:

    peak_doy       argmax_t  y_hat(t)
    peak_ndvi      y_hat(peak_doy)
    amplitude_ndvi peak_ndvi - base_fit
    sos_doy        first  t <  peak_doy where y_hat(t) >= base + 0.20*amp
    eos_doy        last   t >  peak_doy where y_hat(t) >= base + 0.20*amp
    los_days       eos_doy - sos_doy
    auc_growing    trapz(y_hat(t)) for t in [sos_doy, eos_doy]

Silent-bug prevention (per land-eng workflow synthesis)
-------------------------------------------------------
  * Bounded fit — base∈[0,0.5], amp∈[0,1], s1∈[60,180], s2∈[200,340],
    r1,r2∈[0.01,0.5]. Keeps the optimizer in physically plausible territory
    and avoids the failure mode where the peak lands in January.
  * Post-hoc s1>s2 → invalidate. If curve_fit still finds a solution where
    greenup happens *after* senescence, that district-year is garbage.
  * Amplitude floor — amp_fit<0.05 (essentially no seasonal signal) →
    invalidate. Prevents "phenology" being reported for perpetually-bare
    or perpetually-forested pixels.
  * At least 15 non-null NDVI observations per district-year, else skip.
  * If >20% of (landkreis, year) rows end up NaN we log a WARNING and refuse
    to overwrite any existing parquet — the merge script has to decide
    whether the columns are usable at all.
  * NDVI ONLY (no EVI, no LAI). The whole point of using NDVI is that the
    pre-LAI-fix years have identical NDVI to the post-fix years, so features
    are comparable across the 2003-2023 span.

Emits ``data/processed/static/phenology_proxy.parquet`` with columns:
    landkreis, year, sos_doy, eos_doy, los_days,
    peak_doy, peak_ndvi, amplitude_ndvi, auc_growing
"""
from __future__ import annotations

import logging
import os
import sys
import warnings
from multiprocessing import Pool, cpu_count
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

IN_FILE = REPO / "data" / "processed" / "gee_satellite.parquet"
OUT_DIR = REPO / "data" / "processed" / "static"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "phenology_proxy.parquet"

# --- Fit config -------------------------------------------------------------
MIN_POINTS = 15                # minimum non-null NDVI observations per year
DOY_GRID = np.arange(1, 367)   # evaluate fitted curve on this grid
SOS_EOS_FRAC = 0.20            # 20% of amplitude threshold (Jönsson & Eklundh)
MIN_AMP = 0.05                 # amplitude floor to accept a fit as "seasonal"
NAN_WARN_FRAC = 0.20           # if >20% of rows NaN, warn + refuse to write

# Parameter bounds: (base, amp, s1, s2, r1, r2)
LOWER = (0.0,  0.0, 60.0,  200.0, 0.01, 0.01)
UPPER = (0.5,  1.0, 180.0, 340.0, 0.5,  0.5)

# Feature column order (kept explicit for downstream stability)
FEATURE_COLS = [
    "sos_doy", "eos_doy", "los_days",
    "peak_doy", "peak_ndvi", "amplitude_ndvi", "auc_growing",
]

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("phenology_proxy")


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def double_logistic(t, base, amp, s1, s2, r1, r2):
    """Beck-style double-logistic. t may be scalar or ndarray."""
    # Clip exponent argument to avoid overflow warnings — the sigmoids
    # saturate anyway.
    a1 = np.clip(-r1 * (t - s1), -50.0, 50.0)
    a2 = np.clip(-r2 * (t - s2), -50.0, 50.0)
    return base + amp * (1.0 / (1.0 + np.exp(a1)) - 1.0 / (1.0 + np.exp(a2)))


def _initial_guess(doy: np.ndarray, ndvi: np.ndarray) -> tuple[float, ...]:
    """Cheap heuristic starting point staying inside the bounds."""
    lo = float(np.nanmin(ndvi))
    hi = float(np.nanmax(ndvi))
    base0 = float(np.clip(lo, LOWER[0], UPPER[0]))
    amp0 = float(np.clip(hi - lo, MIN_AMP, UPPER[1]))
    # Rough peak location as a data-driven anchor for s1/s2 midpoints.
    try:
        peak_doy = float(doy[np.nanargmax(ndvi)])
    except (ValueError, IndexError):
        peak_doy = 180.0
    s1_0 = float(np.clip(peak_doy - 60.0, LOWER[2], UPPER[2]))
    s2_0 = float(np.clip(peak_doy + 60.0, LOWER[3], UPPER[3]))
    return (base0, amp0, s1_0, s2_0, 0.1, 0.1)


# --------------------------------------------------------------------------- #
# Per-(landkreis, year) fit + phenology
# --------------------------------------------------------------------------- #
def _phenology_from_curve(y_hat: np.ndarray, base_fit: float
                          ) -> tuple[float, float, float, float, float, float, float] | None:
    """Derive the 7 phenology features from a densely-sampled fitted curve.

    Returns None if the curve is unusable (amplitude too small, no rising
    crossing, no falling crossing).
    """
    peak_idx = int(np.argmax(y_hat))
    peak_doy = float(DOY_GRID[peak_idx])
    peak_ndvi = float(y_hat[peak_idx])

    amplitude = peak_ndvi - base_fit
    if amplitude < MIN_AMP:
        return None

    threshold = base_fit + SOS_EOS_FRAC * amplitude

    # SOS: first day before peak where the curve rises above threshold.
    rising = np.where(y_hat[:peak_idx + 1] >= threshold)[0]
    if rising.size == 0:
        return None
    sos_idx = int(rising[0])
    sos_doy = float(DOY_GRID[sos_idx])

    # EOS: last day after peak where the curve is still above threshold.
    tail = y_hat[peak_idx:]
    still_above = np.where(tail >= threshold)[0]
    if still_above.size == 0:
        return None
    eos_idx = peak_idx + int(still_above[-1])
    eos_doy = float(DOY_GRID[eos_idx])

    los = eos_doy - sos_doy
    if los <= 0:
        return None

    # Trapezoidal integral of the fitted curve over the growing season.
    auc = float(np.trapz(y_hat[sos_idx:eos_idx + 1], DOY_GRID[sos_idx:eos_idx + 1]))

    return sos_doy, eos_doy, los, peak_doy, peak_ndvi, amplitude, auc


def _fit_one(args: tuple[str, int, np.ndarray, np.ndarray]
             ) -> tuple[str, int, dict, str]:
    """Fit one (landkreis, year). Returns (lk, year, feature_dict, status).

    status is one of: 'ok', 'too_few', 'fit_failed', 'invalid_shape',
    'invalid_pheno'. Kept for logging aggregate drop causes.
    """
    lk, year, doy, ndvi = args
    empty = {c: np.nan for c in FEATURE_COLS}

    if ndvi.size < MIN_POINTS:
        return lk, year, empty, "too_few"

    p0 = _initial_guess(doy, ndvi)
    try:
        popt, _ = curve_fit(
            double_logistic, doy, ndvi,
            p0=p0, bounds=(LOWER, UPPER),
            maxfev=5000,
        )
    except (RuntimeError, ValueError, TypeError):
        return lk, year, empty, "fit_failed"

    base_fit, amp_fit, s1_fit, s2_fit, _r1, _r2 = popt

    # Silent-bug guard: greenup after senescence is nonsense — reject.
    if s1_fit >= s2_fit:
        return lk, year, empty, "invalid_shape"

    # Silent-bug guard: fitted amplitude too small — no seasonal signal.
    if amp_fit < MIN_AMP:
        return lk, year, empty, "invalid_shape"

    y_hat = double_logistic(DOY_GRID, *popt)
    pheno = _phenology_from_curve(y_hat, base_fit)
    if pheno is None:
        return lk, year, empty, "invalid_pheno"

    sos, eos, los, pk_doy, pk_ndvi, amp, auc = pheno
    feats = {
        "sos_doy": sos,
        "eos_doy": eos,
        "los_days": los,
        "peak_doy": pk_doy,
        "peak_ndvi": pk_ndvi,
        "amplitude_ndvi": amp,
        "auc_growing": auc,
    }
    return lk, year, feats, "ok"


# --------------------------------------------------------------------------- #
# Data loading / iteration
# --------------------------------------------------------------------------- #
def _load_ndvi() -> pd.DataFrame:
    if not IN_FILE.exists():
        raise FileNotFoundError(
            f"{IN_FILE} not found. Run the GEE backfill (with the LAI fix) first."
        )
    log.info("Loading %s ...", IN_FILE)
    df = pd.read_parquet(IN_FILE, columns=["landkreis", "date", "ndvi"])
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["doy"] = df["date"].dt.dayofyear
    # Only rows with a real NDVI observation matter for the fit.
    n_pre = len(df)
    df = df.dropna(subset=["ndvi"])
    log.info("  %d rows total, %d with non-null NDVI", n_pre, len(df))
    log.info("  %d landkreise, years %d..%d",
             df["landkreis"].nunique(), df["year"].min(), df["year"].max())
    return df


def _iter_groups(df: pd.DataFrame):
    """Yield (landkreis, year, doy_array, ndvi_array) tuples."""
    for (lk, yr), sub in df.groupby(["landkreis", "year"], sort=False):
        # Deduplicate to at most one obs per DOY (average if MOD13Q1 happens
        # to have two composites hitting the same day after date coercion).
        agg = sub.groupby("doy", as_index=False)["ndvi"].mean()
        yield lk, int(yr), agg["doy"].to_numpy(dtype=float), agg["ndvi"].to_numpy(dtype=float)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    df = _load_ndvi()

    tasks = list(_iter_groups(df))
    n_total = len(tasks)
    log.info("Fitting %d (landkreis, year) NDVI curves...", n_total)

    n_jobs = max(1, min(cpu_count() - 1, 8))
    log.info("  using %d worker processes", n_jobs)

    results = []
    if n_jobs == 1:
        for i, t in enumerate(tasks, 1):
            results.append(_fit_one(t))
            if i % 500 == 0:
                log.info("  %d / %d fits done", i, n_total)
    else:
        with Pool(processes=n_jobs) as pool:
            for i, res in enumerate(pool.imap_unordered(_fit_one, tasks, chunksize=32), 1):
                results.append(res)
                if i % 500 == 0:
                    log.info("  %d / %d fits done", i, n_total)

    # Aggregate outcomes for diagnostics.
    status_counts: dict[str, int] = {}
    rows = []
    for lk, yr, feats, status in results:
        status_counts[status] = status_counts.get(status, 0) + 1
        rows.append({"landkreis": lk, "year": yr, **feats})
    out = pd.DataFrame(rows).sort_values(["landkreis", "year"]).reset_index(drop=True)

    log.info("Outcome breakdown:")
    for k in sorted(status_counts):
        log.info("  %-15s %d", k, status_counts[k])

    n_nan = out["sos_doy"].isna().sum()
    nan_frac = n_nan / max(1, len(out))
    log.info("Rows with NaN features: %d / %d (%.1f%%)", n_nan, len(out), 100 * nan_frac)

    # Silent-bug guard: refuse to write a mostly-empty table.
    if nan_frac > NAN_WARN_FRAC:
        log.warning(
            "More than %.0f%% of (landkreis, year) rows are NaN — this proxy is "
            "unreliable for this NDVI dataset. NOT overwriting %s. Delete this "
            "file manually and re-run if you want to force emission, or let "
            "the merge script decide whether to drop the columns.",
            100 * NAN_WARN_FRAC, OUT_FILE,
        )
        # Write a sidecar so the failed attempt is at least visible.
        sidecar = OUT_FILE.with_suffix(".REJECTED.parquet")
        out.to_parquet(sidecar, index=False)
        log.warning("  wrote diagnostic sidecar: %s", sidecar)
        sys.exit(2)

    out.to_parquet(OUT_FILE, index=False)
    log.info("Wrote %s (%d rows, %d cols)", OUT_FILE, len(out), out.shape[1])

    # Quick sanity print.
    print()
    print("sample rows (first 5 non-NaN):")
    ok = out.dropna(subset=["sos_doy"]).head(5)
    if not ok.empty:
        print(ok.to_string(index=False))
    print()
    print("per-feature non-null counts:")
    for c in ["landkreis", "year", *FEATURE_COLS]:
        print(f"  {c:18s} {out[c].notna().sum():>6d} / {len(out)}")
    print()
    print("per-feature summary (non-NaN only):")
    print(out[FEATURE_COLS].describe().round(2).to_string())


if __name__ == "__main__":
    main()

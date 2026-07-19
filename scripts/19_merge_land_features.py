"""Merge the three new static/annual land feature groups into the unified dataset.

Left-joins into ``data/processed/dataset.parquet``:
  1. ``data/processed/static/dem_features.parquet``       (per landkreis)
       elevation_mean, elevation_std, elevation_min, elevation_max,
       slope_mean, slope_std, aspect_cos_mean, aspect_sin_mean
  2. ``data/processed/static/landcover_features.parquet`` (per landkreis)
       frac_arable, frac_permanent_crops, frac_pasture, frac_forest,
       frac_urban, frac_water, frac_other
  3. ``data/processed/static/phenology_proxy.parquet``    (per landkreis+year)
       sos_doy, eos_doy, los_days, peak_doy, peak_ndvi,
       amplitude_ndvi, auc_growing

Join order (each is a strict LEFT JOIN — dataset row count is preserved):
    dataset <- dem (on landkreis)
            <- landcover (on landkreis)
            <- phenology (on [landkreis, year])

Silent-bug prevention (verbatim from the land-eng workflow synthesis):
  1. Extend ``CANDIDATE_STATIC`` in ``scripts/03_train.py:45`` — new statics
     will NOT reach the model otherwise.
  2. Mirror at ``scripts/06b_loo_xgboost.py:95`` startswith allowlist —
     new prefixes (elevation_, slope_, aspect_, frac_, sos_, eos_, los_,
     peak_, amplitude_, auc_) must be added or LOO silently drops them.
  3. Delete ``results/metrics/fusion_predictions_{val,test}.parquet`` BEFORE
     the next fusion run — else ``07_fusion.py`` reuses stale preds.
  4. Version ``results/checkpoints/`` BEFORE retrain — the 9 saved pipelines
     will hard-error at load with new feature dims; must preserve the
     pre-change baseline for reproducibility.
  5. NaN-fill allowlists in ``src/data/dataset.py:102-104`` must include the
     new columns (this script pre-fills DEM/landcover with the per-column
     mean and phenology with the per-landkreis mean, so downstream code
     does not need to change — but the fill-list still SHOULD be extended
     if ``build_unified_dataset`` is ever re-run and the merge is inlined
     there).

Usage
-----
    python scripts/19_merge_land_features.py                     # merge + write
    python scripts/19_merge_land_features.py --dry-run           # inspect only
    python scripts/19_merge_land_features.py --no-backup         # skip snapshotting
    python scripts/19_merge_land_features.py --skip-missing      # tolerate absent inputs
"""
from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


DATASET_PATH = REPO / "data" / "processed" / "dataset.parquet"
STATIC_DIR = REPO / "data" / "processed" / "static"
DEM_PATH = STATIC_DIR / "dem_features.parquet"
LANDCOVER_PATH = STATIC_DIR / "landcover_features.parquet"
PHENOLOGY_PATH = STATIC_DIR / "phenology_proxy.parquet"

CHECKPOINT_DIR = REPO / "results" / "checkpoints"
CHECKPOINT_ARCHIVE_ROOT = REPO / "results" / "checkpoints_archive"
FUSION_VAL = REPO / "results" / "metrics" / "fusion_predictions_val.parquet"
FUSION_TEST = REPO / "results" / "metrics" / "fusion_predictions_test.parquet"

# Canonical expected columns for each source. Kept explicit so a schema drift
# in an upstream feature-builder is caught here, not silently at train time.
DEM_COLS = [
    "elevation_mean", "elevation_std", "elevation_min", "elevation_max",
    "slope_mean", "slope_std", "aspect_cos_mean", "aspect_sin_mean",
]
LANDCOVER_COLS = [
    "frac_arable", "frac_permanent_crops", "frac_pasture",
    "frac_forest", "frac_urban", "frac_water", "frac_other",
]
PHENOLOGY_COLS = [
    "sos_doy", "eos_doy", "los_days", "peak_doy",
    "peak_ndvi", "amplitude_ndvi", "auc_growing",
]

# Prefixes downstream code (03_train.py, 06b_loo_xgboost.py) must allow.
NEW_STATIC_PREFIXES = (
    "elevation_", "slope_", "aspect_",
    "frac_",
    "sos_", "eos_", "los_", "peak_", "amplitude_", "auc_",
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _load(path: Path, expected_cols: list[str], join_keys: list[str],
          skip_missing: bool) -> pd.DataFrame | None:
    """Load one feature parquet, verify schema, return None if allowed-missing."""
    if not path.exists():
        msg = f"missing feature parquet: {path.relative_to(REPO)}"
        if skip_missing:
            print(f"[warn] {msg} — skipping (--skip-missing)")
            return None
        raise FileNotFoundError(msg)

    df = pd.read_parquet(path)
    missing = [c for c in expected_cols + join_keys if c not in df.columns]
    if missing:
        raise ValueError(
            f"{path.name}: missing expected columns {missing}. "
            f"Present: {df.columns.tolist()}"
        )
    # Enforce uniqueness on join keys — a stray duplicate would fan out the
    # 15M-row dataset into a many-to-many explosion.
    dup = df.duplicated(subset=join_keys).sum()
    if dup:
        raise ValueError(
            f"{path.name}: {dup} duplicate rows on join keys {join_keys}. "
            "Left-join would explode row count; fix upstream."
        )
    # Keep only join keys + expected feature columns (drop any extras so the
    # merged dataset schema stays predictable and free of accidental columns).
    return df[join_keys + expected_cols]


def _schema_diff(before: pd.DataFrame, after: pd.DataFrame) -> list[str]:
    """Columns present in `after` but not `before`, preserving `after`'s order."""
    before_set = set(before.columns)
    return [c for c in after.columns if c not in before_set]


def _null_report(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    n = len(df)
    return pd.DataFrame({
        "column": cols,
        "n_null": [int(df[c].isna().sum()) for c in cols],
        "pct_null": [
            round(100.0 * df[c].isna().sum() / n, 3) if n else 0.0
            for c in cols
        ],
        "n_rows": n,
    })


def _fill_static_broadcast(
    df: pd.DataFrame, cols: list[str], group_key: str = "landkreis"
) -> pd.DataFrame:
    """Fill per-landkreis missing values with the column-wide mean.

    For per-landkreis broadcast features (dem, landcover), a NaN means the
    landkreis is entirely absent from the source parquet — no per-district
    imputation is possible. Fall back to the column mean so downstream
    z-scoring doesn't produce all-NaN.
    """
    for c in cols:
        if c not in df.columns:
            continue
        mu = df[c].mean()
        n_missing_districts = df.loc[df[c].isna(), group_key].nunique()
        if n_missing_districts:
            print(f"    fill {c}: {n_missing_districts} districts NaN "
                  f"-> filled with column mean ({mu:.4g})")
        df[c] = df[c].fillna(mu)
        # Belt-and-braces: if the column is entirely NaN, mean is NaN too.
        df[c] = df[c].fillna(0.0)
    return df


def _fill_phenology(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Fill phenology NaNs with the per-landkreis mean, then column mean."""
    for c in cols:
        if c not in df.columns:
            continue
        # Per-landkreis mean over years first (a district might miss one year).
        df[c] = df.groupby("landkreis")[c].transform(
            lambda s: s.fillna(s.mean()))
        # Then column mean for districts fully absent from phenology.
        mu = df[c].mean()
        n_missing = int(df[c].isna().sum())
        if n_missing:
            print(f"    fill {c}: {n_missing} row-NaN after per-district "
                  f"fill -> filled with column mean ({mu:.4g})")
        df[c] = df[c].fillna(mu).fillna(0.0)
    return df


def _archive_checkpoints() -> Path | None:
    """Snapshot ``results/checkpoints/`` before overwriting.

    The 9 saved pipeline JSONs pin the feature-list they were trained with;
    loading them after new statics are added raises a dim-mismatch error. A
    dated copy lets us reproduce the pre-change baseline.
    """
    if not CHECKPOINT_DIR.exists():
        print("[archive] no results/checkpoints/ to snapshot")
        return None
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    dst = CHECKPOINT_ARCHIVE_ROOT / f"pre_land_features_{stamp}"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(CHECKPOINT_DIR, dst)
    print(f"[archive] snapshotted {CHECKPOINT_DIR.relative_to(REPO)}"
          f" -> {dst.relative_to(REPO)}")
    return dst


def _clear_fusion_cache() -> None:
    """Delete stale fusion prediction parquets.

    ``07_fusion.py`` will reuse whatever is on disk; leaving pre-change
    predictions in place produces a metric that mixes old and new features
    without warning.
    """
    for p in (FUSION_VAL, FUSION_TEST):
        if p.exists():
            p.unlink()
            print(f"[fusion-cache] removed stale {p.relative_to(REPO)}")


def _remind_downstream_edits(new_cols: list[str]) -> None:
    """Print the exact downstream edits the caller still owes."""
    print("\n" + "=" * 72)
    print("REMINDERS — downstream edits required before retraining")
    print("=" * 72)
    print("  1. scripts/03_train.py:45 CANDIDATE_STATIC — extend with:")
    for c in new_cols:
        print(f"        \"{c}\",")
    print("  2. scripts/06b_loo_xgboost.py:95 static allowlist — replace the")
    print("     current `c in (\"sand\",...)` check with a startswith over")
    print("     these new prefixes, kept alongside the existing static names:")
    print(f"        prefixes: {NEW_STATIC_PREFIXES}")
    print("        names:    ('sand', 'clay', 'soc', 'ph', 'bulk_density',")
    print("                   'landkreis_id')")
    print("  3. src/data/dataset.py:102-104 NaN-fill allowlists — extend so")
    print("     future re-runs of build_unified_dataset also fill these.")
    print("  4. Fusion predictions cache: cleared automatically (see above).")
    print("  5. Checkpoints archive: snapshotted automatically (see above).")
    print("=" * 72 + "\n")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="Print schema diff + null counts, do NOT overwrite "
                         "dataset.parquet, do NOT archive checkpoints, do "
                         "NOT clear fusion cache.")
    ap.add_argument("--no-backup", action="store_true",
                    help="Skip archiving results/checkpoints/. Only use if "
                         "you have already versioned them separately.")
    ap.add_argument("--skip-missing", action="store_true",
                    help="Tolerate a missing landcover_features.parquet or "
                         "phenology_proxy.parquet (dem is required).")
    ap.add_argument("--out", default=None,
                    help="Alternate output path. Default: overwrite "
                         "data/processed/dataset.parquet.")
    args = ap.parse_args()

    if not DATASET_PATH.exists():
        print(f"[error] {DATASET_PATH.relative_to(REPO)} not found — "
              "run scripts/02_preprocess.py first.", file=sys.stderr)
        sys.exit(1)

    print(f"[load] {DATASET_PATH.relative_to(REPO)}")
    df = pd.read_parquet(DATASET_PATH)
    n_rows_before = len(df)
    cols_before = df.columns.tolist()
    print(f"       {n_rows_before:,} rows, {len(cols_before)} cols")

    # DEM is required (it's the only file that must exist regardless of
    # --skip-missing) — the workflow presumes DEM as the baseline static.
    print(f"[load] {DEM_PATH.relative_to(REPO)}")
    dem = _load(DEM_PATH, DEM_COLS, ["landkreis"], skip_missing=False)

    print(f"[load] {LANDCOVER_PATH.relative_to(REPO)}")
    landcover = _load(LANDCOVER_PATH, LANDCOVER_COLS, ["landkreis"],
                      skip_missing=args.skip_missing)

    print(f"[load] {PHENOLOGY_PATH.relative_to(REPO)}")
    phenology = _load(PHENOLOGY_PATH, PHENOLOGY_COLS, ["landkreis", "year"],
                      skip_missing=args.skip_missing)

    # --------------------------------------------------------------------- #
    # Left joins — order matters for column layout but not for correctness.
    # --------------------------------------------------------------------- #
    print("\n[merge] dataset <- dem  (on landkreis)")
    df = df.merge(dem, on="landkreis", how="left")
    assert len(df) == n_rows_before, (
        f"row count changed after DEM merge: {n_rows_before} -> {len(df)}"
    )
    df = _fill_static_broadcast(df, DEM_COLS)

    if landcover is not None:
        print("[merge] dataset <- landcover  (on landkreis)")
        df = df.merge(landcover, on="landkreis", how="left")
        assert len(df) == n_rows_before, (
            f"row count changed after landcover merge: "
            f"{n_rows_before} -> {len(df)}"
        )
        df = _fill_static_broadcast(df, LANDCOVER_COLS)
    else:
        print("[merge] skipping landcover (missing + --skip-missing)")

    if phenology is not None:
        print("[merge] dataset <- phenology  (on [landkreis, year])")
        df = df.merge(phenology, on=["landkreis", "year"], how="left")
        assert len(df) == n_rows_before, (
            f"row count changed after phenology merge: "
            f"{n_rows_before} -> {len(df)}"
        )
        df = _fill_phenology(df, PHENOLOGY_COLS)
    else:
        print("[merge] skipping phenology (missing + --skip-missing)")

    new_cols = _schema_diff(pd.DataFrame(columns=cols_before), df)

    # --------------------------------------------------------------------- #
    # Report
    # --------------------------------------------------------------------- #
    print("\n" + "-" * 72)
    print(f"Schema diff: +{len(new_cols)} columns")
    print("-" * 72)
    for c in new_cols:
        print(f"  + {c}")

    print("\n" + "-" * 72)
    print("Null-count report (post-fill; non-zero would indicate a bug)")
    print("-" * 72)
    print(_null_report(df, new_cols).to_string(index=False))

    print(f"\nFinal shape: {df.shape[0]:,} rows × {df.shape[1]} cols "
          f"(was {n_rows_before:,} × {len(cols_before)})")

    # --------------------------------------------------------------------- #
    # Write (or bail out on dry-run)
    # --------------------------------------------------------------------- #
    if args.dry_run:
        print("\n[dry-run] no writes performed. "
              "Re-run without --dry-run to persist.")
        return

    if not args.no_backup:
        _archive_checkpoints()
    else:
        print("[archive] --no-backup set; NOT snapshotting checkpoints "
              "(caller asserts they are already versioned).")

    _clear_fusion_cache()

    out_path = Path(args.out) if args.out else DATASET_PATH
    # Snapshot the dataset itself too, in case the caller wants to roll back
    # without re-running the whole preprocess pipeline.
    if out_path == DATASET_PATH:
        stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        backup = DATASET_PATH.with_suffix(f".pre_land.{stamp}.parquet")
        shutil.copy2(DATASET_PATH, backup)
        print(f"[archive] {DATASET_PATH.name} -> {backup.name}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"[write] {out_path.relative_to(REPO)}  "
          f"({df.shape[0]:,} rows × {df.shape[1]} cols)")

    _remind_downstream_edits(new_cols)


if __name__ == "__main__":
    main()

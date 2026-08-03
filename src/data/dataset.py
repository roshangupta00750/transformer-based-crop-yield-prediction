"""Unified dataset, feature pipeline, and PyTorch wrapper.

Layout of the unified parquet (one row per Landkreis-year-day):
    landkreis, crop, year, doy, date,
    <weather features>, <satellite features (optional)>,
    <soil features (optional)>, yield_dt_per_ha

For training, this is collapsed to time-series tensors with:
    X_past:    (B, T, F)   past observed (weather + satellite history)
    X_static:  (B, S)      static covariates (soil, district-mean yield, district id)
    X_future:  (B, T, F')  known-future inputs (calendar)
    y:         (B,)        z-scored yield anomaly (target)

Two things matter for R²:
  1. **Normalization**: features and target are z-scored using stats fit on
     *training years only*. Inverse-transform converts model output back to
     absolute yield (dt/ha) so metrics are reported in physical units.
  2. **Anomaly framing**: we predict `yield - per_district_mean(train)` instead
     of absolute yield. The district baseline carries most of the variance, so
     models that fail to learn it look terrible; subtracting it lets the model
     focus on year-to-year weather/satellite-driven deviations.
  3. **Phenology windowing**: each (landkreis, year, crop) sample is sliced to
     the crop's growing season (start_doy..harvest_doy) so the model isn't
     fed irrelevant post-harvest weather.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

FEATURE_POLICY_VERSION = 3
SATELLITE_MAX_AGE_DAYS = {
    "ndvi": 32,
    "evi": 32,
    "lai": 16,
    "ndwi": 8,
}

try:
    import torch
    from torch.utils.data import Dataset
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    Dataset = object  # type: ignore


@dataclass
class FeatureSpec:
    """Names of columns in the unified table, by category."""
    past_observed: list[str]
    static: list[str]
    known_future: list[str]
    target: str = "yield_dt_per_ha"


def dataset_policy_manifest_path(dataset_path: str | Path) -> Path:
    """Return the sidecar path used to prove safe preprocessing was run."""
    return Path(dataset_path).with_suffix(".policy.json")


def require_current_dataset_policy(dataset_path: str | Path) -> None:
    """Reject legacy processed data that hid missingness non-causally."""
    manifest_path = dataset_policy_manifest_path(dataset_path)
    if not manifest_path.exists():
        raise RuntimeError(
            f"{manifest_path.name} is missing. Rebuild the processed dataset "
            "with `python scripts/02_preprocess.py` before retraining."
        )
    manifest = json.loads(manifest_path.read_text())
    version = int(manifest.get("feature_policy_version", 1))
    if version != FEATURE_POLICY_VERSION:
        raise RuntimeError(
            f"processed dataset uses policy v{version}; current policy is "
            f"v{FEATURE_POLICY_VERSION}. Rebuild it before retraining."
        )


def build_feature_spec(
    df: pd.DataFrame,
    data_config: dict,
    static_features: Sequence[str] | None = None,
) -> FeatureSpec:
    """Build a leakage-safe feature specification from ``configs/data.yaml``.

    The configured ``excluded`` list is enforced even when a per-model static
    override names an unsafe feature. Missing optional columns are ignored so
    the current parquet remains usable while imputation indicators are added
    on the next data rebuild.
    """
    configured = data_config.get("features")
    if not configured:
        raise KeyError(
            "data config is missing the required leakage-safe `features` section"
        )

    excluded = set(configured.get("excluded", []))

    def available(names: Sequence[str]) -> list[str]:
        return [c for c in names if c in df.columns and c not in excluded]

    static_pool = (
        list(static_features)
        if static_features is not None
        else configured.get("static", [])
    )
    return FeatureSpec(
        past_observed=available(configured.get("past_observed", [])),
        static=available(static_pool),
        known_future=available(configured.get("known_future", [])),
    )


# --------------------------------------------------------------------------- #
# Building the unified table from per-source interim files
# --------------------------------------------------------------------------- #
def build_unified_dataset(
    interim_dir: str | Path,
    out_path: str | Path,
    crops: Sequence[str],
    year_range: tuple[int, int],
) -> pd.DataFrame:
    interim_dir = Path(interim_dir)
    out_path = Path(out_path)

    weather = pd.read_parquet(interim_dir / "weather.parquet")
    yld = pd.read_parquet(interim_dir / "yield.parquet")
    sat_path = interim_dir / "satellite.parquet"
    satellite = pd.read_parquet(sat_path) if sat_path.exists() else None
    soil_path = interim_dir / "soil.parquet"
    soil = pd.read_parquet(soil_path) if soil_path.exists() else None

    yld = yld[(yld.year >= year_range[0]) & (yld.year <= year_range[1])]
    yld = yld[yld.crop.isin(crops)]

    weather["date"] = pd.to_datetime(weather["date"]).dt.tz_localize(None)
    if satellite is not None:
        satellite["date"] = (pd.to_datetime(satellite["date"], utc=True)
                              .dt.tz_localize(None)
                              .dt.normalize())
        df = weather.merge(satellite, on=["landkreis", "date"], how="left")
        df = df.sort_values(["landkreis", "date"])
        # Weather can omit the exact day of a native MODIS composite, so retain
        # the age calculated on the complete satellite calendar. Recomputing
        # age after this left join would falsely create multi-year gaps.
        for index_name, max_age in SATELLITE_MAX_AGE_DAYS.items():
            observed_col = f"{index_name}_observed"
            age_col = f"{index_name}_age_days"
            stale_col = f"{index_name}_stale"
            if observed_col not in df.columns:
                continue
            df[observed_col] = df[observed_col].fillna(0).astype("int8")
            df[age_col] = df[age_col].fillna(max_age + 1)
            if stale_col in df.columns:
                df[stale_col] = df[stale_col].fillna(1).astype("int8")
    else:
        df = weather
    df["year"] = df["date"].dt.year
    df["doy"] = df["date"].dt.dayofyear

    # Left-join yield so unlabelled future years (e.g. 2024 forecast when
    # Destatis has not published yet) still land in dataset.parquet with
    # yield_dt_per_ha = NaN. Training code filters on target-notna, so 2024
    # rows are naturally excluded from training. But 2024 rows DO need a
    # crop assignment for the target years — inject one row per crop per
    # (landkreis, year) with NaN target so per-crop inference paths still work.
    labelled_years = sorted(yld["year"].unique())
    all_years = sorted(df["year"].unique())
    unlabelled_years = [y for y in all_years if y not in labelled_years]
    df_labelled = df.merge(yld, on=["landkreis", "year"], how="inner")
    if unlabelled_years:
        df_unlab = df[df["year"].isin(unlabelled_years)].copy()
        crops_present = sorted(yld["crop"].unique())
        # Cross-join unlabelled years with all crops so inference sweeps each
        # (crop, district, year=unlabelled).
        df_unlab = df_unlab.merge(pd.DataFrame({"crop": crops_present}), how="cross")
        df_unlab["yield_dt_per_ha"] = np.nan
        # Align columns and concat
        cols = df_labelled.columns
        df_unlab = df_unlab.reindex(columns=cols)
        df = pd.concat([df_labelled, df_unlab], ignore_index=True)
    else:
        df = df_labelled
    if soil is not None:
        df = df.merge(soil, on="landkreis", how="left")

    # Preserve descriptive land features on a normal rebuild. Predictive
    # eligibility is still controlled exclusively by configs/data.yaml.
    static_dir = out_path.parent / "static"
    optional_static = [
        (static_dir / "dem_features.parquet", ["landkreis"]),
        (static_dir / "landcover_features.parquet", ["landkreis"]),
        (static_dir / "phenology_proxy.parquet", ["landkreis", "year"]),
    ]
    for static_path, keys in optional_static:
        if not static_path.exists():
            continue
        static = pd.read_parquet(static_path)
        overlap = [c for c in static.columns if c in df.columns and c not in keys]
        if overlap:
            static = static.drop(columns=overlap)
        rows_before = len(df)
        df = df.merge(static, on=keys, how="left", validate="many_to_one")
        if len(df) != rows_before:
            raise RuntimeError(
                f"row count changed while merging {static_path.name}"
            )

    df["sin_doy"] = np.sin(2 * np.pi * df["doy"] / 365)
    df["cos_doy"] = np.cos(2 * np.pi * df["doy"] / 365)

    # Integer label-encode landkreis so it can be used as a static feature.
    # The order is stable (sorted NUTS-3 codes) so encodings are consistent
    # across runs / between train and eval.
    codes = sorted(df["landkreis"].unique())
    code_to_id = {c: i for i, c in enumerate(codes)}
    df["landkreis_id"] = df["landkreis"].map(code_to_id).astype("int32")

    # Record missingness before filling so models can distinguish measurements
    # from carried-forward values. Fill causally within each district only:
    # bfill would use a future observation, while a global mean computed here
    # would leak validation/test distribution information. Any leading NaNs are
    # left for FeaturePipeline.transform_features(), which uses train-only means.
    weather_cols = [c for c in df.columns
                    if c.startswith(("temperature_", "precip", "radiation", "humid"))]
    sat_cols = [c for c in df.columns if c in ("ndvi", "evi", "lai", "ndwi")]
    indicator_cols = weather_cols + sat_cols
    if indicator_cols:
        df = df.sort_values(["landkreis", "date"])
        for c in indicator_cols:
            indicator = f"{c}_missing"
            if indicator not in df.columns:
                df[indicator] = df[c].isna().astype("int8")
        # Weather is daily, so carry at most one week. Satellite composites
        # were already filled with source-specific expiry limits by their
        # cleaner and must not be filled again here.
        df[weather_cols] = (
            df.groupby("landkreis", sort=False)[weather_cols].ffill(limit=7)
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    dataset_policy_manifest_path(out_path).write_text(
        json.dumps(
            {
                "feature_policy_version": FEATURE_POLICY_VERSION,
                "rows": len(df),
                "columns": list(df.columns),
                "missing_value_policy": (
                    "bounded_causal_ffill_then_train_mean"
                ),
            },
            indent=2,
        )
    )
    return df


# --------------------------------------------------------------------------- #
# FeaturePipeline — fits normalization stats on the train split and applies
# them everywhere. Also handles the anomaly framing for the target.
# --------------------------------------------------------------------------- #
class FeaturePipeline:
    """Per-column z-scoring + optional per-(landkreis, crop) target anomaly framing.

    Two modes:
      - ``anomaly_framing=True`` (default): target = (yield − district_mean) / std.
        R² on inverse-transformed predictions measures within-district year-to-
        year skill (hard).
      - ``anomaly_framing=False``: target = (yield − global_mean) / global_std.
        Standard absolute-yield z-scoring. Easier R² because district baselines
        carry most variance; the model needs district info (e.g. district_id
        encoded as a static feature) to take advantage.

    Fit only on training rows. ``transform`` is applied to all splits.
    """

    def __init__(self, feature_spec: FeatureSpec,
                 anomaly_framing: bool = True) -> None:
        self.spec = feature_spec
        self.anomaly_framing = anomaly_framing
        self.feature_policy_version = FEATURE_POLICY_VERSION
        self.feature_mean_: dict[str, float] = {}
        self.feature_std_: dict[str, float] = {}
        self.target_mean_: dict[tuple[str, str], float] = {}  # (landkreis, crop) -> mean
        self.target_mean_by_crop_: dict[str, float] = {}     # crop -> mean (fallback)
        self.target_trend_by_crop_: dict[str, float] = {}    # crop -> dt/ha per year
        self.trend_ref_year_: float = 0.0
        self.target_global_mean_: float = 0.0
        self.target_global_std_: float = 1.0
        self.target_anomaly_std_: float = 1.0          # global fallback
        self.target_anomaly_std_by_crop_: dict[str, float] = {}  # per-crop
        # Plausible yield bounds per crop, used to clip predictions so a single
        # out-of-distribution district can't produce a physically impossible
        # value (e.g. 175 dt/ha rapeseed) that wrecks pooled RMSE / R².
        self.target_clip_by_crop_: dict[str, list[float]] = {}
        self.sequence_length_: int | None = None
        # Encoder window the TFT checkpoint was trained with. Evaluation
        # must rebuild its reference TimeSeriesDataSet with the same value
        # or the model sees a different context length than it learned on.
        # None on pre-existing checkpoints, which all used 180.
        self.tft_max_encoder_length_: int | None = None
        self.fitted_ = False

    def fit(self, df_train: pd.DataFrame) -> "FeaturePipeline":
        cols = (self.spec.past_observed + self.spec.static + self.spec.known_future)
        for c in cols:
            if c not in df_train.columns:
                continue
            v = pd.to_numeric(df_train[c], errors="coerce")
            mu, sd = float(v.mean()), float(v.std())
            self.feature_mean_[c] = mu
            self.feature_std_[c] = sd if sd > 1e-9 else 1.0

        # One yield per (landkreis, year, crop) — values broadcast over days,
        # so take .first() to avoid weighting by sequence length.
        y_per_key = (df_train.groupby(["landkreis", "crop", "year"])
                              [self.spec.target].first().reset_index())
        self.target_global_mean_ = float(y_per_key[self.spec.target].mean())
        self.target_global_std_ = float(y_per_key[self.spec.target].std() or 1.0)
        # Per-crop mean, used as the baseline for a (landkreis, crop) that has
        # no training history. The global mean is pooled ACROSS crops, so it
        # sits between wheat (~73 dt/ha) and sugar beet (~673) and is not a
        # plausible yield for either — falling back to it produced physically
        # impossible predictions that no model output could recover from.
        self.target_mean_by_crop_ = {
            cr: float(v) for cr, v in
            y_per_key.groupby("crop")[self.spec.target].mean().items()
        }

        # Per-(landkreis, crop) mean for anomaly framing.
        baseline = y_per_key.groupby(["landkreis", "crop"])[self.spec.target].mean()
        self.target_mean_ = {(lk, cr): float(v) for (lk, cr), v in baseline.items()}
        y_per_key = y_per_key.copy()

        # Per-crop linear yield trend, fit on TRAINING YEARS ONLY.
        #
        # The district baseline above is a flat mean over ~20 training years,
        # but several German crops trend hard: sugar beet gains ~10.4 dt/ha/yr,
        # so by the 2024-25 test window the flat mean is stale by +142 dt/ha —
        # +1.47 anomaly-sigma. No feature encodes the calendar year (correctly,
        # to avoid leakage), so a model cannot express that drift itself and
        # the bias lands entirely in the residual: it alone accounts for -1.70
        # of sugar beet's -1.72 R², i.e. essentially all of the "catastrophic
        # failure" that every architecture showed on that crop.
        #
        # Extrapolating a train-only slope forward is leakage-free: nothing
        # about the test years informs the fit.
        self.trend_ref_year_ = float(y_per_key["year"].mean())
        self.target_trend_by_crop_ = {}
        for cr, g in y_per_key.groupby("crop"):
            resid = np.array([
                row[self.spec.target]
                - self.target_mean_.get(
                    (row["landkreis"], cr), self.target_mean_by_crop_.get(cr, 0.0))
                for _, row in g.iterrows()
            ], dtype=float)
            yrs = g["year"].to_numpy(dtype=float) - self.trend_ref_year_
            # Only carry a trend forward when the training years actually show
            # one. Fitting a slope unconditionally overfits: measured against
            # the null model, an ungated trend rescues sugar beet (+1.51 R²,
            # a real +9.0 dt/ha/yr signal) but *costs* silage maize -0.10 and
            # winter wheat -0.10, whose slopes are noise. Gate on the OLS
            # t-statistic so only significant trends are extrapolated.
            slope = 0.0
            if len(np.unique(yrs)) >= 5 and np.isfinite(resid).all():
                try:
                    from scipy import stats as _st
                    lr = _st.linregress(yrs, resid)
                    # p < 0.01 and a slope large enough to matter over the
                    # extrapolation horizon relative to the residual spread.
                    horizon = float(np.abs(yrs).max()) or 1.0
                    meaningful = abs(lr.slope) * horizon > 0.25 * float(resid.std())
                    if lr.pvalue < 0.01 and meaningful:
                        slope = float(lr.slope)
                except Exception:
                    slope = 0.0
            self.target_trend_by_crop_[cr] = slope

        y_per_key["anomaly"] = [
            row[self.spec.target] - self.baseline_yield(
                row["landkreis"], row["crop"], row["year"])
            for _, row in y_per_key.iterrows()
        ]
        # Global anomaly std (fallback) + per-crop anomaly std. Per-crop is
        # essential for multi-crop training: sugar-beet anomalies (~±100 dt/ha)
        # would otherwise dwarf rapeseed (~±5), so a single global std makes the
        # z-scored target dominated by high-yield crops and the model ignores
        # the low-yield ones.
        sd = float(y_per_key["anomaly"].std())
        self.target_anomaly_std_ = sd if sd > 1e-6 else 1.0
        self.target_anomaly_std_by_crop_ = {
            cr: (float(s) if s and s > 1e-6 else self.target_anomaly_std_)
            for cr, s in y_per_key.groupby("crop")["anomaly"].std().items()
        }
        # Clip bounds: [0, 1.5 × train max] per crop. Generous enough to allow
        # genuine record yields, tight enough to catch absurd extrapolations.
        grp = y_per_key.groupby("crop")[self.spec.target]
        self.target_clip_by_crop_ = {
            cr: [0.0, float(mx) * 1.5] for cr, mx in grp.max().items()
        }

        self.fitted_ = True
        return self

    def _anomaly_std(self, crop: str) -> float:
        return self.target_anomaly_std_by_crop_.get(crop, self.target_anomaly_std_)

    def _clip(self, crop: str, value):
        bounds = self.target_clip_by_crop_.get(crop)
        if not bounds:
            return value
        return np.clip(value, bounds[0], bounds[1])

    def transform_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for c, mu in self.feature_mean_.items():
            if c not in df.columns:
                continue
            df[c] = (pd.to_numeric(df[c], errors="coerce").fillna(mu) - mu) / self.feature_std_[c]
        return df

    def require_current_feature_policy(self) -> None:
        """Refuse evaluation of checkpoints trained under an unsafe policy."""
        if self.feature_policy_version != FEATURE_POLICY_VERSION:
            raise RuntimeError(
                "checkpoint uses feature policy "
                f"v{self.feature_policy_version}; current policy is "
                f"v{FEATURE_POLICY_VERSION}. Retrain the model before reporting "
                "metrics."
            )

    def baseline_yield(self, landkreis: str, crop: str,
                       year: float | None = None) -> float:
        """District baseline, degrading to the crop mean then the global mean.

        When ``year`` is given and a train-only trend was fitted, the baseline
        is carried forward along that trend. Pipelines written before trends
        existed have an empty ``target_trend_by_crop_`` and are unaffected.
        """
        v = self.target_mean_.get((landkreis, crop))
        if v is None:
            v = self.target_mean_by_crop_.get(crop, self.target_global_mean_)
        if year is not None and self.target_trend_by_crop_:
            slope = self.target_trend_by_crop_.get(crop, 0.0)
            v = v + slope * (float(year) - self.trend_ref_year_)
        return v

    def transform_target(self, landkreis: str, crop: str, y: float,
                         year: float | None = None) -> float:
        if self.anomaly_framing:
            return ((y - self.baseline_yield(landkreis, crop, year))
                    / self._anomaly_std(crop))
        return (y - self.target_global_mean_) / self.target_global_std_

    # An anomaly more than this many std from the district mean is treated as
    # a model extrapolation error and clamped. 4σ is very generous — genuine
    # year-to-year yield anomalies almost never exceed 3σ.
    Z_CLIP = 4.0

    def inverse_transform_target(
        self, landkreis: str, crop: str, z: float
    ) -> float:
        if self.anomaly_framing:
            z = float(np.clip(z, -self.Z_CLIP, self.Z_CLIP))
            val = z * self._anomaly_std(crop) + self.baseline_yield(landkreis, crop)
        else:
            val = z * self.target_global_std_ + self.target_global_mean_
        return float(self._clip(crop, val))

    def inverse_transform_observed_target(
        self, landkreis: str, crop: str, z: float
    ) -> float:
        """Recover an observed target exactly, without prediction clipping."""
        if self.anomaly_framing:
            return float(
                z * self._anomaly_std(crop)
                + self.baseline_yield(landkreis, crop)
            )
        return float(
            z * self.target_global_std_ + self.target_global_mean_
        )

    def inverse_transform_observed_array(
        self, landkreis: np.ndarray, crop: np.ndarray, z: np.ndarray,
        year: np.ndarray | None = None,
    ) -> np.ndarray:
        _years = year if year is not None else [None] * len(np.asarray(z))
        """Vectorized exact inverse for ground-truth targets."""
        z = np.asarray(z, dtype=float)
        if self.anomaly_framing:
            base = np.array([
                self.baseline_yield(lk, cr, yr)
                for lk, cr, yr in zip(landkreis, crop, _years)
            ])
            std = np.array([self._anomaly_std(cr) for cr in crop])
            return z * std + base
        return z * self.target_global_std_ + self.target_global_mean_

    def inverse_transform_array(
        self, landkreis: np.ndarray, crop: np.ndarray, z: np.ndarray,
        year: np.ndarray | None = None,
    ) -> np.ndarray:
        _years = year if year is not None else [None] * len(np.asarray(z))
        z = np.asarray(z, dtype=float)
        if self.anomaly_framing:
            z = np.clip(z, -self.Z_CLIP, self.Z_CLIP)
            base = np.array([
                self.baseline_yield(lk, cr, yr)
                for lk, cr, yr in zip(landkreis, crop, _years)
            ])
            std = np.array([self._anomaly_std(cr) for cr in crop])
            val = z * std + base
        else:
            val = z * self.target_global_std_ + self.target_global_mean_
        # Final physical-bounds clip per crop.
        lo = np.array([self.target_clip_by_crop_.get(cr, [0.0, np.inf])[0] for cr in crop])
        hi = np.array([self.target_clip_by_crop_.get(cr, [0.0, np.inf])[1] for cr in crop])
        return np.clip(val, lo, hi)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # JSON-friendly keys (tuple → "landkreis|crop")
        payload = {
            "spec": asdict(self.spec),
            "feature_policy_version": self.feature_policy_version,
            "anomaly_framing": self.anomaly_framing,
            "feature_mean": self.feature_mean_,
            "feature_std": self.feature_std_,
            "target_mean": {f"{k[0]}|{k[1]}": v for k, v in self.target_mean_.items()},
            "target_mean_by_crop": self.target_mean_by_crop_,
            "target_trend_by_crop": self.target_trend_by_crop_,
            "trend_ref_year": self.trend_ref_year_,
            "target_global_mean": self.target_global_mean_,
            "target_global_std": self.target_global_std_,
            "target_anomaly_std": self.target_anomaly_std_,
            "target_anomaly_std_by_crop": self.target_anomaly_std_by_crop_,
            "target_clip_by_crop": self.target_clip_by_crop_,
            "sequence_length": self.sequence_length_,
            "tft_max_encoder_length": self.tft_max_encoder_length_,
        }
        path.write_text(json.dumps(payload))

    @classmethod
    def load(cls, path: str | Path) -> "FeaturePipeline":
        d = json.loads(Path(path).read_text())
        p = cls(FeatureSpec(**d["spec"]),
                anomaly_framing=d.get("anomaly_framing", True))
        p.feature_policy_version = d.get("feature_policy_version", 1)
        p.feature_mean_ = d["feature_mean"]
        p.feature_std_ = d["feature_std"]
        p.target_mean_ = {tuple(k.split("|", 1)): v for k, v in d["target_mean"].items()}
        p.target_mean_by_crop_ = d.get("target_mean_by_crop", {})
        p.target_trend_by_crop_ = d.get("target_trend_by_crop", {})
        p.trend_ref_year_ = float(d.get("trend_ref_year", 0.0))
        p.target_global_mean_ = d["target_global_mean"]
        p.target_global_std_ = d.get("target_global_std", 1.0)
        p.target_anomaly_std_ = d["target_anomaly_std"]
        p.target_anomaly_std_by_crop_ = d.get("target_anomaly_std_by_crop", {})
        p.target_clip_by_crop_ = d.get("target_clip_by_crop", {})
        p.sequence_length_ = d.get("sequence_length")
        p.tft_max_encoder_length_ = d.get("tft_max_encoder_length")
        p.fitted_ = True
        return p


# --------------------------------------------------------------------------- #
# Phenology windowing helper
# --------------------------------------------------------------------------- #
def apply_phenology_window(
    df: pd.DataFrame, phenology: dict[str, dict[str, int]]
) -> pd.DataFrame:
    """Slice each (district, year, crop) sample to its growing-season DOYs.

    For winter crops (sowing in prev year), we slice to [1, harvest_doy] of
    the harvest year. We intentionally do not include autumn weather from the
    previous calendar year — that would require cross-year sequence building
    and adds complexity for a relatively small gain.
    """
    keep_idx = []
    for crop, p in phenology.items():
        harvest = p["harvest_doy"]
        sowing = p["sowing_doy"]
        mask = df["crop"] == crop
        if sowing < harvest:
            # Summer crop: window is [sowing, harvest] within one calendar year.
            mask &= (df["doy"] >= sowing) & (df["doy"] <= harvest)
        else:
            # Winter crop: harvest year contains [1, harvest_doy].
            mask &= df["doy"] <= harvest
        keep_idx.append(df.index[mask])
    if not keep_idx:
        return df
    keep = np.concatenate([idx.to_numpy() for idx in keep_idx])
    return df.loc[keep].sort_values(["landkreis", "crop", "year", "doy"])


def filter_incomplete_sequences(
    df: pd.DataFrame,
    phenology: dict[str, dict[str, int]],
    min_coverage: float = 0.90,
) -> pd.DataFrame:
    """Drop crop-year sequences below the expected-window coverage threshold.

    Call this after :func:`apply_phenology_window` and before splitting or
    applying an in-season cutoff. Coverage counts unique DOYs, preventing
    duplicated dates from making an incomplete sequence appear complete.
    """
    if not 0 < min_coverage <= 1:
        raise ValueError("min_coverage must be in the interval (0, 1]")
    if df.empty:
        return df

    expected_by_crop: dict[str, int] = {}
    for crop, bounds in phenology.items():
        sowing = int(bounds["sowing_doy"])
        harvest = int(bounds["harvest_doy"])
        expected_by_crop[crop] = (
            harvest if sowing > harvest else harvest - sowing + 1
        )

    expected = df["crop"].map(expected_by_crop)
    if expected.isna().any():
        unknown = sorted(df.loc[expected.isna(), "crop"].astype(str).unique())
        raise KeyError(f"missing phenology configuration for crops: {unknown}")

    observed = (
        df.groupby(["landkreis", "crop", "year"], sort=False)["doy"]
        .transform("nunique")
    )
    keep = observed >= np.ceil(expected * min_coverage)
    return df.loc[keep].copy()


def crop_indicator_columns(data_config: dict) -> list[str]:
    """Names of the one-hot crop indicators, in the order data.yaml lists."""
    return [f"crop_is_{c}" for c in data_config["crops"]]


def add_crop_indicators(df: pd.DataFrame, data_config: dict) -> pd.DataFrame:
    """Attach one-hot crop indicators.

    These models are trained pooled across all ten crops, whose yields span
    ~36 dt/ha (winter rapeseed) to ~660 (sugar beet), yet nothing in the
    feature set identified the crop — `features.static` carried soil, terrain
    and landkreis_id only. The models had been inferring crop identity from an
    artefact: collate_pad padded every sequence to the longest phenology
    window, so the length of the zero tail was a de-facto crop code. Excluding
    that padding (as correctness requires) removed the signal with it, which
    is why XGBoost lost 0.047 mean R² when the padding fix landed.

    One-hot rather than an integer code: an ordinal `crop_id` would be
    z-scored and fed as a continuous covariate, implying an ordering between
    crops that does not exist — the same defect `landkreis_id` already has.
    Ten binary columns cost little and are unambiguous for both trees and
    networks.

    Columns come from the configured crop list, so the encoding is identical
    at train and eval time without depending on which crops a given frame
    happens to contain.
    """
    out = df.copy()
    crop_col = out["crop"].astype(str)
    for crop in data_config["crops"]:
        out[f"crop_is_{crop}"] = (crop_col == crop).astype("int8")
    return out


def prepare_modeling_frame(df: pd.DataFrame, data_config: dict) -> pd.DataFrame:
    """Apply the shared crop window and data-quality rules."""
    phenology = data_config["phenology"]
    out = apply_phenology_window(df, phenology)
    min_coverage = float(
        data_config.get("quality", {}).get("min_sequence_coverage", 0.90)
    )
    out = filter_incomplete_sequences(out, phenology, min_coverage)
    return add_crop_indicators(out, data_config)


# --------------------------------------------------------------------------- #
# PyTorch dataset
# --------------------------------------------------------------------------- #
class CropYieldDataset(Dataset):
    """Wraps the unified df + pipeline into one sample per (landkreis, year, crop).

    Assumes the input ``df`` has already been:
      1. filtered to a single split,
      2. phenology-windowed,
      3. feature-normalized via ``pipeline.transform_features``.

    The pipeline is also used to z-score the per-(landkreis, crop, year) yield
    target into an anomaly.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_spec: FeatureSpec,
        pipeline: FeaturePipeline,
        cutoff_doy: int | None = None,
    ) -> None:
        if not _HAS_TORCH:
            raise RuntimeError("PyTorch is required for CropYieldDataset.")
        self.spec = feature_spec
        self.pipeline = pipeline
        self.cutoff_doy = cutoff_doy

        self._groups: list[pd.DataFrame] = []
        self._targets_z: list[float] = []
        self._keys: list[tuple[str, str, int]] = []

        df = df.sort_values(["landkreis", "crop", "year", "doy"])
        for (lk, cr, yr), g in df.groupby(["landkreis", "crop", "year"], sort=False):
            if cutoff_doy is not None:
                g = g[g["doy"] <= cutoff_doy]
            if g.empty:
                continue
            y_abs = float(g.iloc[0][feature_spec.target])
            self._groups.append(g.reset_index(drop=True))
            self._targets_z.append(
                pipeline.transform_target(lk, cr, y_abs, yr))
            self._keys.append((lk, cr, int(yr)))

    def __len__(self) -> int:
        return len(self._groups)

    def __getitem__(self, idx: int) -> dict[str, "torch.Tensor"]:
        g = self._groups[idx]
        x_past = torch.tensor(g[self.spec.past_observed].to_numpy(np.float32))
        x_future = torch.tensor(g[self.spec.known_future].to_numpy(np.float32))
        if self.spec.static:
            x_static = torch.tensor(g.iloc[0][self.spec.static].to_numpy(np.float32))
        else:
            x_static = torch.zeros(0, dtype=torch.float32)
        y = torch.tensor(self._targets_z[idx], dtype=torch.float32)
        return {"x_past": x_past, "x_static": x_static, "x_future": x_future, "y": y}

    @property
    def keys(self) -> list[tuple[str, str, int]]:
        return self._keys


# --------------------------------------------------------------------------- #
# Collate
# --------------------------------------------------------------------------- #
def collate_pad(
    batch: list[dict],
    pad_to: int | None = None,
) -> dict[str, "torch.Tensor"]:
    observed_max = max(item["x_past"].shape[0] for item in batch)
    if pad_to is not None and observed_max > pad_to:
        raise ValueError(
            f"batch sequence length {observed_max} exceeds pad_to={pad_to}"
        )
    max_T = max(observed_max, pad_to or 0)
    F_past = batch[0]["x_past"].shape[1]
    F_fut = batch[0]["x_future"].shape[1]
    S = batch[0]["x_static"].shape[0]
    B = len(batch)

    x_past = torch.zeros(B, max_T, F_past)
    x_future = torch.zeros(B, max_T, F_fut)
    x_static = torch.zeros(B, S)
    y = torch.zeros(B)
    mask = torch.zeros(B, max_T, dtype=torch.bool)

    for i, item in enumerate(batch):
        T = item["x_past"].shape[0]
        x_past[i, :T] = item["x_past"]
        x_future[i, :T] = item["x_future"]
        x_static[i] = item["x_static"]
        y[i] = item["y"]
        mask[i, :T] = True

    return {
        "x_past": x_past, "x_future": x_future,
        "x_static": x_static, "y": y, "mask": mask,
    }

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
    drop_features: Sequence[str] | None = None,
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

    # drop_features lets a run switch features OFF (e.g. the crop-yield
    # rolling test's "without drivers" arm). It can only remove, never add,
    # so it cannot re-enable anything the excluded list forbids.
    excluded = set(configured.get("excluded", [])) | set(drop_features or [])

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
                 anomaly_framing: bool = True,
                 trend_test: str = "district_year") -> None:
        self.spec = feature_spec
        self.anomaly_framing = anomaly_framing
        # "district_year" (v1-v3): trend significance tested on every
        # district-year; "yearly" (preregistration v4 §2.1): on the yearly mean
        # residual, one point per training year, because a national good or bad
        # year moves all districts together.
        self.trend_test = trend_test
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
        self.window_: str = "calendar"
        self.common_sample_: bool = False
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
        # preregistration v10 §2.3: each district-crop's own mean training year,
        # the year its mean refers to (used when trend_district_ref is on)
        self.target_mean_year_ = {(lk, cr): float(v) for (lk, cr), v in
                                  y_per_key.groupby(["landkreis", "crop"])["year"].mean().items()}
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
                    if getattr(self, "trend_test", "district_year") == "yearly":
                        ym = pd.Series(resid).groupby(yrs).mean()
                        lr = _st.linregress(ym.index.to_numpy(float), ym.to_numpy(float))
                    else:
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

        if getattr(self, "baseline_blend_", False):
            # preregistration v9 §2: least-squares weight of the recent level per
            # crop, on TRAINING crop-years only, clipped to [0, 1]
            self.recent_weight_by_crop_ = {}
            for cr, g in y_per_key.groupby("crop"):
                num = den = 0.0
                slope = self.target_trend_by_crop_.get(cr, 0.0)
                for lk, yr, yv in zip(g["landkreis"], g["year"], g[self.spec.target]):
                    lt, b5 = baseline_components(self.yield_hist_.get((lk, cr), {}), int(yr), slope)
                    if lt == lt and b5 == b5 and yv == yv:
                        num += (yv - lt) * (b5 - lt); den += (b5 - lt) ** 2
                self.recent_weight_by_crop_[cr] = float(np.clip(num / den, 0.0, 1.0)) if den > 0 else 0.0

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
        clim = getattr(self, "district_clim_", None)
        if clim and not all(c in df.columns for c in anomaly_columns(clim)):
            df = add_district_anomalies(df, clim)
        if getattr(self, "recent_baseline_", False) and "ylag1_rel" not in df.columns:
            df = add_yield_lags(df, self.yield_hist_)
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
        if getattr(self, "baseline_blend_", False) and year is not None:
            # preregistration v9 §2: w_c x B5 + (1 - w_c) x LT, w_c fitted on training years
            h = self.yield_hist_.get((landkreis, crop))
            if h:
                lt, b5 = baseline_components(h, int(year), self.target_trend_by_crop_.get(crop, 0.0))
                if lt == lt:
                    w = self.recent_weight_by_crop_.get(crop, 0.0)
                    return float(w * b5 + (1 - w) * lt) if b5 == b5 else float(lt)
        if getattr(self, "recent_baseline_", False) and not getattr(self, "baseline_blend_", False) \
                and year is not None:
            # preregistration v7 §2.1: mean of the district's yields in t-5..t-1
            # (>= 3 years) moved along the fold's trend to year t
            h = self.yield_hist_.get((landkreis, crop))
            if h:
                w = recent_window(h, year)
                if len(w) >= RECENT_MIN_YEARS:
                    slope = self.target_trend_by_crop_.get(crop, 0.0)
                    return float(np.mean([h[t] for t in w]) + slope * (float(year) - np.mean(w)))
        v = self.target_mean_.get((landkreis, crop))
        if v is None:
            v = self.target_mean_by_crop_.get(crop, self.target_global_mean_)
        if year is not None and self.target_trend_by_crop_:
            slope = self.target_trend_by_crop_.get(crop, 0.0)
            ref = (self.target_mean_year_.get((landkreis, crop), self.trend_ref_year_)
                   if getattr(self, "trend_district_ref", False) else self.trend_ref_year_)
            v = v + slope * (float(year) - ref)
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
            "window": self.window_,
            "common_sample": self.common_sample_,
            "tft_max_encoder_length": self.tft_max_encoder_length_,
            "trend_test": getattr(self, "trend_test", "district_year"),
            "trend_district_ref": getattr(self, "trend_district_ref", False),
            "target_mean_year": {f"{k[0]}|{k[1]}": v for k, v in getattr(self, "target_mean_year_", {}).items()},
            "recent_baseline": getattr(self, "recent_baseline_", False),
            "baseline_blend": getattr(self, "baseline_blend_", False),
            "recent_weight_by_crop": getattr(self, "recent_weight_by_crop_", {}),
            "yield_hist": ({f"{k[0]}|{k[1]}": {str(t): v for t, v in h.items()}
                            for k, h in self.yield_hist_.items()}
                           if getattr(self, "recent_baseline_", False) else None),
            "phenology": getattr(self, "phenology_", None),
            "national": getattr(self, "national_", None),
            "phenology_version": getattr(self, "phenology_version_", None),
            "stage_features": getattr(self, "stage_features_", False),
            "district_clim": getattr(self, "district_clim_", None),
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
        p.window_ = d.get("window", "calendar")
        p.common_sample_ = bool(d.get("common_sample", False))
        p.tft_max_encoder_length_ = d.get("tft_max_encoder_length")
        p.phenology_ = d.get("phenology")
        p.trend_test = d.get("trend_test", "district_year")
        p.trend_district_ref = bool(d.get("trend_district_ref", False))
        p.target_mean_year_ = {tuple(k.split("|", 1)): v for k, v in d.get("target_mean_year", {}).items()}
        p.recent_baseline_ = bool(d.get("recent_baseline", False))
        p.baseline_blend_ = bool(d.get("baseline_blend", False))
        p.recent_weight_by_crop_ = d.get("recent_weight_by_crop", {})
        p.yield_hist_ = ({tuple(k.split("|", 1)): {int(t): v for t, v in h.items()}
                          for k, h in d["yield_hist"].items()} if d.get("yield_hist") else {})
        p.national_ = d.get("national")
        p.phenology_version_ = d.get("phenology_version")
        p.stage_features_ = bool(d.get("stage_features", False))
        p.district_clim_ = d.get("district_clim")
        p.fitted_ = True
        return p


# --------------------------------------------------------------------------- #
# Phenology windowing helper
# --------------------------------------------------------------------------- #
SAMPLE_KEY = ["landkreis", "crop", "year"]


def _is_winter(bounds: dict[str, int]) -> bool:
    return int(bounds["sowing_doy"]) > int(bounds["harvest_doy"])


def _autumn_rows(df: pd.DataFrame, phenology: dict[str, dict[str, int]],
                 target: str = "yield_dt_per_ha") -> pd.DataFrame:
    """Previous-autumn rows (sowing_doy .. 31 Dec of Y-1) for winter crop-year Y.

    The rows carry the crop-year's identity: ``year`` = Y and the target column
    = the yield of harvest year Y (never the previous year's yield, which is
    what the source rows hold). Weather, satellite and driver values come from
    the same crop's own rows of year Y-1 when they exist; otherwise from another
    crop's rows for that district and date (daily inputs are district-level;
    humidity differs slightly between crops, so same-crop is preferred).
    """
    parts = []
    by_date = (df.sort_values(["landkreis", "date", "crop"])
                 .drop_duplicates(["landkreis", "date"]))
    for crop, b in phenology.items():
        if not _is_winter(b):
            continue
        sowing, harvest = int(b["sowing_doy"]), int(b["harvest_doy"])
        cur = df[(df["crop"] == crop) & (df["doy"] <= harvest)]
        if cur.empty:
            continue
        y_of = cur.groupby(["landkreis", "year"])[target].first()
        wanted = pd.DataFrame(y_of.index.tolist(), columns=["landkreis", "year"])

        same = df[(df["crop"] == crop) & (df["doy"] >= sowing)].copy()
        same["year"] = same["year"] + 1
        have = set(zip(same["landkreis"], same["year"]))
        missing = wanted[[(lk, y) not in have for lk, y in zip(wanted.landkreis, wanted.year)]]

        other = by_date[by_date["doy"] >= sowing].copy()
        other["year"] = other["year"] + 1
        other = other.merge(missing, on=["landkreis", "year"], how="inner")

        a = pd.concat([same, other], ignore_index=True)
        a["crop"] = crop
        # inner join keeps only crop-years that exist in year Y, and replaces
        # the source rows' (previous-year) yield with the crop-year's yield
        a = (a.drop(columns=[target])
               .merge(y_of.rename(target).reset_index(), on=["landkreis", "year"], how="inner"))
        parts.append(a)
    if not parts:
        return df.iloc[0:0]
    return pd.concat(parts, ignore_index=True)


def autumn_complete_keys(df: pd.DataFrame, phenology: dict[str, dict[str, int]],
                         min_coverage: float = 0.90) -> set[tuple]:
    """Winter crop-years whose previous autumn is >= min_coverage complete."""
    a = _autumn_rows(df, phenology)
    if a.empty:
        return set()
    need = {c: 365 - int(b["sowing_doy"]) + 1 for c, b in phenology.items() if _is_winter(b)}
    got = a.groupby(SAMPLE_KEY)["doy"].nunique()
    exp = np.array([need[k[1]] for k in got.index])
    return set(got.index[got.to_numpy() >= np.ceil(exp * min_coverage)])


def apply_phenology_window(
    df: pd.DataFrame, phenology: dict[str, dict[str, int]],
    window: str = "calendar",
) -> pd.DataFrame:
    """Slice each (district, year, crop) sample to its growing-season window.

    window="calendar" (the historic default): winter crops (sowing in the
    previous year) are cut to [1 Jan, harvest_doy] of the harvest year.
    window="crop_year": winter crops also get the previous autumn
    [sowing_doy of Y-1, 31 Dec of Y-1] (see ``_autumn_rows``), so every crop's
    sequence runs sowing -> harvest. Spring crops are identical in both modes.
    Rows are ordered by date, which for crop_year keeps autumn before spring.
    """
    if window not in ("calendar", "crop_year"):
        raise ValueError(f"window must be calendar|crop_year, got {window!r}")
    base = _apply_calendar_window(df, phenology)
    if window == "crop_year":
        base = pd.concat([base, _autumn_rows(df, phenology)], ignore_index=True)
    return base.sort_values(["landkreis", "crop", "year", "date"])


def _apply_calendar_window(
    df: pd.DataFrame, phenology: dict[str, dict[str, int]]
) -> pd.DataFrame:
    """The historic window: winter crops [1 Jan, harvest], spring [sowing, harvest]."""
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
    return df.loc[keep].sort_values(["landkreis", "crop", "year", "date"])


def filter_incomplete_sequences(
    df: pd.DataFrame,
    phenology: dict[str, dict[str, int]],
    min_coverage: float = 0.90,
    window: str = "calendar",
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
        if sowing > harvest:
            # winter crop: [1 Jan, harvest], plus the previous autumn
            # [sowing, 31 Dec] under the crop-year window
            expected_by_crop[crop] = harvest + (
                365 - sowing + 1 if window == "crop_year" else 0)
        else:
            expected_by_crop[crop] = harvest - sowing + 1

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


def prepare_modeling_frame(df: pd.DataFrame, data_config: dict,
                           window: str = "calendar",
                           common_sample: bool = False) -> pd.DataFrame:
    """Apply the shared crop window and data-quality rules.

    window: "calendar" (historic) or "crop_year" (winter crops from sowing in
    the previous autumn). common_sample=True drops winter crop-years whose
    previous autumn is incomplete, in EITHER window mode, so that calendar and
    crop-year experiments score exactly the same crop-years.
    """
    phenology = data_config["phenology"]
    min_coverage = float(
        data_config.get("quality", {}).get("min_sequence_coverage", 0.90)
    )
    out = apply_phenology_window(df, phenology, window=window)
    out = filter_incomplete_sequences(out, phenology, min_coverage, window=window)
    if common_sample:
        # A winter crop-year is kept only if BOTH parts are >= min_coverage
        # complete: the 1 Jan -> harvest part and the previous autumn. Checking
        # each part separately (not the crop-year total) is what makes the
        # calendar and crop-year sample sets identical: a full autumn must not
        # compensate for a gap in winter/spring.
        ok = (autumn_complete_keys(df, phenology, min_coverage)
              & _current_part_complete_keys(df, phenology, min_coverage))
        winter = [c for c, b in phenology.items() if _is_winter(b)]
        keys = list(zip(out["landkreis"], out["crop"], out["year"]))
        keep = [(cr not in winter) or (k in ok)
                for k, cr in zip(keys, out["crop"])]
        out = out.loc[keep]
    return add_crop_indicators(out, data_config)


# --------------------------------------------------------------------------- #
# Observed phenology (preregistration_2026-09-28_v2.md §2.2)
# --------------------------------------------------------------------------- #
PHENOLOGY_SLOTS = ["sow", "emerge", "stem", "head", "ripe"]
PHENOLOGY_OBS = Path(__file__).resolve().parents[2] / "data/processed/static/phenology_obs.parquet"
# preregistration v3 §2.1: harvest slot, separate table (v2 file unchanged)
PHENOLOGY_SLOTS_V3 = PHENOLOGY_SLOTS + ["harvest"]
PHENOLOGY_OBS_V3 = PHENOLOGY_OBS.with_name("phenology_obs_v3.parquet")


def phenology_columns(slots: Sequence[str] = PHENOLOGY_SLOTS) -> list[str]:
    return ([f"ph_{s}" for s in slots]
            + [f"ph_{s}_missing" for s in slots])


def add_phenology_features(df: pd.DataFrame, network: str,
                           obs_path: str | Path | None = None,
                           slots: Sequence[str] = PHENOLOGY_SLOTS) -> pd.DataFrame:
    """Daily step inputs from observed crop development dates.

    ph_<slot> = 1 from the district's observed date of that stage on, else 0;
    ph_<slot>_missing = 1 where the district has no date. Applied AFTER the
    crop window (autumn rows are copied from the previous year and must get
    the harvest year's own dates). Keyed on (landkreis, crop, harvest year),
    compared on the absolute date, so it is causal at every horizon.
    """
    obs = pd.read_parquet(obs_path or PHENOLOGY_OBS)
    obs = obs[obs["network"] == network]
    wide = obs.pivot_table(index=["landkreis", "crop", "year"], columns="slot",
                           values="date", aggfunc="first")
    wide.columns = [f"_ph_{c}" for c in wide.columns]
    out = df.merge(wide, left_on=["landkreis", "crop", "year"], right_index=True, how="left")
    date = pd.to_datetime(out["date"])
    for s_ in slots:
        d = out.get(f"_ph_{s_}", pd.Series(pd.NaT, index=out.index))
        d = pd.to_datetime(d)
        out[f"ph_{s_}"] = (date >= d).astype("int8")
        out[f"ph_{s_}_missing"] = d.isna().astype("int8")
    return out.drop(columns=[c for c in out.columns if c.startswith("_ph_")])


# --------------------------------------------------------------------------- #
# Crop-stage weather (preregistration v3 §2.2) -- running totals, causal
# --------------------------------------------------------------------------- #
STAGE_COLUMNS = (["cum_gdd", "cum_prec", "cum_cwb", "cum_heat"]
                 + ["gf_days", "gf_gdd", "gf_prec", "gf_cwb", "gf_heat", "days_since_stage"])
_TMAX, _TMIN, _PREC, _CWB = ("temperature_air_max_2m", "temperature_air_min_2m",
                             "precipitation_height", "cwb")


def add_stage_features(df: pd.DataFrame, network: str,
                       obs_path: str | Path | None = None) -> pd.DataFrame:
    """Weather accumulated over the crop's observed stages, as daily running totals.

    cum_*: since the observed sowing date (window start if none).
    gf_*:  from the observed heading/flowering date to the observed ripeness or
           harvest date (first of the two), frozen afterwards; 0 before heading.
    days_since_stage: days since the latest observed stage (-1 if none yet).
    A value on day d depends only on stages observed on or before d and on
    weather up to d, so the frame stays causal at every horizon.
    """
    obs = pd.read_parquet(obs_path or PHENOLOGY_OBS_V3)
    wide = (obs[obs["network"] == network]
            .pivot_table(index=["landkreis", "crop", "year"], columns="slot",
                         values="date", aggfunc="first"))
    key = ["landkreis", "crop", "year"]
    out = df.reset_index(drop=True)
    j = out[key].merge(wide, left_on=key, right_index=True, how="left")
    date = pd.to_datetime(out["date"]).to_numpy()
    nat = np.datetime64("NaT")

    def col(name):
        return pd.to_datetime(j[name]).to_numpy() if name in j else np.full(len(j), nat)

    sow, head = col("sow"), col("head")
    ripe, harv = col("ripe"), col("harvest")
    end = np.where(np.isnat(ripe), harv, np.where(np.isnat(harv), ripe, np.minimum(ripe, harv)))

    tmax = pd.to_numeric(out[_TMAX], errors="coerce").to_numpy(float)
    tmin = pd.to_numeric(out[_TMIN], errors="coerce").to_numpy(float)
    x = {"gdd": np.nan_to_num(np.clip((tmax + tmin) / 2, 0, None)),
         "prec": np.nan_to_num(pd.to_numeric(out[_PREC], errors="coerce").to_numpy(float)),
         "cwb": np.nan_to_num(pd.to_numeric(out[_CWB], errors="coerce").to_numpy(float))
                if _CWB in out else np.zeros(len(out)),
         "heat": (np.nan_to_num(tmax) >= 30).astype(float)}

    season = np.isnat(sow) | (date >= sow)
    gf = ~np.isnat(head) & (date >= head) & (np.isnat(end) | (date <= end))
    order = np.lexsort((date, out["year"].to_numpy(), out["crop"].to_numpy(),
                        out["landkreis"].to_numpy()))
    grp = out[key].iloc[order]
    gid = grp.ne(grp.shift()).any(axis=1).cumsum().to_numpy()

    def running(v):
        s = pd.Series(v[order]).groupby(gid).cumsum().to_numpy()
        r = np.empty_like(s); r[order] = s
        return r

    for k, v in x.items():
        out[f"cum_{k}"] = running(v * season)
        out[f"gf_{k}"] = running(v * gf)
    out["gf_days"] = running(gf.astype(float))

    last = np.full(len(out), nat, dtype="datetime64[ns]")
    for s_ in PHENOLOGY_SLOTS_V3:
        d = col(s_).astype("datetime64[ns]")
        seen = ~np.isnat(d) & (date >= d)
        last = np.where(seen & (np.isnat(last) | (d > last)), d, last)
    since = (date.astype("datetime64[ns]") - last) / np.timedelta64(1, "D")
    out["days_since_stage"] = np.where(np.isnat(last), -1.0, since)
    return out


STRESS_COLUMNS = ["se_prec", "se_cwb", "se_heat", "se_swr", "fl_hdd", "fl_tmax", "fl_vpd", "fl_cwb",
                  "gf_hdd", "gf_vpd", "gf_swr", "frost_stem"]
FLOWER_DAYS = 14


def add_stress_features(df: pd.DataFrame, network: str,
                        obs_path: str | Path | None = None) -> pd.DataFrame:
    """Crop stress at the critical stages (preregistration v20 §2), as daily running values.

    se_*: stem elongation, observed stem date -> observed heading (sums; se_swr = running mean of swroot7).
    fl_*: flowering, the first FLOWER_DAYS days after the observed heading date (heat degree days above 30 C,
          running max tmax, mean vpd, summed cwb).
    gf_hdd/gf_vpd/gf_swr: grain filling, heading -> ripeness/harvest (as the existing gf_* inputs).
    frost_stem: days with tmin <= -2 C from the observed stem date on.
    Every value uses only stages observed on or before the day and weather up to the day (causal); 0 before
    its window opens.
    """
    obs = pd.read_parquet(obs_path or PHENOLOGY_OBS_V3)
    wide = (obs[obs["network"] == network]
            .pivot_table(index=["landkreis", "crop", "year"], columns="slot", values="date", aggfunc="first"))
    key = ["landkreis", "crop", "year"]
    out = df.reset_index(drop=True)
    j = out[key].merge(wide, left_on=key, right_index=True, how="left")
    date = pd.to_datetime(out["date"]).to_numpy().astype("datetime64[ns]")
    nat = np.datetime64("NaT")

    def col(name):
        return (pd.to_datetime(j[name]).to_numpy().astype("datetime64[ns]") if name in j
                else np.full(len(j), nat, dtype="datetime64[ns]"))

    stem, head, ripe, harv = col("stem"), col("head"), col("ripe"), col("harvest")
    end = np.where(np.isnat(ripe), harv, np.where(np.isnat(harv), ripe, np.minimum(ripe, harv)))
    num = lambda c: (np.nan_to_num(pd.to_numeric(out[c], errors="coerce").to_numpy(float))
                     if c in out else np.zeros(len(out)))
    tmax, tmin = num(_TMAX), num(_TMIN)
    prec, cwb, vpd, swr = num(_PREC), num(_CWB), num("vpd"), num("swroot7")
    hdd = np.clip(tmax - 30.0, 0, None)

    se = ~np.isnat(stem) & (date >= stem) & (np.isnat(head) | (date < head))
    fl = ~np.isnat(head) & (date >= head) & (date < head + np.timedelta64(FLOWER_DAYS, "D"))
    gf = ~np.isnat(head) & (date >= head) & (np.isnat(end) | (date <= end))
    after_stem = ~np.isnat(stem) & (date >= stem)

    order = np.lexsort((date, out["year"].to_numpy(), out["crop"].to_numpy(), out["landkreis"].to_numpy()))
    grp = out[key].iloc[order]
    gid = grp.ne(grp.shift()).any(axis=1).cumsum().to_numpy()

    def run(v, how="sum"):
        s = pd.Series(v[order])
        r_ = s.groupby(gid).cummax() if how == "max" else s.groupby(gid).cumsum()
        r = np.empty(len(v)); r[order] = r_.to_numpy()
        return r

    def mean(v, m):
        n = run(m.astype(float))
        return np.where(n > 0, run(v * m) / np.maximum(n, 1), 0.0)

    out["se_prec"], out["se_cwb"] = run(prec * se), run(cwb * se)
    out["se_heat"], out["se_swr"] = run((tmax >= 30) * se), mean(swr, se)
    out["fl_hdd"], out["fl_tmax"] = run(hdd * fl), run(np.where(fl, tmax, -np.inf), "max")
    out["fl_tmax"] = np.where(np.isfinite(out["fl_tmax"]), out["fl_tmax"], 0.0)
    out["fl_vpd"], out["fl_cwb"] = mean(vpd, fl), run(cwb * fl)
    out["gf_hdd"], out["gf_vpd"], out["gf_swr"] = run(hdd * gf), mean(vpd, gf), mean(swr, gf)
    out["frost_stem"] = run((tmin <= -2.0) * after_stem)
    return out


# --------------------------------------------------------------------------- #
# Flowering index (preregistration v8 §2.1) -- MODIS NDYI over cropland, Mar-Jun
# --------------------------------------------------------------------------- #
NDYI_PATH = Path(__file__).resolve().parents[2] / "data/processed/static/ndyi_crop.parquet"
FLOWER_COLUMNS = ["ndyi_crop", "ndyi_observed"]


def add_flowering_index(df: pd.DataFrame, path: str | Path | None = None) -> pd.DataFrame:
    """Merge daily district NDYI by (landkreis, date); missing -> NaN, observed flag 0."""
    nd = pd.read_parquet(path or NDYI_PATH)[["landkreis", "date", "ndyi_crop"]]
    nd["date"] = pd.to_datetime(nd["date"])
    out = df.drop(columns=[c for c in FLOWER_COLUMNS if c in df.columns])
    key = pd.to_datetime(out["date"])
    out = out.assign(_d=key.dt.tz_localize(None) if key.dt.tz is not None else key)
    out = out.merge(nd.rename(columns={"date": "_d"}), on=["landkreis", "_d"], how="left").drop(columns="_d")
    out["ndyi_observed"] = out["ndyi_crop"].notna().astype("int8")
    return out


def reliable_series(df: pd.DataFrame, min_abs: int = 5, frac: float = 0.6) -> pd.DataFrame:
    """Keep rows of district-crop series reported in >= max(min_abs, frac x n) of the
    n distinct years present in `df` (preregistration v12 §2; apply to TRAINING rows only)."""
    n = df["year"].nunique()
    need = max(min_abs, int(np.ceil(frac * n)))
    cnt = df.groupby(["landkreis", "crop"])["year"].nunique()
    keep = cnt[cnt >= need].index
    return df[pd.MultiIndex.from_frame(df[["landkreis", "crop"]]).isin(keep)]


# --------------------------------------------------------------------------- #
# Crop-specific satellite (preregistration v13 §2) -- MODIS over each crop's own fields
# --------------------------------------------------------------------------- #
CS_MASK = {"winter_wheat": "wheat", "winter_barley": "barley", "summer_barley": "barley", "rye": "rye",
           "oats": "oats", "silage_maize": "maize", "triticale": "triticale", "potatoes": "potato",
           "sugar_beet": "sugarbeet", "winter_rapeseed": "rapeseed"}
CS_MAX_AGE = {"ndvi": 32, "evi": 32, "lai": 16, "fpar": 16}       # carry forward, as arm D
CS_MIN_SHARE = 0.005


def add_crop_specific_sat(df: pd.DataFrame, sat_path: str | Path | None = None,
                          share_path: str | Path | None = None) -> pd.DataFrame:
    """Replace ndvi_crop/evi_crop/lai_crop/fpar_crop by the row crop's own-field averages.

    Fallback to the cropland values (cs_fallback = 1) where the crop's mapped share of
    the district is below 0.5% or no crop-specific value is available.
    """
    static = Path(__file__).resolve().parents[2] / "data/processed/static"
    sat = pd.read_parquet(sat_path or static / "cs_sat.parquet")
    share = pd.read_csv(share_path or static / "cs_share.csv").set_index("landkreis")
    out = df.copy()
    out["cs_fallback"] = np.int8(1)
    dates = pd.to_datetime(out["date"])
    dates = (dates.dt.tz_localize(None) if dates.dt.tz is not None else dates).astype("datetime64[ns]")
    sat = sat.assign(date=pd.to_datetime(sat["date"]).astype("datetime64[ns]"))   # same unit for merge_asof
    for crop, mask in CS_MASK.items():
        rows = out.index[out["crop"] == crop]
        if not len(rows):
            continue
        f = pd.DataFrame({"landkreis": out.loc[rows, "landkreis"].astype(str), "date": dates.loc[rows],
                          "_i": rows}).sort_values("date")
        ok_share = f["landkreis"].map(share[mask]).fillna(0) >= CS_MIN_SHARE
        s = sat[sat["mask"] == mask]
        got = pd.Series(False, index=rows)
        for b, age in CS_MAX_AGE.items():
            o = s[["landkreis", "date", b]].dropna().sort_values("date")
            m = pd.merge_asof(f, o, on="date", by="landkreis", direction="backward",
                              tolerance=pd.Timedelta(days=age)).set_index("_i")[b]
            use = m.notna() & ok_share.reindex(m.index).values
            out.loc[use[use].index, f"{b}_crop"] = m[use]
            got = got | use.reindex(rows, fill_value=False)
        out.loc[got[got].index, "cs_fallback"] = np.int8(0)
    return out


# --------------------------------------------------------------------------- #
# Area-weighted training (preregistration v15 §4)
# --------------------------------------------------------------------------- #
AREA_W_CLIP = (0.25, 4.0)
AREA_FILL_QUANTILE = 0.05


def area_weights(rows: pd.DataFrame, census_path: str | Path | None = None) -> dict:
    """{(landkreis, crop, year): w} for training rows, w = sqrt(causal crop area), normalised per crop.

    Causal area = latest census year <= year with a reported area for the district-crop; none ->
    the crop's 5th-percentile area in that census. Normalised so each crop's mean weight over
    ``rows`` is 1, then clipped to [0.25, 4].
    """
    static = Path(__file__).resolve().parents[2] / "data/processed/static"
    cen = pd.read_parquet(census_path or static / "crop_area_census.parquet")
    keys = rows[["landkreis", "crop", "year"]].drop_duplicates()
    years = sorted(cen.census_year.unique())
    fill = cen.groupby(["crop", "census_year"]).area_ha.quantile(AREA_FILL_QUANTILE)
    lookup = cen.set_index(["landkreis", "crop", "census_year"]).area_ha.sort_index()
    by_dc = {k: g.droplevel([0, 1]) for k, g in lookup.groupby(level=[0, 1])}
    area = []
    for lk, cr, yr in keys.itertuples(index=False):
        past = [c for c in years if c <= int(yr)] or years[:1]
        s = by_dc.get((lk, cr))
        hit = s[s.index <= int(yr)] if s is not None else None
        if hit is not None and len(hit):
            area.append(float(hit.iloc[-1]))
        else:
            area.append(float(fill.get((cr, past[-1]), np.nan)))
    k = keys.assign(r=np.sqrt(np.asarray(area, dtype=float)))
    k["r"] = k["r"].fillna(k.groupby("crop")["r"].transform("median"))
    k["w"] = (k["r"] / k.groupby("crop")["r"].transform("mean")).clip(*AREA_W_CLIP)
    return {(lk, cr, int(yr)): float(w) for lk, cr, yr, w in k[["landkreis", "crop", "year", "w"]].itertuples(index=False)}


# --------------------------------------------------------------------------- #
# Recency (preregistration v7 §2) -- recent district yields, years < t only
# --------------------------------------------------------------------------- #
YIELD_LAG_COLUMNS = ["ylag1_rel", "ylag2_rel", "ylag3_rel", "yrecent_vs_all"]
RECENT_WINDOW, RECENT_MIN_YEARS = 5, 3


def build_yield_history(yields: pd.DataFrame, target: str = "yield_dt_per_ha") -> dict:
    """{(landkreis, crop): {year: yield}} from a table of reported yields."""
    hist: dict = {}
    for lk, cr, yr, v in yields[["landkreis", "crop", "year", target]].dropna().itertuples(index=False):
        hist.setdefault((lk, cr), {})[int(yr)] = float(v)
    return hist


def recent_window(h: dict, year: int) -> list[int]:
    return [t for t in range(int(year) - RECENT_WINDOW, int(year)) if t in h]


def baseline_components(h: dict, year: int, slope: float = 0.0) -> tuple[float, float]:
    """(LT, B5) for year t from yields of years < t (preregistration v9 §2).

    LT = mean of all years < t, B5 = mean of t-5..t-1; each needs >= 3 years and
    is moved along `slope` to year t. NaN where unavailable.
    """
    past = [t for t in h if t < year]
    w = recent_window(h, year)
    lt = (float(np.mean([h[t] for t in past]) + slope * (year - np.mean(past)))
          if len(past) >= RECENT_MIN_YEARS else np.nan)
    b5 = (float(np.mean([h[t] for t in w]) + slope * (year - np.mean(w)))
          if len(w) >= RECENT_MIN_YEARS else np.nan)
    return lt, b5


def add_yield_lags(df: pd.DataFrame, hist: dict) -> pd.DataFrame:
    """Four constant-over-season input channels from yields of years < t.

    ylag{k}_rel = y(t-k) / m - 1 with m the mean of y(t-5..t-1) (>= 3 years);
    yrecent_vs_all = m / mean(all years < t) - 1. Missing -> 0.
    """
    rows = []
    for lk, cr, yr in df[["landkreis", "crop", "year"]].drop_duplicates().itertuples(index=False):
        h = hist.get((lk, cr), {})
        w = recent_window(h, yr)
        m = float(np.mean([h[t] for t in w])) if len(w) >= RECENT_MIN_YEARS else np.nan
        past = [v for t, v in h.items() if t < yr]
        lags = [(h[yr - k] / m - 1.0) if (yr - k in h and m == m and m > 0) else 0.0 for k in (1, 2, 3)]
        rec = (m / float(np.mean(past)) - 1.0) if (m == m and past) else 0.0
        rows.append((lk, cr, yr, *lags, rec))
    L = pd.DataFrame(rows, columns=["landkreis", "crop", "year"] + YIELD_LAG_COLUMNS)
    return df.drop(columns=[c for c in YIELD_LAG_COLUMNS if c in df.columns]).merge(
        L, on=["landkreis", "crop", "year"], how="left")


# --------------------------------------------------------------------------- #
# District anomalies (preregistration v3 §2.3) -- climatology from train years
# --------------------------------------------------------------------------- #
ANOMALY_BASE = ["temperature_air_max_2m", "temperature_air_min_2m", "precipitation_height",
                "cwb", "vpd", "swroot7", "ssrd", "ndvi_crop", "evi_crop", "lai_crop", "fpar_crop"]


def fit_district_climatology(df_train: pd.DataFrame,
                             cols: Sequence[str] = ANOMALY_BASE) -> dict:
    """Per (district, calendar month) mean of each column over training rows."""
    cols = [c for c in cols if c in df_train.columns]
    u = df_train.drop_duplicates(["landkreis", "date"])[["landkreis", "date"] + cols].copy()
    u["m"] = pd.to_datetime(u["date"]).dt.month
    g = u.groupby(["landkreis", "m"])[cols].mean()
    return {c: {f"{lk}|{m}": float(v) for (lk, m), v in g[c].dropna().items()} for c in cols}


def add_district_anomalies(df: pd.DataFrame, clim: dict) -> pd.DataFrame:
    """<col>_anom = value - district-month training mean (NaN where unknown)."""
    out = df.copy()
    k = out["landkreis"].astype(str) + "|" + pd.to_datetime(out["date"]).dt.month.astype(str)
    for c, table in clim.items():
        out[f"{c}_anom"] = pd.to_numeric(out[c], errors="coerce") - k.map(table)
    return out


def anomaly_columns(clim: dict) -> list[str]:
    return [f"{c}_anom" for c in clim]


# Bump whenever the windowing / sample-selection code changes, so a cached
# frame built by older code can never be reused.
FRAME_CACHE_VERSION = 2


def load_modeling_frame(unified_path: str | Path, data_config: dict,
                        window: str = "calendar", common_sample: bool = False,
                        cache_dir: str | Path | None = None,
                        phenology: str | None = None,
                        phenology_version: str = "v2",
                        stage_features: bool = False,
                        flowering_index: bool = False,
                        crop_specific_sat: bool = False,
                        stress_features: bool = False) -> pd.DataFrame:
    """prepare_modeling_frame on the unified parquet, cached on disk.

    The cache key covers the dataset file (size + mtime), the window, the
    common-sample flag, the phenology/quality config and FRAME_CACHE_VERSION.
    Writes are atomic (temp file + rename), so concurrent jobs cannot read a
    half-written cache.
    """
    import hashlib
    import os
    unified_path = Path(unified_path)
    if stress_features:                                   # preregistration v20 §2
        return add_stress_features(load_modeling_frame(
            unified_path, data_config, window, common_sample, cache_dir, phenology, phenology_version,
            stage_features, flowering_index, crop_specific_sat), phenology or "full")
    if crop_specific_sat:
        return add_crop_specific_sat(load_modeling_frame(
            unified_path, data_config, window, common_sample, cache_dir, phenology,
            phenology_version, stage_features, flowering_index))
    if flowering_index:
        return add_flowering_index(load_modeling_frame(
            unified_path, data_config, window, common_sample, cache_dir, phenology,
            phenology_version, stage_features))
    if phenology:
        base = load_modeling_frame(unified_path, data_config, window, common_sample, cache_dir)
        if phenology_version == "v3":
            f = add_phenology_features(base, phenology, PHENOLOGY_OBS_V3, PHENOLOGY_SLOTS_V3)
            return add_stage_features(f, phenology) if stage_features else f
        return add_phenology_features(base, phenology)
    if cache_dir is None:
        return prepare_modeling_frame(pd.read_parquet(unified_path), data_config,
                                      window=window, common_sample=common_sample)
    st = unified_path.stat()
    key = json.dumps({"size": st.st_size, "mtime": st.st_mtime_ns, "window": window,
                      "common": bool(common_sample), "v": FRAME_CACHE_VERSION,
                      "phen": data_config["phenology"], "quality": data_config.get("quality"),
                      "crops": data_config.get("crops")}, sort_keys=True)
    h = hashlib.sha256(key.encode()).hexdigest()[:16]
    cache_dir = Path(cache_dir); cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"frame_{window}_{int(bool(common_sample))}_{h}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    out = prepare_modeling_frame(pd.read_parquet(unified_path), data_config,
                                 window=window, common_sample=common_sample)
    tmp = path.with_suffix(f".tmp{os.getpid()}")
    out.to_parquet(tmp, index=True)
    os.replace(tmp, path)
    return out


def _current_part_complete_keys(df: pd.DataFrame, phenology: dict[str, dict[str, int]],
                                min_coverage: float) -> set[tuple]:
    """Winter crop-years whose 1 Jan -> harvest part is >= min_coverage complete."""
    ok: set[tuple] = set()
    for crop, b in phenology.items():
        if not _is_winter(b):
            continue
        h = int(b["harvest_doy"])
        cur = df[(df["crop"] == crop) & (df["doy"] <= h)]
        got = cur.groupby(SAMPLE_KEY)["doy"].nunique()
        ok |= set(got.index[got.to_numpy() >= np.ceil(h * min_coverage)])
    return ok


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
        target_offset: dict[tuple[str, int], float] | None = None,
        sample_weight: dict[tuple[str, str, int], float] | None = None,
    ) -> None:
        if not _HAS_TORCH:
            raise RuntimeError("PyTorch is required for CropYieldDataset.")
        self.spec = feature_spec
        self._sample_weight = sample_weight            # v15 §4: training-loss weight per (landkreis, crop, year)
        self.pipeline = pipeline
        self.cutoff_doy = cutoff_doy

        self._groups: list[pd.DataFrame] = []
        self._targets_z: list[float] = []
        self._keys: list[tuple[str, str, int]] = []

        # Order by date, not day-of-year: identical for single-year windows,
        # and keeps a crop-year's autumn (DOY 235-365) before its spring.
        df = df.sort_values(["landkreis", "crop", "year", "date"])
        for (lk, cr, yr), g in df.groupby(["landkreis", "crop", "year"], sort=False):
            if cutoff_doy is not None:
                g = g[g["doy"] <= cutoff_doy]
            if g.empty:
                continue
            y_abs = float(g.iloc[0][feature_spec.target])
            self._groups.append(g.reset_index(drop=True))
            z = pipeline.transform_target(lk, cr, y_abs, yr)
            if target_offset is not None:
                z -= target_offset.get((cr, int(yr)), 0.0)
            self._targets_z.append(z)
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
        item = {"x_past": x_past, "x_static": x_static, "x_future": x_future, "y": y}
        if self._sample_weight is not None:
            item["w"] = torch.tensor(float(self._sample_weight.get(self._keys[idx], 1.0)), dtype=torch.float32)
        return item

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

    out = {
        "x_past": x_past, "x_future": x_future,
        "x_static": x_static, "y": y, "mask": mask,
    }
    if "w" in batch[0]:
        out["w"] = torch.stack([item["w"] for item in batch])
    return out

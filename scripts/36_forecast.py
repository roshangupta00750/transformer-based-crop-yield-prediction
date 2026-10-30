"""Issue a frozen district x crop yield forecast (paper/preregistration_2026-09-26.md §7).

Model: arm D (selected by the registered rule), checkpoints
itransformer_preD_fc<MODEL_YEAR>_s{42,43,44} from scripts/sbatch_forecast.sh
(train <= MODEL_YEAR-2, early stop on MODEL_YEAR-1). Prediction = mean of the
three seeds; seed_sd = their spread.

Inputs: data/processed/forecast_<YEAR>.parquet (scripts/35). Data end =
the last day on which every daily source (DWD weather, ERA5-Land, cropland
satellite) has values. A crop whose harvest is after the data end keeps its
full timeline and every later day is set to missing, exactly as §6.1 does for
horizons (feature -> train mean, *_missing/_stale -> 1, *_observed -> 0);
lead_days records how far before harvest the forecast was made.

--baseline-only: no in-season data yet (e.g. 2027 issued in autumn 2026). The
backtests show the model is worse than the historical baseline at leads
beyond ~30-60 days, so the registered forecast at such leads IS the
baseline: train-only district mean + crop trend, extrapolated to YEAR.

80% interval (§6.2): prediction + 10th/90th percentile of the same method's
district errors in all earlier test years for that crop (arm D ensemble
backtests 2016-2025 for the model; y_true - y_hist for the baseline).

Output: results/forecasts/forecast_<YEAR>_issued_<DATE>.csv and its SHA-256
in results/forecasts/FROZEN.sha256 (never overwritten).

Usage:
    python scripts/36_forecast.py --year 2026
    python scripts/36_forecast.py --year 2027 --baseline-only --model-year 2026
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.data.dataset import FeaturePipeline, prepare_modeling_frame  # noqa: E402
from src.models import build_model  # noqa: E402
from src.utils import load_data_config, load_model_config  # noqa: E402

_s = importlib.util.spec_from_file_location("ev", ROOT / "scripts/34_eval_prereg.py")
ev = importlib.util.module_from_spec(_s); _s.loader.exec_module(ev)

CK = ROOT / "results/checkpoints"
OUT = ROOT / "results/forecasts"
SEEDS = (42, 43, 44)
SOURCES = ["temperature_air_max_2m", "swvl1", "soiltemp", "ndvi_crop", "lai_crop"]


def data_end(f: pd.DataFrame, year: int) -> pd.Timestamp:
    """Last date on which every source column has values for >= 90% of districts."""
    g = f[f["year"] == year].drop_duplicates(["landkreis", "date"])
    share = g.groupby("date")[SOURCES].apply(lambda d: d.notna().mean())
    ok = share[(share >= 0.9).all(axis=1)]
    return ok.index.max()


def harvest_date(year: int, crop: str, phen: dict) -> pd.Timestamp:
    return pd.Timestamp(f"{year}-01-01") + pd.Timedelta(days=phen[crop]["harvest_doy"] - 1)


def extend_to_harvest(f: pd.DataFrame, year: int, end: pd.Timestamp, phen: dict) -> pd.DataFrame:
    """Add placeholder rows up to harvest for crops harvested after the data end."""
    f = f[~((f["year"] == year) & (f["date"] > end))]
    add = []
    for crop in f["crop"].unique():
        h = harvest_date(year, crop, phen)
        if h <= end:
            continue
        cur = f[(f["year"] == year) & (f["crop"] == crop)].sort_values("date")
        last = cur.groupby("landkreis").tail(1)
        for _, r0 in last.iterrows():
            days = pd.date_range(r0["date"] + pd.Timedelta(days=1), h)
            if len(days):
                r = pd.DataFrame([r0] * len(days))
                r["date"], r["doy"] = days, days.dayofyear
                add.append(r)
    out = pd.concat([f] + add, ignore_index=True) if add else f
    out["sin_doy"] = np.sin(2 * np.pi * out["doy"] / 365)
    out["cos_doy"] = np.cos(2 * np.pi * out["doy"] / 365)
    return out


def mask_after_date(raw: pd.DataFrame, pipe: FeaturePipeline, end: pd.Timestamp) -> pd.DataFrame:
    """ev.mask_after with an absolute cut date instead of harvest - h."""
    out = raw.copy()
    late = pd.to_datetime(out["date"]) > end
    # only columns already in the frame: derived inputs (district anomalies, recent-yield
    # lags) are rebuilt by transform_features from the masked raw columns; masking them
    # here would create all-NaN columns that transform_features then keeps (v10 audit)
    for c in [c for c in pipe.spec.past_observed if c in out.columns]:
        if c.endswith("_missing") or c.endswith("_stale"):
            out.loc[late, c] = 1
        elif c.endswith("_observed"):
            out.loc[late, c] = 0
        else:
            out.loc[late, c] = pipe.feature_mean_[c]
    return out


def interval_table(method: str) -> pd.DataFrame:
    """Per-crop q10/q90 of errors over all backtest years (arm D ensemble)."""
    P = pd.read_parquet(ROOT / "results/metrics/prereg_predictions.parquet")
    P = P[(P["arm"] == "D") & (P["seed"] == "ens") & (P["horizon"] == 0)]
    err = P["y_true"] - (P["y_pred"] if method == "model" else P["y_hist"])
    q = err.groupby(P["crop"]).quantile([0.1, 0.9]).unstack()
    q.columns = ["q10", "q90"]
    return q


def freeze(df: pd.DataFrame, year: int, tag: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"forecast_{year}_{tag}issued_{date.today().isoformat()}.csv"
    if path.exists():
        sys.exit(f"{path.name} already exists; frozen forecasts are never overwritten")
    df.to_csv(path, index=False, float_format="%.3f")
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    with open(OUT / "FROZEN.sha256", "a") as fh:
        fh.write(f"{sha}  {path.name}\n")
    print(f"frozen {path.relative_to(ROOT)}\nsha256 {sha}")
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--model-year", type=int, default=None,
                    help="checkpoints itransformer_preD_fc<MODEL_YEAR>_s* (default: --year)")
    ap.add_argument("--baseline-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="print, do not freeze")
    args = ap.parse_args()
    y, my = args.year, args.model_year or args.year
    dc = load_data_config()
    phen = dc["phenology"]
    pipes = {s: FeaturePipeline.load(CK / f"itransformer_preD_fc{my}_s{s}.pipeline.json")
             for s in SEEDS}
    for p in pipes.values():
        p.require_current_feature_policy()
    pipe0 = pipes[SEEDS[0]]

    if args.baseline_only:
        keys = pd.read_parquet(ROOT / f"data/processed/forecast_{my}.parquet",
                               columns=["landkreis", "crop"]).drop_duplicates()
        lk, cr = keys["landkreis"].to_numpy(), keys["crop"].to_numpy()
        base = pipe0.inverse_transform_array(lk, cr, np.zeros(len(keys)), np.full(len(keys), y))
        fc = pd.DataFrame({"landkreis": lk, "crop": cr, "year": y, "y_pred": base,
                           "y_hist": base, "seed_sd": 0.0, "method": "baseline"})
        today = pd.Timestamp(date.today())
        fc["data_end"] = "none (season not started)"
        fc["lead_days"] = [(harvest_date(y, c, phen) - today).days for c in cr]
        q = interval_table("baseline")
    else:
        f = pd.read_parquet(ROOT / f"data/processed/forecast_{y}.parquet")
        f["date"] = pd.to_datetime(f["date"])
        end = data_end(f, y)
        print(f"data end for {y}: {end.date()}")
        f = extend_to_harvest(f, y, end, phen)
        frame = prepare_modeling_frame(f, dc, window=pipe0.window_,
                                       common_sample=pipe0.common_sample_)
        raw = frame[frame["year"] == y]
        preds = []
        for s, pipe in pipes.items():
            model = build_model("itransformer", load_model_config("itransformer_preD"))
            model.load(CK / f"itransformer_preD_fc{my}_s{s}.pt")
            preds.append(ev.predict(model, pipe, mask_after_date(raw, pipe, end)).assign(seed=s))
        P = pd.concat(preds)
        fc = (P.groupby(["landkreis", "crop", "year"], as_index=False)
                .agg(y_pred=("y_pred", "mean"), seed_sd=("y_pred", "std"),
                     y_hist=("y_hist", "first")))
        fc["method"] = "itransformer_armD_3seed"
        fc["data_end"] = end.date().isoformat()
        fc["lead_days"] = [max(0, (harvest_date(y, c, phen) - end).days) for c in fc["crop"]]
        q = interval_table("model")

    # Only district x crop pairs the district actually grows: a train-only
    # district mean exists (reported yields up to MODEL_YEAR-2).
    grown = [(lk, cr) in pipe0.target_mean_ for lk, cr in zip(fc["landkreis"], fc["crop"])]
    print(f"{len(fc) - sum(grown)} district x crop pairs without yield history dropped")
    fc = fc[grown]
    fc = fc.join(q, on="crop")
    fc["lo80"], fc["hi80"] = fc["y_pred"] + fc["q10"], fc["y_pred"] + fc["q90"]
    fc = fc.drop(columns=["q10", "q90"])
    fc["issued"] = date.today().isoformat()
    fc["model_trained_through"] = f"train<={my - 2}, val={my - 1}"
    fc = fc.sort_values(["crop", "landkreis"]).reset_index(drop=True)
    summary = fc.groupby("crop").agg(districts=("landkreis", "size"), lead_days=("lead_days", "first"),
                                     forecast=("y_pred", "mean"), baseline=("y_hist", "mean"),
                                     seed_sd=("seed_sd", "mean"))
    summary["vs_baseline_pct"] = (summary["forecast"] / summary["baseline"] - 1) * 100
    print(summary.round(2).to_string())
    if not args.dry_run:
        freeze(fc, y, "baseline_" if args.baseline_only else "")


if __name__ == "__main__":
    main()

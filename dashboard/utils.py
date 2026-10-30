"""Shared helpers used across the dashboard pages.

Kept deliberately small — caching, path resolution, JSON loading, and a
subprocess wrapper for kicking off training scripts without blocking the
Streamlit event loop.
"""
from __future__ import annotations

# These two env vars MUST be set before any library that brings its own
# OpenMP (torch, xgboost, lightning, numpy/MKL) is imported. On macOS Intel
# we otherwise hit the classic "two libomp" segfault when two of those
# libraries are imported into the same Python process.
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import subprocess
import sys
import time
import warnings
from pathlib import Path

# Silence noisy upstream warnings that fire whenever a model checkpoint is
# loaded for inference on a per-model page. They're informational only.
warnings.filterwarnings(
    "ignore",
    message=r"Attribute '.*' is an instance of `nn\.Module`",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*save_hyperparameters.*",
    category=UserWarning,
)

import pandas as pd
import streamlit as st


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED = DATA_DIR / "processed" / "dataset.parquet"
INTERIM = DATA_DIR / "interim"
RAW = DATA_DIR / "raw"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
CHECKPOINTS_DIR = PROJECT_ROOT / "results" / "checkpoints"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
LOGS_DIR = Path("/tmp")
DATA_POLICY_PATH = PROCESSED.with_suffix(".policy.json")
RESULTS_POLICY_PATH = METRICS_DIR / "feature_policy.json"

MODELS = ["xgboost", "lstm", "dlinear", "patchtst", "itransformer", "tft"]

MODEL_INFO = {
    "xgboost":      {"kind": "Gradient Boosting",
                     "ref": "Chen & Guestrin, KDD 2016"},
    "lstm":         {"kind": "Recurrent NN",
                     "ref": "Hochreiter & Schmidhuber, 1997"},
    "dlinear":      {"kind": "Linear baseline",
                     "ref": "Zeng et al., AAAI 2023"},
    "patchtst":     {"kind": "Patch Transformer",
                     "ref": "Nie et al., ICLR 2023"},
    "itransformer": {"kind": "Inverted Transformer",
                     "ref": "Liu et al., ICLR 2024"},
    "tft":          {"kind": "Temporal Fusion Transformer",
                     "ref": "Lim et al., 2021"},
}


# --------------------------------------------------------------------------- #
# Data sources (used by the Data page)
# --------------------------------------------------------------------------- #
DATA_SOURCES = [
    {
        "modality": "Weather",
        "source": "Deutscher Wetterdienst (DWD) via wetterdienst",
        "url": "https://opendata.dwd.de/",
        "resolution": "Daily, per-station → Landkreis mean",
        "variables": "T_max, T_min, precipitation, radiation, humidity",
        "years": "2003–2025",
        "license": "DWD CDC: Free use",
    },
    {
        "modality": "Satellite (NDVI / EVI)",
        "source": "MODIS MOD13Q1 via Google Earth Engine",
        "url": "https://lpdaac.usgs.gov/products/mod13q1v061/",
        "resolution": "16-day composite, 250 m → district mean",
        "variables": "NDVI, EVI",
        "years": "2003–2025",
        "license": "NASA EOSDIS: Free for research",
    },
    {
        "modality": "Soil",
        "source": "ISRIC SoilGrids 2.0",
        "url": "https://rest.isric.org/soilgrids/v2.0/",
        "resolution": "Per Landkreis centroid, 0–30 cm topsoil",
        "variables": "Sand %, clay %, SOC, pH (H₂O), bulk density",
        "years": "Static",
        "license": "CC BY 4.0",
    },
    {
        "modality": "Yield labels",
        "source": "Regionaldatenbank Deutschland (table 41241-01-03-4)",
        "url": "https://www.regionalstatistik.de/",
        "resolution": "Annual, Kreis-level (NUTS-3)",
        "variables": "Yield per hectare (dt/ha)",
        "years": "2003–2025",
        "license": "DL-DE BY 2.0",
    },
    {
        "modality": "Geography",
        "source": "Eurostat GISCO NUTS-3 (2021 vintage)",
        "url": "https://ec.europa.eu/eurostat/web/gisco",
        "resolution": "Polygon shapefile",
        "variables": "282 German Landkreis polygons",
        "years": "—",
        "license": "© EuroGeographics",
    },
]


# --------------------------------------------------------------------------- #
# Cached loaders
# --------------------------------------------------------------------------- #
@st.cache_data(ttl=300)
def load_unified() -> pd.DataFrame:
    if not PROCESSED.exists():
        return pd.DataFrame()
    return pd.read_parquet(PROCESSED)


@st.cache_data(ttl=60)
def load_all_metrics() -> pd.DataFrame:
    """Read every <model>_<crop>_horizons.json under results/metrics/.

    DEPRECATED for headline numbers. These files are written by an older
    horizons sweep, are not regenerated by scripts/05b_eval_multicrop.py, and
    mix crops into a single "best per model" row — the Leaderboard was
    rendering May artifacts (tft 0.566) under an August policy marker while
    the truth was -0.037, and one row was winter rapeseed among winter-wheat
    rows. Use load_multicrop_summary() for anything comparative; this stays
    only for the in-season horizon views, and is now also gated on file
    freshness rather than the policy version alone.
    """
    if not results_policy_is_current():
        return pd.DataFrame()
    rows = []
    if not METRICS_DIR.exists():
        return pd.DataFrame()
    # Only files at least as new as the newest checkpoint describe the models
    # currently on disk.
    _fresh_only = True
    for fp in METRICS_DIR.glob("*_horizons.json"):
        # Skip artifacts older than the checkpoints they claim to describe.
        if not artifact_is_fresh(fp):
            continue
        try:
            data = json.loads(fp.read_text())
        except Exception:
            continue
        parts = fp.stem.split("_")
        # filename: <model>(_<suffix>)_<crop>_horizons
        # Treat anything before "winter" / known crop tokens as the model name.
        crop_tokens = {"winter", "summer", "rye", "oats", "potatoes", "sugar",
                       "silage", "triticale"}
        cut = next((i for i, p in enumerate(parts) if p in crop_tokens), len(parts) - 1)
        model = "_".join(parts[:cut])
        crop = "_".join(parts[cut:-1])
        for horizon, m in data.items():
            rows.append({"model": model, "crop": crop, "horizon": horizon, **m})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["h_days"] = df["horizon"].str.extract(r"(\d+)").astype(int)
    return df


@st.cache_data(ttl=3600)
def landkreis_name_map() -> dict[str, str]:
    """Map NUTS-3 code -> human-readable district name (e.g. DE111 -> Stuttgart)."""
    # The interim centroid parquet already carries both — cheap read.
    cent = INTERIM / "landkreis_centroids.parquet"
    if cent.exists():
        df = pd.read_parquet(cent)
        if "landkreis_name" in df.columns:
            return dict(zip(df["landkreis"].astype(str),
                            df["landkreis_name"].astype(str)))
    # Fallback to the shapefile directly.
    shp = RAW / "geo" / "NUTS_RG_01M_2021_4326_LEVL_3.shp"
    if shp.exists():
        try:
            import geopandas as gpd
            g = gpd.read_file(shp)
            g = g[g["CNTR_CODE"] == "DE"]
            return dict(zip(g["NUTS_ID"].astype(str), g["NUTS_NAME"].astype(str)))
        except Exception:
            pass
    return {}


def landkreis_label(code: str, name_map: dict[str, str] | None = None) -> str:
    """Return 'Name (CODE)' if the name is known, else just the code."""
    if name_map is None:
        name_map = landkreis_name_map()
    name = name_map.get(str(code))
    return f"{name} ({code})" if name else str(code)


def newest_checkpoint_mtime() -> float:
    """Newest checkpoint mtime, or 0.0 when none exist."""
    if not CHECKPOINTS_DIR.exists():
        return 0.0
    times = [p.stat().st_mtime for p in CHECKPOINTS_DIR.iterdir()
             if p.suffix in (".pt", ".ubj")]
    return max(times, default=0.0)


def artifact_is_fresh(path) -> bool:
    """True when ``path`` is at least as new as the newest checkpoint.

    The version gate below compares feature_policy_version integers only, so a
    metric file written weeks before the checkpoints it claims to describe
    still passes it. That is not hypothetical: *_horizons.json dated May and
    multicrop_horizons.csv dated June were both being rendered as current
    under an August policy marker, showing pre-v3 numbers (tft 0.566 where the
    truth is -0.037).
    """
    path = Path(path)
    if not path.exists():
        return False
    return path.stat().st_mtime >= newest_checkpoint_mtime()


def results_policy_is_current() -> bool:
    """True only when result artefacts match the processed-data policy."""
    if not DATA_POLICY_PATH.exists() or not RESULTS_POLICY_PATH.exists():
        return False
    try:
        data_policy = json.loads(DATA_POLICY_PATH.read_text())
        result_policy = json.loads(RESULTS_POLICY_PATH.read_text())
        return (
            int(result_policy["feature_policy_version"])
            == int(data_policy["feature_policy_version"])
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


@st.cache_data(ttl=60)
def load_climatology_baseline() -> pd.DataFrame:
    """Per-crop R² of predicting each district's train-fitted baseline.

    The learn-nothing reference: emit the district baseline, learn no anomaly.
    Because R² is computed in absolute dt/ha against an anomaly-framed target,
    the district baseline is handed to every model for free — so without this
    reference the reported numbers look far better than the skill they carry.
    Currently 0.4545 mean, which no model beats.

    Written by the results pages; empty until they have run once.
    """
    p = METRICS_DIR / "climatology_baseline.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


@st.cache_data(ttl=60)
def load_multicrop_summary() -> pd.DataFrame:
    """Read `results/metrics/multicrop_summary.csv` — one row per (model, crop)
    with r2, rmse, mae, n. Produced by `scripts/05b_eval_multicrop.py`.
    """
    if not results_policy_is_current():
        return pd.DataFrame()
    p = METRICS_DIR / "multicrop_summary.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


@st.cache_data(ttl=60)
def load_taxonomy() -> dict:
    """Read `configs/taxonomy.yaml` — the crop-model compatibility hypothesis.

    Returns a dict with three top-level keys::

        crops                 crop -> {season_length_days, phenology_family, ...}
        models                model -> {temporal_receptive_field, ...}
        predicted_winners     crop -> {primary, alternates, rationale}

    Empty dict if the file is missing (dashboard degrades gracefully).
    """
    import yaml
    p = PROJECT_ROOT / "configs" / "taxonomy.yaml"
    if not p.exists():
        return {}
    return yaml.safe_load(p.read_text())


def taxonomy_verdict(crop: str, model: str, tax: dict) -> str:
    """Return 'primary', 'alternate', or '' for the (crop, model) pair."""
    if not tax:
        return ""
    pred = tax.get("predicted_winners", {}).get(crop, {})
    if not pred:
        return ""
    if model == pred.get("primary"):
        return "primary"
    if model in (pred.get("alternates") or []):
        return "alternate"
    return ""


@st.cache_data(ttl=60)
def load_taxonomy_validation() -> pd.DataFrame:
    """Read `results/metrics/taxonomy_validation.csv` — one row per crop with
    predicted_primary, observed_best_model, hit_at_1, hit_at_3, etc.
    """
    if not results_policy_is_current():
        return pd.DataFrame()
    p = METRICS_DIR / "taxonomy_validation.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


@st.cache_data(ttl=60)
def load_pipeline(model: str):
    """Load a model's FeaturePipeline from its JSON sidecar."""
    sys.path.insert(0, str(PROJECT_ROOT))
    from src.data.dataset import FeaturePipeline  # local import to keep page fast
    p = CHECKPOINTS_DIR / f"{model}.pipeline.json"
    if not p.exists():
        return None
    return FeaturePipeline.load(p)


# --------------------------------------------------------------------------- #
# Training subprocess helpers: runs start on this computer (CPU, or a GPU if PyTorch finds one).
# Large runs go to a SLURM cluster through scripts/sbatch_*.sh instead.
# --------------------------------------------------------------------------- #
def launch_training(model: str, crop: str | None = None,
                    extra_args: list[str] | None = None) -> tuple[int, str, str]:
    """Spawn a background training run on this computer.

    Returns ``(pid, log_path, location)``; ``location`` is always ``"local"``.
    """
    return _launch_local(model, crop, extra_args)


def _launch_local(model: str, crop: str | None,
                  extra_args: list[str] | None) -> tuple[int, str, str]:
    log_path = LOGS_DIR / f"dashboard_{model}.log"
    args = ["python", "scripts/03_train.py", "--model", model]
    if crop:
        args += ["--crop", crop]
    if extra_args:
        args += extra_args

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = env.get("OMP_NUM_THREADS", "1")

    with open(log_path, "w") as out:
        proc = subprocess.Popen(
            args, cwd=str(PROJECT_ROOT), env=env,
            stdout=out, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    return proc.pid, str(log_path), "local"


def process_alive(pid: int, location: str = "local") -> bool:
    """True iff the training process is still executing.

    ``location`` is kept for compatibility and is always ``"local"``; ``ps -p``
    filters out zombie processes.
    """
    if pid is None or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        out = subprocess.run(
            ["ps", "-p", str(pid), "-o", "stat="],
            capture_output=True, text=True, timeout=2,
        )
    except Exception:
        return True
    state = (out.stdout or "").strip()
    return bool(state) and not state.startswith("Z")


def stop_process(pid: int, location: str = "local") -> None:
    """Best-effort SIGTERM then SIGKILL the training process."""
    try:
        os.kill(pid, 15)
        time.sleep(1)
        if process_alive(pid):
            os.kill(pid, 9)
    except OSError:
        pass


def tail(path: str | Path, n: int = 200, location: str = "local") -> str:
    """Tail the last ``n`` lines of a training log."""
    path = Path(path)
    if not path.exists():
        return ""
    with open(path, "rb") as f:
        try:
            f.seek(-min(64_000, path.stat().st_size), 2)
        except OSError:
            f.seek(0)
        chunk = f.read().decode("utf-8", errors="replace")
    lines = chunk.splitlines()
    return "\n".join(lines[-n:])


# --------------------------------------------------------------------------- #
# Cosmetic helpers
# --------------------------------------------------------------------------- #
def kpi(label: str, value: str | float, delta: str | None = None,
        help: str | None = None) -> None:
    st.metric(label=label, value=value, delta=delta, help=help)


# Plain-language explanations shown behind the (?) icons on every page, so a
# viewer never meets a unit or a technical term without a definition.
HELP = {
    "dt_ha": "dt/ha = decitonnes per hectare, the unit used in German crop "
             "statistics. 1 decitonne = 100 kg, so 80 dt/ha = 8 tonnes per "
             "hectare (a hectare is 100 m × 100 m).",
    "historical": "Historical average: the district's own average yield for "
                  "this crop over the training years 2003–2022, adjusted for a "
                  "long-term trend where one is clear. It is the simplest "
                  "possible forecast, so a useful model has to beat it.",
    "error": "Error (RMSE, in dt/ha): the typical size of a prediction miss on "
             "the test years 2024–2025, which no model saw during training. "
             "Lower is better; big misses count more than small ones.",
    "pct": "% difference in error: how much bigger (+) or smaller (−) a "
           "model's error is than the error of the historical average. "
           "Negative = the model is more accurate than the historical average.",
    "districts": "German districts (Landkreise and kreisfreie Städte, EU "
                 "level NUTS-3) that have yield records.",
    "crops": "Winter wheat, winter barley, rye, triticale, winter rapeseed, "
             "summer barley, oats, silage maize, potatoes and sugar beet.",
    "years": "Models are trained on 2003–2022, tuned on 2023 and tested on "
             "2024–2025, years they never saw.",
    "ndvi": "NDVI (Normalized Difference Vegetation Index): a satellite "
            "measure of how green and dense the plants are, from about 0 "
            "(bare soil) to about 0.9 (dense green crop). From NASA MODIS.",
    "weather": "Temperature in °C: daily maximum and minimum air "
               "temperature at 2 m height. Precipitation in mm/day: 1 mm = "
               "1 litre of rain per square metre.",
    "gain": "Model miss minus historical-average miss, in dt/ha, per "
            "district. Negative (blue) = the model was closer to the actual "
            "yield; positive (red) = the historical average was closer.",
    "coverage": "Number of districts that reported a yield for that crop in "
                "that year. Gaps come from missing statistics or district "
                "boundary changes.",
    "trend": "Line = median yield across districts; shaded band = the middle "
             "50% of districts. Yield in dt/ha (decitonnes per hectare; "
             "1 dt = 100 kg).",
    "accuracy_w": "Weight on prediction accuracy: lower error in dt/ha ranks "
                  "higher.",
    "interp_w": "Weight on interpretability: how easy it is to explain why "
                "the model predicted what it did (1 = fully transparent).",
    "compute_w": "Weight on low compute cost: how cheap the model is to train "
                 "and run (1 = cheapest).",
    "r2": "R² = share of the variation in yield that the predictions explain "
          "(1 = perfect). It includes the historical average every model gets "
          "for free, so compare it with the null, not with 0.",
    "null": "Climatology null: predicting each district's historical average "
            "and learning nothing else. A model that scores below it has not "
            "learned anything useful.",
    "skill_null": "Model R² minus the null's R². Positive = better than just "
                  "using the historical average.",
    "layers": "Actual yield: what was harvested. Predicted yield: what the "
              "model forecast. Error vs historical average: where the model "
              "beat (blue) or lost to (red) the historical average. All in "
              "dt/ha (decitonnes per hectare; 1 dt = 100 kg).",
    "pred_vs_actual": "Each dot is one district in one test year (2024 or "
                      "2025). Horizontal: actual yield; vertical: predicted "
                      "yield, both in dt/ha. Dots on the dotted line were "
                      "predicted exactly.",
    "models_listed": "Only models whose predictions on this machine match the "
                     "official results are listed.",
    "best_model": "The model that scores best for each crop under the weights "
                  "you set. With accuracy alone, it is the model with the "
                  "lowest error.",
}


def divider(label: str = "") -> None:
    if label:
        st.markdown(f"#### {label}")
    st.markdown("---")


def page_header(title: str, subtitle: str = "") -> None:
    st.markdown(f"## {title}")
    if subtitle:
        st.caption(subtitle)
    if not results_policy_is_current():
        st.error(
            "Model results on this page are legacy pre-policy-v3 artefacts. "
            "Rebuild, retrain, and run `scripts/05b_eval_multicrop.py` before "
            "using them as findings."
        )
    st.markdown("---")

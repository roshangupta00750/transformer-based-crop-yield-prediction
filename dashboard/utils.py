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
        "years": "2003–2023",
        "license": "DWD CDC: Free use",
    },
    {
        "modality": "Satellite (NDVI / EVI)",
        "source": "MODIS MOD13Q1 via Google Earth Engine",
        "url": "https://lpdaac.usgs.gov/products/mod13q1v061/",
        "resolution": "16-day composite, 250 m → district mean",
        "variables": "NDVI, EVI",
        "years": "2003–2023",
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
        "years": "2003–2023",
        "license": "DL-DE BY 2.0",
    },
    {
        "modality": "Geography",
        "source": "Eurostat GISCO NUTS-3 (2021 vintage)",
        "url": "https://ec.europa.eu/eurostat/web/gisco",
        "resolution": "Polygon shapefile",
        "variables": "401 German Landkreis polygons",
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
    """Read every <model>_<crop>_horizons.json under results/metrics/."""
    if not results_policy_is_current():
        return pd.DataFrame()
    rows = []
    if not METRICS_DIR.exists():
        return pd.DataFrame()
    for fp in METRICS_DIR.glob("*_horizons.json"):
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
# Training subprocess helpers — GPU box via SSH, with local CPU fallback
# --------------------------------------------------------------------------- #
# SSH alias for the JMU GPU box (see ~/.ssh/config block `Host gpu`).
GPU_HOST = "gpu"
GPU_PROJECT_DIR = "~/crop_yield_benchmark"
GPU_REMOTE_LOG = "/tmp/dash_train_{model}.log"

# Models that empirically cannot train on the GPU box (RAM-bound on the
# 15 GB shared lab machine). pytorch-forecasting's TFT instantiation
# OOMs even at batch_size=32; documented in the project memory.
GPU_BLOCKLIST: set[str] = {"tft"}


def _gpu_reachable(timeout_s: int = 3) -> bool:
    """Quick BatchMode SSH probe — returns True only if passwordless login works."""
    try:
        result = subprocess.run(
            ["ssh", "-o", f"ConnectTimeout={timeout_s}",
             "-o", "BatchMode=yes", GPU_HOST, "true"],
            capture_output=True, timeout=timeout_s + 2,
        )
        return result.returncode == 0
    except Exception:
        return False


def launch_training(model: str, crop: str | None = None,
                    extra_args: list[str] | None = None,
                    force_local: bool = False) -> tuple[int, str, str]:
    """Spawn a background training run.

    Tries the GPU box via SSH first; falls back to local CPU when SSH is
    unreachable, when ``force_local`` is set, or when the model is in
    :data:`GPU_BLOCKLIST` (currently TFT, which OOMs on the lab box).

    Returns ``(pid, log_path, location)`` where ``location`` is
    ``"remote"`` or ``"local"``. For remote runs the PID is the *remote*
    PID and ``log_path`` is the remote path on the GPU box; pass
    ``location`` back to :func:`process_alive` and :func:`tail` so they
    route correctly.
    """
    use_remote = (not force_local
                  and model not in GPU_BLOCKLIST
                  and _gpu_reachable())
    if use_remote:
        try:
            return _launch_remote(model, crop, extra_args)
        except Exception:
            # Any SSH/launch failure: fall back to local rather than block the user.
            pass
    return _launch_local(model, crop, extra_args)


def _launch_remote(model: str, crop: str | None,
                   extra_args: list[str] | None) -> tuple[int, str, str]:
    log_path = GPU_REMOTE_LOG.format(model=model)
    train_cmd = f"python scripts/03_train.py --model {model}"
    if crop:
        train_cmd += f" --crop {crop}"
    if extra_args:
        train_cmd += " " + " ".join(extra_args)

    remote_cmd = (
        f"cd {GPU_PROJECT_DIR} && "
        f"source .venv/bin/activate && "
        f"export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 && "
        f"nohup {train_cmd} > {log_path} 2>&1 & echo $!"
    )
    result = subprocess.run(
        ["ssh", "-o", "ConnectTimeout=5", GPU_HOST, remote_cmd],
        capture_output=True, text=True, timeout=15,
    )
    if result.returncode != 0:
        raise RuntimeError(f"remote launch failed: {result.stderr.strip()[:200]}")
    last = result.stdout.strip().splitlines()[-1].strip()
    if not last.isdigit():
        raise RuntimeError(f"remote launch returned non-PID: {last!r}")
    return int(last), log_path, "remote"


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

    ``location`` is ``"local"`` or ``"remote"``. For remote runs we shell
    out over SSH to ``kill -0 <pid>`` on the GPU box; the local path uses
    ``ps -p`` to filter out zombies (see longer note in the prior version
    of this function).
    """
    if pid is None or pid <= 0:
        return False
    if location == "remote":
        try:
            r = subprocess.run(
                ["ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes",
                 GPU_HOST, f"kill -0 {pid} 2>/dev/null && echo alive"],
                capture_output=True, text=True, timeout=10,
            )
            return r.returncode == 0 and "alive" in r.stdout
        except Exception:
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
    if location == "remote":
        subprocess.run(
            ["ssh", GPU_HOST, f"kill {pid} 2>/dev/null; sleep 1; kill -9 {pid} 2>/dev/null"],
            capture_output=True, timeout=15,
        )
        return
    try:
        os.kill(pid, 15)
        time.sleep(1)
        if process_alive(pid):
            os.kill(pid, 9)
    except OSError:
        pass


def tail(path: str | Path, n: int = 200, location: str = "local") -> str:
    """Tail the last ``n`` lines of a training log, either local or remote."""
    if location == "remote":
        try:
            r = subprocess.run(
                ["ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes",
                 GPU_HOST, f"tail -n {n} {path} 2>/dev/null"],
                capture_output=True, text=True, timeout=10,
            )
            return r.stdout or ""
        except Exception:
            return ""
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
def kpi(label: str, value: str | float, delta: str | None = None) -> None:
    st.metric(label=label, value=value, delta=delta)


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

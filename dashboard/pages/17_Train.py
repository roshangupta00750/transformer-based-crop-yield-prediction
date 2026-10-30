"""Train — training console.

One question: **what is trained, and how do I launch a run?**

Default view is deliberately small — a headline KPI row, the checkpoint
inventory, the launcher, and live job status. Destination artefacts, resolved
hyperparameters, config drift, the score-to-beat breakdown, epoch curves and
sidecar/metadata health are all still here, collapsed one click away.

Every number rendered here is read from disk. Nothing is hardcoded and nothing
is estimated. Heavy runs go to SLURM via `scripts/sbatch_*.sh`; this page only
launches local jobs and knows nothing about cluster queue state.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import (CHECKPOINTS_DIR, HELP, METRICS_DIR,
                   MODEL_INFO, MODELS, PROJECT_ROOT, divider,
                   launch_training, load_climatology_baseline,
                   load_multicrop_summary, page_header, process_alive,
                   stop_process, tail)

st.set_page_config(page_title="Train — Crop Yield Benchmark", layout="wide")

CONFIG_DIR = PROJECT_ROOT / "configs"
TRAIN_LOGS_DIR = PROJECT_ROOT / "results" / "logs"
SUMMARY_CSV = METRICS_DIR / "multicrop_summary.csv"
CKPT_EXTS = (".pt", ".ubj")
# 03_train.py writes <name>_meta.json then <name>.pt back to back; anything
# further apart than this was written by a different run.
META_PAIR_TOLERANCE_S = 300


# --------------------------------------------------------------------------- #
# Cached disk scans
# --------------------------------------------------------------------------- #
@st.cache_data(ttl=300)
def crop_list() -> list[str]:
    """Crops from configs/data.yaml (cheap — avoids loading the 57 MB parquet)."""
    p = CONFIG_DIR / "data.yaml"
    if not p.exists():
        return []
    try:
        return list(yaml.safe_load(p.read_text()).get("crops") or [])
    except Exception:
        return []


@st.cache_data(ttl=300)
def experiment_cfg() -> dict:
    p = CONFIG_DIR / "experiment.yaml"
    if not p.exists():
        return {}
    try:
        return yaml.safe_load(p.read_text()) or {}
    except Exception:
        return {}


@st.cache_data(ttl=300)
def model_cfg(model: str) -> dict:
    p = CONFIG_DIR / "models" / f"{model}.yaml"
    if not p.exists():
        return {}
    try:
        return yaml.safe_load(p.read_text()) or {}
    except Exception:
        return {}


def _split_name(stem: str, crops: list[str]) -> tuple[str, str, str, str]:
    """Split a checkpoint stem into (model, crop, framing, variant).

    Mirrors the naming rule in scripts/03_train.py:
    ``<model>[_<crop>][_abs]``. Anything left over that is not a known crop is
    reported as a free-form variant tag (e.g. ``tft.55m``, ``xgboost_monthly``)
    rather than silently attributed to a crop.
    """
    framing, base = "anomaly", stem
    if base.endswith("_abs"):
        base, framing = base[:-4], "absolute"
    model = ""
    for m in sorted(MODELS, key=len, reverse=True):
        if base == m or base.startswith(m + "_") or base.startswith(m + "."):
            model = m
            break
    rest = base[len(model):].lstrip("_.") if model else base
    crop = rest if rest in crops else ""
    variant = "" if (crop or not rest) else rest
    return model, crop, framing, variant


@st.cache_data(ttl=30)
def scan_checkpoints() -> pd.DataFrame:
    """One row per checkpoint file on disk, with its sidecar / metadata state."""
    crops = crop_list()
    if not CHECKPOINTS_DIR.exists():
        return pd.DataFrame()
    rows = []
    for fp in sorted(CHECKPOINTS_DIR.iterdir()):
        if not fp.is_file() or fp.suffix.lower() not in CKPT_EXTS:
            continue
        stem = fp.stem
        model, crop, framing, variant = _split_name(stem, crops)
        ck_mtime = fp.stat().st_mtime
        pipe_p = CHECKPOINTS_DIR / f"{stem}.pipeline.json"
        meta_p = METRICS_DIR / f"{stem}_meta.json"

        fit_s, best_epoch, meta_status = None, None, "missing"
        if meta_p.exists():
            delta = meta_p.stat().st_mtime - ck_mtime
            if delta < -META_PAIR_TOLERANCE_S:
                meta_status = "stale (predates checkpoint)"
            elif delta > META_PAIR_TOLERANCE_S:
                meta_status = "overwritten by a later run"
            else:
                meta_status = "current"
            try:
                meta = json.loads(meta_p.read_text())
                fit_s = meta.get("fit_seconds")
                best_epoch = meta.get("best_epoch")
            except Exception:
                meta_status = "unreadable"

        rows.append({
            "name": stem,
            "checkpoint": fp.name,
            "model": model or "(unrecognised)",
            "crop": crop or "(all crops)",
            "framing": framing,
            "variant": variant,
            "size_mb": fp.stat().st_size / 1e6,
            "trained_at": datetime.fromtimestamp(ck_mtime),
            "ckpt_mtime": ck_mtime,
            "age_days": (time.time() - ck_mtime) / 86400,
            "pipeline_sidecar": "present" if pipe_p.exists() else "MISSING",
            "meta": meta_status,
            "fit_min": (fit_s / 60.0) if isinstance(fit_s, (int, float)) else None,
            "best_epoch": "not recorded" if best_epoch is None else str(best_epoch),
        })
    return pd.DataFrame(rows)


@st.cache_data(ttl=30)
def summary_mtime() -> float | None:
    return SUMMARY_CSV.stat().st_mtime if SUMMARY_CSV.exists() else None


@st.cache_data(ttl=30)
def scan_epoch_logs() -> pd.DataFrame:
    """Find every per-epoch training log under results/logs/.

    A file counts only if it actually has an ``epoch`` column; the directory it
    sits under names the model.
    """
    if not TRAIN_LOGS_DIR.exists():
        return pd.DataFrame()
    rows = []
    for fp in sorted(TRAIN_LOGS_DIR.rglob("*.csv")):
        try:
            head = pd.read_csv(fp, nrows=1)
        except Exception:
            continue
        if "epoch" not in head.columns:
            continue
        parts = fp.relative_to(TRAIN_LOGS_DIR).parts
        rows.append({
            "model": parts[0] if len(parts) > 1 else fp.stem,
            "path": str(fp),
            "rel_path": str(fp.relative_to(PROJECT_ROOT)),
            "written_at": datetime.fromtimestamp(fp.stat().st_mtime),
        })
    return pd.DataFrame(rows)


@st.cache_data(ttl=30)
def load_epoch_log(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


# --------------------------------------------------------------------------- #
# Eval attribution — which summary rows a given checkpoint actually produced
# --------------------------------------------------------------------------- #
def owned_eval_rows(model: str, crop: str, framing: str, variant: str,
                    summary: pd.DataFrame,
                    percrop_tft: set[str]) -> tuple[pd.DataFrame, str]:
    """Rows of multicrop_summary.csv that this checkpoint produced.

    Replicates the checkpoint-resolution rule in
    ``dashboard/_model_template.py::_predict_test_set``: TFT prefers
    ``tft_<crop>.pt`` when it exists and otherwise falls back to ``tft.pt``;
    every other model is evaluated from its all-crop checkpoint only. Returns
    ``(rows, reason)`` where ``reason`` explains an empty result.
    """
    if summary.empty:
        return summary, "no eval on disk"
    if variant:
        return summary.iloc[0:0], "variant name — outside eval scope"
    if framing != "anomaly":
        return summary.iloc[0:0], "absolute-yield track — eval runs anomaly checkpoints"
    if model not in MODELS:
        return summary.iloc[0:0], "unrecognised model name"
    sub = summary[summary["model"] == model]
    if model == "tft":
        if crop:
            return sub[sub["crop"] == crop], ""
        shadowed = sub[~sub["crop"].isin(percrop_tft)]
        if shadowed.empty:
            return shadowed, "shadowed by per-crop tft checkpoints"
        return shadowed, ""
    if crop:
        return sub.iloc[0:0], "eval loads the all-crop checkpoint for this model"
    return sub, ""


# --------------------------------------------------------------------------- #
# Load once
# --------------------------------------------------------------------------- #
page_header("Training console",
            "What is trained, and how to launch a run. Detail sits in the "
            "collapsed sections at the foot of the page.")

summary = load_multicrop_summary()
clim = load_climatology_baseline()
ckpts = scan_checkpoints()
crops = crop_list()
percrop_tft = {c for c in crops
               if (CHECKPOINTS_DIR / f"tft_{c}.pt").exists()}
exp_cfg = experiment_cfg()
budget = (exp_cfg.get("budget") or {})
sm = summary_mtime()

clim_mean = float(clim["r2"].mean()) if not clim.empty else None
model_means = (summary.groupby("model")["r2"].mean().sort_values(ascending=False)
               if not summary.empty else pd.Series(dtype=float))

# Attribute every checkpoint to the eval rows it actually produced. Needed by
# both the headline KPI row and the inventory table, so it happens once here.
inv = ckpts.copy()
if not inv.empty:
    r2s, n_eval, reasons, freshness = [], [], [], []
    for _, row in inv.iterrows():
        rows, reason = owned_eval_rows(
            row["model"], "" if row["crop"] == "(all crops)" else row["crop"],
            row["framing"], row["variant"], summary, percrop_tft)
        if rows is None or rows.empty:
            r2s.append(None)
            n_eval.append(0)
            reasons.append(reason or "no matching eval row")
            freshness.append("—")
        else:
            r2s.append(float(rows["r2"].mean()))
            n_eval.append(int(rows["crop"].nunique()))
            reasons.append("")
            freshness.append(
                "STALE — eval predates checkpoint"
                if (sm is not None and sm < row["ckpt_mtime"]) else "current")
    inv["eval_r2"] = r2s
    inv["crops_evaluated"] = n_eval
    inv["eval_note"] = reasons
    inv["eval_freshness"] = freshness

n_stale_eval = (int((inv["eval_freshness"] == "STALE — eval predates checkpoint").sum())
                if not inv.empty else 0)
n_meta_bad = int((inv["meta"] != "current").sum()) if not inv.empty else 0
n_no_sidecar = int((inv["pipeline_sidecar"] == "MISSING").sum()) if not inv.empty else 0
n_unscored = int((inv["crops_evaluated"] == 0).sum()) if not inv.empty else 0

# --------------------------------------------------------------------------- #
# PRIMARY 1 — what is on disk, and is any of it better than learning nothing
# --------------------------------------------------------------------------- #
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Checkpoints on disk", str(len(inv)),
          help="Saved trained models in results/checkpoints/.")
k2.metric("Newest",
          inv["trained_at"].max().strftime("%m-%d %H:%M") if not inv.empty else "—")
if not model_means.empty:
    k3.metric(f"Best mean R² ({model_means.index[0]})", f"{model_means.iloc[0]:.3f}",
              help=HELP["r2"])
else:
    k3.metric("Best mean R²", "no eval")
k4.metric("Climatology baseline R²",
          f"{clim_mean:.3f}" if clim_mean is not None else "not computed",
          help=HELP["null"])
if clim_mean is not None and not model_means.empty:
    k5.metric("Models beating it",
              f"{int((model_means > clim_mean).sum())} of {len(model_means)}",
              help=HELP["skill_null"])
else:
    k5.metric("Models beating it", "—")

st.caption(
    "The climatology baseline predicts each district's train-fitted mean and "
    "learns nothing; it is the floor every R² on this page has to clear. "
    "Read live from `results/metrics/climatology_baseline.csv`"
    + (f" ({len(clim)} crops)" if not clim.empty else "")
    + ". Model means from `results/metrics/multicrop_summary.csv`, written "
    + (datetime.fromtimestamp(sm).strftime("%Y-%m-%d %H:%M") if sm else "—") + "."
)

# A stale eval means the R² column below does not describe the file it sits on.
# That one stays visible whenever it fires; everything quieter is in Diagnostics.
if n_stale_eval:
    st.warning(
        f"{n_stale_eval} checkpoint(s) are newer than `multicrop_summary.csv` — "
        f"their `Eval R²` below was measured on an older file. Re-run "
        f"`scripts/05b_eval_multicrop.py`."
    )

# --------------------------------------------------------------------------- #
# PRIMARY 2 — checkpoint inventory
# --------------------------------------------------------------------------- #
divider("What is trained")

if inv.empty:
    st.warning(f"No `.pt` / `.ubj` files under `{CHECKPOINTS_DIR}`.")
else:
    only_scored = st.checkbox(
        "Show only checkpoints the multi-crop eval actually reads", value=False,
        help="Hides absolute-yield and variant checkpoints, and any per-crop "
             "checkpoint the eval script never loads.",
    )
    view = inv[inv["crops_evaluated"] > 0] if only_scored else inv

    show = view[["checkpoint", "model", "crop", "framing", "trained_at",
                 "eval_r2", "crops_evaluated", "eval_note"]].rename(columns={
        "checkpoint": "File", "model": "Model", "crop": "Crop",
        "framing": "Framing", "trained_at": "Trained at",
        "eval_r2": "Eval R²", "crops_evaluated": "Crops eval'd",
        "eval_note": "Why no score",
    })
    st.dataframe(
        show.style.format({"Eval R²": "{:.3f}"}),
        width="stretch", hide_index=True,
        column_config={"Trained at": st.column_config.DatetimeColumn(
            format="YYYY-MM-DD HH:mm")},
    )
    st.caption(
        "`Eval R²` is the mean of the multicrop_summary.csv rows *this file* "
        "produced"
        + (f" — the learn-nothing baseline is {clim_mean:.3f}, so anything below "
           f"that is worse than no model at all."
           if clim_mean is not None else ".")
        + " Sidecar state, fit time and freshness are in Diagnostics below."
    )

# --------------------------------------------------------------------------- #
# PRIMARY 3 — launcher
# --------------------------------------------------------------------------- #
divider("Launch a run")

_pid = st.session_state.get("running_pid")
busy = bool(_pid) and not st.session_state.get("running_finished", False)

left, right = st.columns([1, 1.4])

with left:
    model = st.selectbox("Model family", MODELS)
    crop_choice = st.selectbox("Crop", ["(all crops)"] + crops, index=0)
    crop = None if crop_choice == "(all crops)" else crop_choice
    no_anomaly = st.checkbox(
        "Predict absolute yield (no anomaly framing)", value=False,
        help="Off: the model predicts a yield anomaly against the district "
             "baseline. On: it predicts absolute yield, and is credited for "
             "knowing that baseline. The two tracks write different files.",
    )
    start = st.button("Start training", type="primary", disabled=busy,
                      width="stretch")

target = "this computer"

# Destination artefacts — computed with exactly the rule scripts/03_train.py
# applies, so the overwrite consequence is visible before launch.
suffix = (f"_{crop}" if crop else "") + ("_abs" if no_anomaly else "")
run_name = f"{model}{suffix}"
ckpt_ext = ".ubj" if model == "xgboost" else ".pt"
destinations = [
    ("checkpoint", CHECKPOINTS_DIR / f"{run_name}{ckpt_ext}"),
    ("feature pipeline", CHECKPOINTS_DIR / f"{run_name}.pipeline.json"),
    ("run metadata", METRICS_DIR / f"{run_name}_meta.json"),
]
n_over = sum(1 for _, p in destinations if p.exists())

with right:
    st.markdown(f"Routes to **{target}**, writing under the name `{run_name}`.")
    # One-line overwrite summary — this guarded a real data-loss bug, so it
    # stays out of the expander. The per-file table is in "Run detail".
    if n_over:
        st.warning(
            f"{n_over} of {len(destinations)} destination files already exist "
            f"under `{run_name}` and will be replaced by this run."
        )
    else:
        st.caption(f"All {len(destinations)} destination files are new — "
                   f"nothing existing is overwritten.")
    if crop and model != "tft":
        st.warning(
            f"`scripts/05b_eval_multicrop.py` evaluates `{model}` from the "
            f"all-crop checkpoint only, so `{run_name}{ckpt_ext}` will not "
            f"appear in multicrop_summary.csv. Per-crop checkpoints are read "
            f"back for TFT only."
        )
    st.caption("Runs on this computer. Large runs go to a SLURM cluster through "
               "`scripts/sbatch_*.sh`; their queue state is not visible here.")

if start:
    extra = (["--no-anomaly"] if no_anomaly else [])
    if suffix:
        extra += ["--suffix", suffix]
    pid, log, location = launch_training(model, crop, extra)
    st.session_state.update({
        "running_pid": pid,
        "running_log": log,
        "running_location": location,
        "running_model": model,
        "running_crop": crop or "all crops",
        "running_name": run_name,
        "running_started": time.time(),
        "running_finished": False,
    })
    st.session_state.pop("running_log_final", None)
    st.rerun()

# --------------------------------------------------------------------------- #
# PRIMARY 4 — live status (isolated fragment so polling does not rerun the page)
# --------------------------------------------------------------------------- #
divider("Live status")


def _render_live() -> None:
    pid = st.session_state.get("running_pid")
    if not pid:
        st.caption("No active job. Configure a run above and press "
                   "**Start training**.")
        return

    location = st.session_state.get("running_location", "local")
    finished = st.session_state.get("running_finished", False)
    alive = False if finished else process_alive(pid, location)

    elapsed = int(time.time() - st.session_state.get("running_started",
                                                     time.time()))
    mins, secs = divmod(elapsed, 60)
    where = "this computer"

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Run", st.session_state.get("running_name", "—"))
    c2.metric("Where", where)
    c3.metric("PID", str(pid))
    c4.metric("Status", "running" if alive else "exited")
    c5.metric("Elapsed", f"{mins}m {secs:02d}s")

    if alive:
        log_text = tail(st.session_state["running_log"], n=300,
                        location=location)
    else:
        if not finished:
            st.session_state["running_log_final"] = tail(
                st.session_state["running_log"], n=300, location=location)
            st.session_state["running_finished"] = True
            st.rerun(scope="app")
        log_text = st.session_state.get("running_log_final", "")

    if not alive:
        st.success("Run finished. Re-run `scripts/05b_eval_multicrop.py` to "
                   "refresh multicrop_summary.csv before trusting new scores.")

    b1, b2 = st.columns([1, 6])
    if b1.button("Stop job", disabled=not alive):
        try:
            stop_process(pid, location)
        except Exception as e:
            st.error(f"Could not stop PID {pid}: {e}")
        st.session_state["running_finished"] = True
        st.rerun(scope="app")
    b2.caption(f"Log: `{st.session_state.get('running_log', '')}`")

    if alive:
        # While a job runs the log *is* the live status, so it stays open.
        st.code(log_text or "(no log output yet)", language="text")
    else:
        with st.expander("Final job log (last 300 lines)", expanded=False):
            st.code(log_text or "(no log output yet)", language="text")


st.fragment(run_every="5s")(_render_live)()

# --------------------------------------------------------------------------- #
# Collapsed detail — nothing below is deleted, only folded away
# --------------------------------------------------------------------------- #
divider("Detail")

# --- A. Run detail: destinations, CLI, resolved config, drift --------------- #
with st.expander(
        f"Run detail — destination files, equivalent CLI, resolved "
        f"configuration for `{run_name}`", expanded=False):

    dest_rows = []
    for kind, p in destinations:
        exists = p.exists()
        dest_rows.append({
            "artefact": kind,
            "path": str(p.relative_to(PROJECT_ROOT)),
            "state": "OVERWRITTEN" if exists else "new file",
            "size (MB)": round(p.stat().st_size / 1e6, 2) if exists else None,
            "written": (datetime.fromtimestamp(p.stat().st_mtime)
                        .strftime("%Y-%m-%d %H:%M") if exists else "—"),
        })
    st.dataframe(pd.DataFrame(dest_rows), width="stretch", hide_index=True)

    cli = f"python scripts/03_train.py --model {model}"
    if crop:
        cli += f" --crop {crop}"
    if no_anomaly:
        cli += " --no-anomaly"
    if suffix:
        cli += f" --suffix {suffix}"
    st.caption("Equivalent CLI — paste into an sbatch script for cluster runs:")
    st.code(cli, language="bash")

    mcfg = model_cfg(model)
    cfg_rows = []
    for k, v in mcfg.items():
        if k == "name":
            continue
        if k in ("max_epochs", "early_stopping_patience", "max_minutes"):
            continue
        cfg_rows.append({"parameter": k, "value": str(v),
                         "source": f"configs/models/{model}.yaml"})
    for k in ("max_epochs", "early_stopping_patience", "max_minutes"):
        if k in mcfg:
            cfg_rows.append({"parameter": k, "value": str(mcfg[k]),
                             "source": f"configs/models/{model}.yaml (overrides budget)"})
        elif k in budget:
            cfg_rows.append({"parameter": k, "value": str(budget[k]),
                             "source": "configs/experiment.yaml budget"})
    for k in ("seed", "device", "deterministic"):
        if k in exp_cfg:
            cfg_rows.append({"parameter": k, "value": str(exp_cfg[k]),
                             "source": "configs/experiment.yaml"})
    cfg_rows += [
        {"parameter": "family", "value": MODEL_INFO[model]["kind"],
         "source": "dashboard/utils.py MODEL_INFO"},
        {"parameter": "reference", "value": MODEL_INFO[model]["ref"],
         "source": "dashboard/utils.py MODEL_INFO"},
    ]

    cfg_left, cfg_right = st.columns([1.2, 1])
    with cfg_left:
        if not mcfg:
            st.warning(f"No `configs/models/{model}.yaml` on disk — the run will "
                       f"fall back to the shared budget only.")
        st.dataframe(pd.DataFrame(cfg_rows), width="stretch", hide_index=True)

    with cfg_right:
        meta_p = METRICS_DIR / f"{run_name}_meta.json"
        if not meta_p.exists():
            st.info(f"No `results/metrics/{run_name}_meta.json` — this "
                    f"destination has no recorded previous run.")
        else:
            try:
                meta = json.loads(meta_p.read_text())
            except Exception:
                meta = {}
            prev = meta.get("config") or {}
            m1, m2, m3 = st.columns(3)
            fs = meta.get("fit_seconds")
            m1.metric("Last fit",
                      f"{fs / 60:.1f} min" if isinstance(fs, (int, float)) else "—")
            m2.metric("Best epoch",
                      "not recorded" if meta.get("best_epoch") is None
                      else str(meta["best_epoch"]))
            m3.metric("Metadata written",
                      datetime.fromtimestamp(meta_p.stat().st_mtime)
                      .strftime("%m-%d %H:%M"))
            drift = [{"parameter": k,
                      "last run": str(prev.get(k, "—")),
                      "current yaml": str(mcfg.get(k, "—"))}
                     for k in sorted(set(prev) | set(mcfg))
                     if k != "name" and str(prev.get(k, "—")) != str(mcfg.get(k, "—"))]
            if drift:
                st.caption("Config drift since the last recorded run at this name:")
                st.dataframe(pd.DataFrame(drift), width="stretch", hide_index=True)
            else:
                st.caption("Recorded config matches the current YAML.")

# --- B. Score to beat ------------------------------------------------------- #
with st.expander(
        f"Score to beat — where `{model}` stands on "
        f"{crop_choice.replace('(all crops)', 'all crops')}", expanded=False):

    if summary.empty:
        st.warning("`results/metrics/multicrop_summary.csv` is absent or fails "
                   "the feature-policy check — no scores to compare against.")
    elif crop:
        cols = [c for c in ("model", "r2", "rmse", "mae", "n")
                if c in summary.columns]
        crop_rows = summary[summary["crop"] == crop][cols].copy()
        clim_crop = clim[clim["crop"] == crop] if not clim.empty else clim
        clim_r2 = float(clim_crop["r2"].iloc[0]) if len(clim_crop) else None
        if clim_r2 is not None:
            base = {"model": "climatology baseline"}
            for c in ("r2", "rmse", "mae", "n"):
                if c in cols:
                    base[c] = clim_crop[c].iloc[0]
            crop_rows = pd.concat([crop_rows, pd.DataFrame([base])],
                                  ignore_index=True)
        crop_rows = (crop_rows.sort_values("r2", ascending=False)
                     .reset_index(drop=True))

        cur = crop_rows[crop_rows["model"] == model]
        cur_r2 = float(cur["r2"].iloc[0]) if not cur.empty else None
        best_r2 = float(crop_rows["r2"].iloc[0]) if not crop_rows.empty else None
        best_model = crop_rows["model"].iloc[0] if not crop_rows.empty else "—"
        rank = (int(cur.index[0]) + 1) if not cur.empty else None

        c1, c2, c3, c4 = st.columns(4)
        c1.metric(f"{model} on {crop}",
                  f"{cur_r2:.3f}" if cur_r2 is not None else "no row")
        c2.metric(f"Best on {crop} ({best_model})",
                  f"{best_r2:.3f}" if best_r2 is not None else "—")
        c3.metric(f"Climatology on {crop}",
                  f"{clim_r2:.3f}" if clim_r2 is not None else "not computed")
        c4.metric("Rank incl. baseline",
                  f"{rank} of {len(crop_rows)}" if rank else "—")

        disp = crop_rows.copy()
        disp.insert(0, "", ["<- selected" if m == model else ""
                            for m in disp["model"]])
        st.dataframe(
            disp[[""] + cols]
            .rename(columns={"model": "Model", "r2": "R²",
                             "rmse": "RMSE (dt/ha)", "mae": "MAE (dt/ha)",
                             "n": "n test points"})
            .style.format({"R²": "{:.3f}", "RMSE (dt/ha)": "{:.2f}",
                           "MAE (dt/ha)": "{:.2f}"}),
            width="stretch", hide_index=True,
        )
        st.caption("The `climatology baseline` row is the learn-nothing "
                   "reference, ranked alongside the models rather than beside "
                   "them, so a model that fails to clear it is visibly last.")
    else:
        mine = summary[summary["model"] == model]
        cur_mean = float(mine["r2"].mean()) if not mine.empty else None
        best_mean = float(model_means.iloc[0])
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(f"{model} mean R² (all crops)",
                  f"{cur_mean:.3f}" if cur_mean is not None else "no rows")
        c2.metric(f"Best mean R² ({model_means.index[0]})", f"{best_mean:.3f}")
        c3.metric("Climatology mean R²",
                  f"{clim_mean:.3f}" if clim_mean is not None else "not computed")
        c4.metric("Crops evaluated", f"{mine['crop'].nunique()} of {len(crops)}")

        if not mine.empty:
            per_crop = (summary.pivot_table(index="crop", columns="model",
                                            values="r2")
                        .dropna(how="all"))
            rank_of = per_crop.rank(axis=1, ascending=False, method="min")
            tbl = pd.DataFrame({
                "crop": per_crop.index,
                f"{model} R²": per_crop[model].values,
                "best R²": per_crop.max(axis=1).values,
                "best model": per_crop.idxmax(axis=1).values,
                # A partial summary leaves NaN ranks; keep them nullable rather
                # than crashing the page on astype(int).
                f"{model} rank": rank_of[model].astype("Int64").values,
            })
            if not clim.empty:
                tbl = tbl.merge(
                    clim[["crop", "r2"]].rename(columns={"r2": "climatology R²"}),
                    on="crop", how="left")
            tbl = tbl.sort_values(f"{model} R²", ascending=False)
            fmt = {f"{model} R²": "{:.3f}", "best R²": "{:.3f}"}
            if "climatology R²" in tbl.columns:
                fmt["climatology R²"] = "{:.3f}"
            st.dataframe(tbl.style.format(fmt), width="stretch", hide_index=True)

    st.caption(
        "Source: `results/metrics/multicrop_summary.csv`, written "
        + (datetime.fromtimestamp(sm).strftime("%Y-%m-%d %H:%M") if sm else "—")
        + ", and `results/metrics/climatology_baseline.csv`."
    )

# --- C. Diagnostics: sidecar health, epoch-log coverage, curves ------------- #
with st.expander(
        f"Diagnostics — sidecar and metadata state ({n_meta_bad} of {len(inv)} "
        f"not current), epoch-log coverage, training curves", expanded=False):

    if inv.empty:
        st.caption("No checkpoints on disk, so nothing to diagnose.")
    else:
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Total size", f"{inv['size_mb'].sum() / 1000:.2f} GB")
        d2.metric("Metadata not current", str(n_meta_bad))
        d3.metric("Missing pipeline sidecar", str(n_no_sidecar))
        d4.metric("Not covered by eval", str(n_unscored))

        diag = inv[["checkpoint", "size_mb", "age_days", "fit_min",
                    "pipeline_sidecar", "meta", "best_epoch",
                    "eval_freshness"]].rename(columns={
            "checkpoint": "File", "size_mb": "Size (MB)",
            "age_days": "Age (days)", "fit_min": "Fit (min)",
            "pipeline_sidecar": "Pipeline sidecar", "meta": "Metadata",
            "best_epoch": "Best epoch", "eval_freshness": "Eval freshness",
        })
        st.dataframe(
            diag.style.format({"Size (MB)": "{:.1f}", "Age (days)": "{:.1f}",
                               "Fit (min)": "{:.1f}"}),
            width="stretch", hide_index=True,
        )
        st.caption(
            "`Eval R²` above is attributed with the same rule as "
            "`dashboard/_model_template.py` (TFT prefers `tft_<crop>.pt`, else "
            "`tft.pt`; other models are evaluated from their all-crop "
            "checkpoint only). `Eval freshness` compares multicrop_summary.csv's "
            "mtime to the checkpoint's; `Metadata` compares the `_meta.json` "
            f"mtime to the checkpoint's within {META_PAIR_TOLERANCE_S}s."
        )

    st.markdown("---")

    logs = scan_epoch_logs()
    logged_models = set(logs["model"]) if not logs.empty else set()

    cov = pd.DataFrame([{
        "model": m,
        "per-epoch log": ("yes" if m in logged_models else "none on disk"),
        "log files": int((logs["model"] == m).sum()) if not logs.empty else 0,
        "best_epoch in metadata": (
            ckpts.loc[ckpts["name"] == m, "best_epoch"].iloc[0]
            if not ckpts.empty and (ckpts["name"] == m).any() else "no metadata"),
    } for m in MODELS])
    st.dataframe(cov, width="stretch", hide_index=True)
    st.caption(
        f"{len(MODELS) - len(logged_models & set(MODELS))} of {len(MODELS)} "
        f"models write no per-epoch history: `src/training/trainer.py` attaches "
        f"no CSV logger and hardcodes `best_epoch=None`, so their only "
        f"training-time record on disk is total `fit_seconds`. Nothing exists "
        f"to plot for them."
    )

    if logs.empty:
        st.warning(f"No epoch logs under `{TRAIN_LOGS_DIR}`.")
    else:
        pick = st.selectbox("Epoch log", logs["rel_path"].tolist(),
                            format_func=lambda p: p.replace("results/logs/", ""))
        row = logs[logs["rel_path"] == pick].iloc[0]
        df = load_epoch_log(row["path"])

        if df.empty or "epoch" not in df.columns:
            st.warning(f"`{pick}` has no readable epoch data.")
        else:
            ep_train = (df.dropna(subset=["train_loss_epoch"])[["epoch", "train_loss_epoch"]]
                        if "train_loss_epoch" in df.columns else pd.DataFrame())
            val_cols = [c for c in df.columns if c.startswith("val_")]
            ep_val = (df.dropna(subset=["val_loss"])
                      if "val_loss" in df.columns else pd.DataFrame())
            steps = (df.dropna(subset=["train_loss_step"])[["step", "train_loss_step"]]
                     if "train_loss_step" in df.columns else pd.DataFrame())

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Epoch boundaries logged", str(len(ep_train)))
            m2.metric("Optimiser steps logged", str(len(steps)))
            m3.metric("Final train loss (epoch)",
                      f"{ep_train['train_loss_epoch'].iloc[-1]:.3f}"
                      if not ep_train.empty else "—")
            m4.metric("Best val loss",
                      f"{ep_val['val_loss'].min():.3f}" if not ep_val.empty else "—")

            log_y = st.checkbox("Log-scale loss axis", value=True,
                                help="Train and validation losses sit on "
                                     "different scales in these logs; a log "
                                     "axis keeps both readable in one panel.")

            fig = go.Figure()
            if not steps.empty:
                fig.add_trace(go.Scatter(
                    x=steps["step"], y=steps["train_loss_step"],
                    name="train (step)", mode="lines",
                    line=dict(width=1, color="#9ecae1"), opacity=0.5))
                win = max(5, len(steps) // 50)
                fig.add_trace(go.Scatter(
                    x=steps["step"],
                    y=steps["train_loss_step"].rolling(win, min_periods=1).mean(),
                    name=f"train (rolling {win})", mode="lines",
                    line=dict(width=2.5, color="#2171b5")))
            if not ep_train.empty and "step" in df.columns:
                marks = df.dropna(subset=["train_loss_epoch"])
                fig.add_trace(go.Scatter(
                    x=marks["step"], y=marks["train_loss_epoch"],
                    name="train (epoch mean)", mode="markers+lines",
                    marker=dict(size=10, symbol="square", color="#08306b")))
            if not ep_val.empty and "step" in df.columns:
                fig.add_trace(go.Scatter(
                    x=ep_val["step"], y=ep_val["val_loss"], name="val loss",
                    mode="markers+lines",
                    marker=dict(size=10, symbol="diamond", color="#d94801")))
            fig.update_layout(height=420, xaxis_title="optimiser step",
                              yaxis_title="loss",
                              yaxis_type="log" if log_y else "linear",
                              legend=dict(orientation="h", y=1.12),
                              margin=dict(t=40, b=10))
            st.plotly_chart(fig, width="stretch")

            keep = ["epoch", "step", "train_loss_epoch"] + val_cols
            ep_tbl = df.dropna(subset=[c for c in ("train_loss_epoch", "val_loss")
                                       if c in df.columns], how="all")
            ep_tbl = ep_tbl[[c for c in keep if c in ep_tbl.columns]]
            st.dataframe(ep_tbl.reset_index(drop=True), width="stretch",
                         hide_index=True)
            st.caption(f"Source: `{pick}`, written "
                       f"{row['written_at'].strftime('%Y-%m-%d %H:%M')}. Rows "
                       f"are epoch boundaries only; `val_*` columns come from "
                       f"the validation pass.")

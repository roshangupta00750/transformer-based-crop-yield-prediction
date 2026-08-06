"""Train page — launch model training (GPU box via SSH, with local CPU
fallback) and tail its log in the UI. Does NOT block the Streamlit event
loop.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import (CHECKPOINTS_DIR, GPU_BLOCKLIST, GPU_HOST, MODEL_INFO,
                   MODELS, _gpu_reachable, launch_training, page_header,
                   process_alive, stop_process, tail)

st.set_page_config(page_title="Train — Crop Yield Benchmark",
                   layout="wide")

page_header("Train models",
            "Tries the GPU box via SSH first; falls back to local CPU when "
            "the box is unreachable or the model is on the GPU blocklist "
            "(currently TFT, which OOMs the lab box during pf init).")
st.caption("_Launch a training run from the browser — routed to the GPU box when reachable, falling back to local CPU for TFT — and tail the running job's log inline._")

# --------------------------------------------------------------------------- #
# Device probe
# --------------------------------------------------------------------------- #
gpu_up = _gpu_reachable()
chip = (f":green[GPU box `{GPU_HOST}` reachable]" if gpu_up
        else f":red[GPU box `{GPU_HOST}` not reachable — local CPU fallback]")
st.markdown(f"**Compute target:** {chip}")

# --------------------------------------------------------------------------- #
# Form
# --------------------------------------------------------------------------- #
left, right = st.columns([1, 2])

with left:
    st.subheader("Configure run")
    model = st.selectbox("Model family", MODELS)
    from utils import load_unified
    available_crops = sorted(load_unified()["crop"].unique().tolist()) \
        if not load_unified().empty else ["winter_wheat", "winter_rapeseed"]
    crop = st.selectbox("Crop", available_crops + ["(all)"],
                        index=len(available_crops))
    no_anomaly = st.checkbox(
        "Predict absolute yield (no anomaly framing)",
        value=False,
        help="When on, the model predicts absolute yield. R² will look higher "
             "but mostly because the model is being credited for knowing the "
             "district baseline. See Home page for the framing discussion.",
    )
    force_local = st.checkbox(
        "Force local CPU (skip GPU box)",
        value=False,
        help="Override the auto-routing. Useful for reproducibility checks "
             "or when you specifically want CPU numbers.",
    )
    suffix = "_abs" if no_anomaly else ""
    st.caption(f"Checkpoint will be saved as "
               f"`results/checkpoints/{model}{suffix}.pt`")

    # Existing checkpoint status
    existing_pt = (CHECKPOINTS_DIR / f"{model}{suffix}.pt").exists() or \
                  (CHECKPOINTS_DIR / f"{model}{suffix}.ubj").exists()
    if existing_pt:
        st.info("Checkpoint already exists. Re-training will overwrite it.")

    # Predict where this run will land before the user presses go.
    if force_local:
        target = "local CPU (forced)"
    elif model in GPU_BLOCKLIST:
        target = f"local CPU ({model} is on the GPU blocklist — RAM-bound on lab box)"
    elif gpu_up:
        target = f"GPU box (`{GPU_HOST}`)"
    else:
        target = "local CPU (GPU box unreachable)"
    st.caption(f"This run will train on: **{target}**")

    start = st.button(
        "Start training", type="primary",
        disabled=("running_pid" in st.session_state and
                  process_alive(st.session_state.get("running_pid", -1),
                                st.session_state.get("running_location", "local"))),
    )

with right:
    st.subheader("Selected model")
    info = MODEL_INFO[model]
    st.markdown(
        f"### {model.upper()}\n"
        f"**Family:** {info['kind']}  \n"
        f"**Reference:** {info['ref']}"
    )
    on_gpu_box = (not force_local and model not in GPU_BLOCKLIST and gpu_up)
    st.markdown(
        f"**Config file:** `configs/models/{model}.yaml`  \n"
        f"**Wall-clock cap:** 60 min (TFT enforced at 55 min)  \n"
        f"**Device:** {'CUDA (remote RTX 3060)' if on_gpu_box else 'CPU (local)'}"
    )

# --------------------------------------------------------------------------- #
# Launch
# --------------------------------------------------------------------------- #
if start:
    real_crop = None if crop == "(all)" else crop
    # Compose the suffix so a per-crop run never collides with the all-crop
    # artefacts, and so the absolute-yield track stays distinct from both.
    suffix = (f"_{real_crop}" if real_crop else "") + ("_abs" if no_anomaly else "")
    extra = (["--no-anomaly"] if no_anomaly else [])
    if suffix:
        extra += ["--suffix", suffix]
    pid, log, location = launch_training(model, real_crop, extra,
                                         force_local=force_local)
    st.session_state["running_pid"] = pid
    st.session_state["running_log"] = log
    st.session_state["running_location"] = location
    st.session_state["running_model"] = model
    st.session_state["running_crop"] = real_crop or "all"
    st.session_state["running_started"] = time.time()
    where = "GPU box" if location == "remote" else "local CPU"
    st.success(f"Launched **{model}** on {where} (PID {pid}). Log: `{log}`")
    st.rerun()

# --------------------------------------------------------------------------- #
# Live status / log tail
# --------------------------------------------------------------------------- #
st.markdown("---")
st.subheader("Live status")

pid = st.session_state.get("running_pid")
if pid:
    location = st.session_state.get("running_location", "local")
    alive = process_alive(pid, location)
    elapsed = int(time.time() - st.session_state["running_started"])
    status = "running" if alive else "exited"
    mins, secs = divmod(elapsed, 60)
    where = "GPU box (remote)" if location == "remote" else "local CPU"
    st.markdown(
        f"**Model:** {st.session_state['running_model']} · "
        f"**Crop:** {st.session_state['running_crop']} · "
        f"**Where:** {where} · "
        f"**PID:** {pid} · "
        f"**Status:** {status} · "
        f"**Elapsed:** {mins}m {secs}s"
    )
    if not alive:
        st.success("Run finished — check the per-model page for new metrics.")
    if st.button("Stop background job"):
        try:
            stop_process(pid, location)
            st.warning(f"Stopped PID {pid} on {where}.")
        except Exception as e:
            st.error(f"Could not stop: {e}")
        st.session_state.pop("running_pid", None)
        st.session_state.pop("running_location", None)
        st.rerun()

    log_text = tail(st.session_state["running_log"], n=300, location=location)
    if log_text:
        st.code(log_text, language="text")
    else:
        st.caption("(no log output yet)")

    if alive:
        # Auto-refresh every 5s so logs trickle in. Remote tail costs one
        # SSH round-trip per refresh (~0.5s), which is fine.
        time.sleep(5)
        st.rerun()
else:
    st.caption("No active job. Configure a run above and press **Start training**.")

"""Reusable template for the per-model dashboard pages.

Each ``pages/3..8_<model>.py`` is a thin wrapper that calls ``render_model_page``
with the model name. Keeps the six pages DRY.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (CHECKPOINTS_DIR, METRICS_DIR, MODEL_INFO, PROJECT_ROOT,
                   divider, kpi, landkreis_name_map, load_all_metrics,
                   load_multicrop_summary, load_pipeline, load_taxonomy,
                   load_unified, page_header)


def _read_model_config(model: str) -> dict | None:
    import yaml
    p = PROJECT_ROOT / "configs" / "models" / f"{model}.yaml"
    if not p.exists():
        return None
    return yaml.safe_load(p.read_text())


def _read_meta(model: str) -> dict | None:
    p = METRICS_DIR / f"{model}_meta.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _predict_test_set_tft(crop: str,
                          cutoff_doy: int | None = None,
                          years: list[int] | None = None) -> tuple[pd.DataFrame, dict | None]:
    """TFT prediction adapted for the (landkreis, crop) series scheme.

    Each district-crop is one series spanning 2003-2023. For each target
    year we slice the dataset to end at ``cutoff_doy`` (or harvest_doy
    if not given) of that year, then ask pf to predict the LAST day of
    each series. ``years`` defaults to configured ``test_years``.
    """
    try:
        from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
        from src.data.dataset import apply_phenology_window
        from src.training.tft_adapter import (build_timeseries_dataset,
                                              prepare_tft_dataframe)
        from src.utils import load_data_config
    except Exception as e:  # noqa: BLE001
        return pd.DataFrame(), {"error": f"TFT eval imports failed: {e}"}

    data_cfg = load_data_config()
    df_all = load_unified()
    if df_all.empty:
        return pd.DataFrame(), {"error": "Unified dataset not found."}
    df_all = apply_phenology_window(df_all, data_cfg["phenology"])

    train_end = data_cfg["splits"]["temporal"]["train_end_year"]
    test_years = years if years is not None else \
        data_cfg["splits"]["temporal"]["test_years"]

    soil_cols = [c for c in df_all.columns
                 if c in ("sand", "clay", "soc", "ph", "bulk_density")]
    weather_sat_cols = [c for c in df_all.columns
                        if c.startswith(("temperature_", "precip", "radiation",
                                         "humid", "ndvi", "evi", "lai", "ndwi"))]

    # IMPORTANT: build train_ds on ALL crops the model was trained on, so the
    # categorical encoders + GroupNormalizer match the checkpoint. Filtering to
    # one crop here would re-fit the normalizers and produce garbage.
    df_train_all = df_all[df_all["year"] <= train_end]
    if df_train_all.empty:
        return pd.DataFrame(), {"error": "Empty training data."}
    df_tr, cols = prepare_tft_dataframe(
        df_train_all, soil_cols=soil_cols, weather_sat_cols=weather_sat_cols,
    )
    train_ds = build_timeseries_dataset(
        df_tr, cols, max_encoder_length=180, max_prediction_length=1,
    )

    # Now restrict to the requested crop for prediction only.
    df = df_all[df_all["crop"] == crop].copy()

    ckpt = None
    for cand in (CHECKPOINTS_DIR / "tft.pt",
                 CHECKPOINTS_DIR / "tft_abs.pt"):
        if cand.exists():
            ckpt = cand
            break
    if ckpt is None:
        return pd.DataFrame(), {"error": "No TFT checkpoint found."}
    try:
        m = TemporalFusionTransformer.load_from_checkpoint(str(ckpt))
        m.eval()
    except Exception as e:  # noqa: BLE001
        return pd.DataFrame(), {"error": f"TFT load failed: {e}"}

    # Predict one year at a time: include 2003 through that year, slice each
    # series at the requested cutoff (or harvest day), and let pf predict the
    # final day. cutoff_doy lets the caller evaluate in-season horizons.
    cap_doy = cutoff_doy if cutoff_doy is not None else None
    rows: list[dict] = []
    for yr in test_years:
        if cap_doy is not None:
            df_slice = df[(df["year"] < yr)
                          | ((df["year"] == yr) & (df["doy"] <= cap_doy))].copy()
        else:
            df_slice = df[df["year"] <= yr].copy()
        df_slice_p, _ = prepare_tft_dataframe(
            df_slice, soil_cols=soil_cols, weather_sat_cols=weather_sat_cols,
        )
        try:
            test_ds = TimeSeriesDataSet.from_dataset(
                train_ds, df_slice_p, stop_randomization=True, predict=True,
            )
        except Exception as e:  # noqa: BLE001
            return pd.DataFrame(), {"error": f"TFT test dataset year={yr}: {e}"}

        loader = test_ds.to_dataloader(
            train=False, batch_size=max(len(test_ds), 1), num_workers=0,
        )
        # Use mode="quantiles" so we get all three trained quantiles (P10/P50/P90)
        # rather than the to_prediction() median. Falls back to single-column
        # output if the loaded checkpoint isn't multi-quantile.
        try:
            preds = m.predict(loader, mode="quantiles",
                              return_y=True, return_index=True)
        except Exception:
            try:
                preds = m.predict(loader, mode="prediction",
                                  return_y=True, return_index=True)
            except Exception as e:  # noqa: BLE001
                return pd.DataFrame(), {"error": f"TFT predict year={yr}: {e}"}

        raw = preds.output.cpu().numpy()
        # Possible shapes: (N,), (N, 1), (N, Q), (N, 1, Q) → normalise to (N, Q).
        while raw.ndim > 2:
            raw = raw.squeeze(1) if raw.shape[1] == 1 else raw.reshape(raw.shape[0], -1)
        if raw.ndim == 1:
            raw = raw.reshape(-1, 1)
        n_q = raw.shape[1]
        if n_q == 3:
            p10, p50, p90 = raw[:, 0], raw[:, 1], raw[:, 2]
        elif n_q == 1:
            p10 = p50 = p90 = raw[:, 0]
        else:  # e.g. 7-quantile output — pick the median and the extremes
            mid = n_q // 2
            p10, p50, p90 = raw[:, 0], raw[:, mid], raw[:, -1]

        y_true = (preds.y[0].cpu().numpy().squeeze()
                  if hasattr(preds, "y") and preds.y is not None else None)

        idx = preds.index if hasattr(preds, "index") else None
        if idx is None or "series_id" not in idx.columns:
            continue
        for i, sid in enumerate(idx["series_id"].astype(str).tolist()):
            lk = sid.split("__", 1)[0]
            yt = float(y_true[i]) if (y_true is not None and i < len(y_true)) else None
            rows.append({
                "landkreis": lk, "year": int(yr),
                "y_pred": float(p50[i]) if i < len(p50) else None,
                "y_pred_p10": float(p10[i]) if i < len(p10) else None,
                "y_pred_p90": float(p90[i]) if i < len(p90) else None,
                "y_true": yt,
            })

    if not rows:
        return pd.DataFrame(), {"error": "TFT produced no predictions."}
    out = pd.DataFrame(rows).dropna(subset=["y_pred", "y_true"])
    out["residual"] = out["y_pred"] - out["y_true"]
    return out, None


def _predict_test_set(model: str, crop: str,
                      cutoff_doy: int | None = None,
                      years: list[int] | None = None) -> tuple[pd.DataFrame, dict | None]:
    """Recompute predictions on a split for plotting / fusion.

    Returns (pred_df, error). pred_df has columns: landkreis, year, doy,
    y_true, y_pred (both in dt/ha at the sample level — one row per
    (landkreis, year, crop)).

    Pass ``cutoff_doy`` to evaluate at an in-season horizon: the test
    time-series is truncated at that day-of-year before prediction.

    Pass ``years`` to predict on an arbitrary set of years (e.g., the
    validation split for stacking / fusion). Default = configured
    ``test_years`` — preserves backward compatibility with all callers.
    """
    if model == "tft":
        return _predict_test_set_tft(crop, cutoff_doy=cutoff_doy, years=years)

    pipeline = load_pipeline(model)
    if pipeline is None:
        return pd.DataFrame(), {"error": "Pipeline JSON missing — train first."}

    from src.data.dataset import (CropYieldDataset, apply_phenology_window,
                                  collate_pad)
    from src.models import build_model
    from src.utils import load_data_config, load_model_config
    from torch.utils.data import DataLoader

    data_cfg = load_data_config()
    model_cfg = load_model_config(model)

    df = load_unified()
    if df.empty:
        return pd.DataFrame(), {"error": "Unified dataset not found."}
    df = df[df["crop"] == crop]
    df = apply_phenology_window(df, data_cfg["phenology"])
    target_years = years if years is not None else \
        data_cfg["splits"]["temporal"]["test_years"]
    df_test = df[df["year"].isin(target_years)]
    if df_test.empty:
        return pd.DataFrame(), {"error": f"Empty split for years={target_years}."}

    df_test_t = pipeline.transform_features(df_test)
    if "district_baseline" in pipeline.spec.static \
            and "district_baseline" not in df_test_t.columns:
        gm = pipeline.target_global_mean_
        df_test_t = df_test_t.copy()
        df_test_t["district_baseline"] = [
            pipeline.target_mean_.get((lk, cr), gm)
            for lk, cr in zip(df_test_t["landkreis"], df_test_t["crop"])
        ]

    # Load checkpoint, handling .pt / .ubj / .json
    m = build_model(model, model_cfg)
    ckpt = None
    for ext in (".pt", ".ubj", ".json"):
        cand = CHECKPOINTS_DIR / f"{model}{ext}"
        if cand.exists():
            ckpt = cand
            break
    if ckpt is None:
        return pd.DataFrame(), {"error": f"No checkpoint for {model}."}
    try:
        m.load(ckpt)
    except Exception as e:  # noqa: BLE001
        return pd.DataFrame(), {"error": f"Load failed: {e}"}
    m.device = "cpu"

    ds = CropYieldDataset(df_test_t, pipeline.spec, pipeline, cutoff_doy=cutoff_doy)
    if len(ds) == 0:
        return pd.DataFrame(), {"error": "Empty CropYieldDataset."}
    loader = DataLoader(ds, batch_size=128, collate_fn=collate_pad)
    try:
        z_pred = np.asarray(m.predict(loader)).ravel()
    except Exception as e:  # noqa: BLE001
        return pd.DataFrame(), {"error": f"Predict failed: {e}"}

    keys = ds.keys
    lk = np.array([k[0] for k in keys])
    cr = np.array([k[1] for k in keys])
    yrs = np.array([k[2] for k in keys])
    y_pred = pipeline.inverse_transform_array(lk, cr, z_pred)
    y_true = np.array([pipeline.inverse_transform_target(*k[:2], z)
                       for k, z in zip(keys, ds._targets_z)])

    return pd.DataFrame({
        "landkreis": lk, "year": yrs, "y_true": y_true, "y_pred": y_pred,
        "residual": y_pred - y_true,
    }), None


def render_model_page(model: str, narrative_intro: str | None = None,
                      tldr: str | None = None) -> None:
    info = MODEL_INFO.get(model, {})
    st.set_page_config(page_title=f"{model.upper()} — Crop Yield",
                       layout="wide")
    page_header(model.upper(), info.get("ref", ""))
    if tldr:
        st.markdown(f"### TL;DR\n\n{tldr}")
    if narrative_intro:
        st.caption(f"_{narrative_intro}_")

    # Config + meta
    cfg = _read_model_config(model)
    meta = _read_meta(model)

    c1, c2, c3, c4 = st.columns(4)
    metrics_all = load_all_metrics()
    sub = metrics_all[(metrics_all["model"] == model)]
    best = sub.sort_values("r2", ascending=False).head(1)
    with c1:
        kpi("Best R²", f"{best['r2'].values[0]:.3f}" if len(best) else "—")
    with c2:
        kpi("Best RMSE (dt/ha)",
            f"{best['rmse'].values[0]:.2f}" if len(best) else "—")
    with c3:
        kpi("Best @ horizon",
            best["horizon"].values[0] if len(best) else "—")
    with c4:
        if meta and meta.get("fit_seconds"):
            mins, secs = divmod(int(meta["fit_seconds"]), 60)
            kpi("Train time", f"{mins}m {secs}s")
        else:
            kpi("Train time", "—")

    # Hyperparameters
    divider("Hyperparameters")
    if cfg:
        st.json(cfg, expanded=False)
    else:
        st.caption("No config found.")

    # ── Taxonomy panel: where this model belongs ─────────────────────────
    tax = load_taxonomy()
    mc = load_multicrop_summary()
    if tax and not mc.empty:
        divider("Where this model belongs — taxonomy view")

        # Predicted-for (from taxonomy) — crops where this model is primary/alt
        predicted_primary = [c for c, v in tax.get("predicted_winners", {}).items()
                             if (v or {}).get("primary") == model]
        predicted_alt = [c for c, v in tax.get("predicted_winners", {}).items()
                         if model in ((v or {}).get("alternates") or [])]

        # Observed-wins: crops where this model has the highest R² across all models
        model_wins = (mc.sort_values("r2", ascending=False)
                        .groupby("crop", as_index=False)
                        .first())
        observed_wins = model_wins[model_wins["model"] == model]["crop"].tolist()

        # Hit / miss for this model's predicted-primary crops
        hits = [c for c in predicted_primary if c in observed_wins]
        misses = [c for c in predicted_primary if c not in observed_wins]

        # Taxonomy attribute row for this model (architecture family, receptive
        # field, feature interaction, …)
        model_attr = tax.get("models", {}).get(model, {})

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            kpi("Predicted primary for",
                f"{len(predicted_primary)} crops",
                delta=", ".join(predicted_primary) if predicted_primary else "—")
        with k2:
            kpi("Also listed alternate on",
                f"{len(predicted_alt)} crops",
                delta=", ".join(predicted_alt) if predicted_alt else "—")
        with k3:
            kpi("Observed test winner on",
                f"{len(observed_wins)} crops",
                delta=", ".join(observed_wins) if observed_wins else "—")
        with k4:
            n_hits = len(hits)
            n_pri = len(predicted_primary)
            kpi("Taxonomy hit rate (this model)",
                f"{n_hits}/{n_pri}" if n_pri else "—",
                delta="of its predicted primaries won on test" if n_pri else None)

        if model_attr:
            st.markdown("**Inductive-bias fingerprint:**")
            st.dataframe(
                pd.DataFrame([{
                    "Architecture family": model_attr.get("architecture_family", "—"),
                    "Temporal receptive field": model_attr.get("temporal_receptive_field", "—"),
                    "Feature interaction": model_attr.get("feature_interaction", "—"),
                    "Long-horizon capacity": model_attr.get("long_horizon_capacity", "—"),
                    "Narrow-window capacity": model_attr.get("narrow_window_capacity", "—"),
                    "Training-data efficiency": model_attr.get("training_data_efficiency", "—"),
                }]),
                width="stretch", hide_index=True,
            )

        if hits or misses:
            cA, cB = st.columns(2)
            with cA:
                if hits:
                    st.success(
                        f"**Taxonomy confirmed on:** " + ", ".join(f"`{c}`" for c in hits)
                    )
            with cB:
                if misses:
                    actual_for_miss = {
                        c: model_wins[model_wins["crop"] == c]["model"].iloc[0]
                        for c in misses
                    }
                    formatted = ", ".join(
                        f"`{c}` (won by `{v}`)" for c, v in actual_for_miss.items()
                    )
                    st.warning(f"**Taxonomy predicted this model for but lost:** {formatted}")

        # Rationale strings from taxonomy per crop this model is primary for
        if predicted_primary:
            with st.expander("Taxonomy rationales for this model's primaries"):
                for c in predicted_primary:
                    rat = (tax.get("predicted_winners", {}).get(c) or {}).get("rationale", "")
                    if rat:
                        st.markdown(f"- **{c}**: {rat}")

    # Horizon metrics table + plot
    divider("In-season horizons")
    if sub.empty:
        st.warning("No evaluation results yet. Run `04_evaluate.py --model "
                   f"{model}` or use the **Train** page (eval auto-runs after train).")
        return

    horizon_show = sub.sort_values("h_days")[
        ["horizon", "r2", "rmse", "mae", "mape", "n"]
    ].rename(columns={"horizon": "Horizon", "r2": "R²", "rmse": "RMSE",
                      "mae": "MAE", "mape": "MAPE %", "n": "n"})
    st.dataframe(horizon_show.style.format({"R²": "{:.3f}", "RMSE": "{:.2f}",
                                            "MAE": "{:.2f}", "MAPE %": "{:.2f}"}),
                 width="stretch", hide_index=True)

    fig = px.line(sub.sort_values("h_days"), x="h_days",
                  y=["r2", "rmse", "mae"], markers=True,
                  labels={"h_days": "Days pre-harvest", "value": "metric",
                          "variable": "metric"})
    fig.update_layout(height=350)
    st.plotly_chart(fig, width="stretch")

    # Predictions
    divider("Predictions vs. truth (2022–2023 test set)")
    available_crops = sorted(load_unified()["crop"].unique().tolist()) \
        if not load_unified().empty else ["winter_wheat", "winter_rapeseed"]
    default_idx = available_crops.index("winter_wheat") if "winter_wheat" in available_crops else 0
    crop = st.selectbox("Crop", available_crops, index=default_idx, key=f"{model}_crop")

    with st.spinner("Loading checkpoint and running inference…"):
        preds, err = _predict_test_set(model, crop)

    if err:
        st.error(err["error"])
        return

    # Decorate predictions with human-readable district names.
    name_map = landkreis_name_map()
    preds = preds.copy()
    preds["landkreis_name"] = preds["landkreis"].astype(str).map(
        lambda c: name_map.get(c, c))

    # Scatter — hover shows district name + NUTS-3 code.
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=preds["y_true"], y=preds["y_pred"], mode="markers",
        marker=dict(size=6, opacity=0.55, color=preds["year"], colorscale="Viridis",
                    showscale=True, colorbar=dict(title="Year")),
        customdata=preds[["landkreis_name", "landkreis"]].to_numpy(),
        hovertemplate=(
            "%{customdata[0]}<br>NUTS-3: %{customdata[1]}<br>"
            "Year: %{marker.color}<br>"
            "Actual: %{x:.1f} dt/ha<br>Predicted: %{y:.1f} dt/ha<extra></extra>"),
    ))
    lo = min(preds["y_true"].min(), preds["y_pred"].min())
    hi = max(preds["y_true"].max(), preds["y_pred"].max())
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                             line=dict(dash="dash", color="grey"),
                             name="y = x"))
    fig.update_layout(
        height=480, title=f"Predicted vs. actual — {crop}",
        xaxis_title="Actual yield (dt/ha)",
        yaxis_title="Predicted yield (dt/ha)",
        showlegend=False,
    )
    st.plotly_chart(fig, width="stretch")

    # Residuals
    rcol1, rcol2 = st.columns(2)
    with rcol1:
        fig = px.histogram(preds, x="residual", nbins=40,
                           title="Residual distribution (pred − actual)")
        fig.update_layout(height=320, xaxis_title="dt/ha")
        st.plotly_chart(fig, width="stretch")
    with rcol2:
        per_year = preds.groupby("year")["residual"].agg(
            mean="mean", abs_mean=lambda s: s.abs().mean(), std="std"
        ).reset_index()
        fig = px.bar(per_year, x="year", y="abs_mean",
                     title="Mean absolute residual by year",
                     labels={"abs_mean": "MAE (dt/ha)"})
        fig.update_layout(height=320)
        st.plotly_chart(fig, width="stretch")

    # Quantile / calibration block — only when the predictions carry P10/P90.
    if {"y_pred_p10", "y_pred_p90"}.issubset(preds.columns):
        divider("Prediction intervals (P10–P90) and calibration")
        st.caption(
            "TFT was trained with QuantileLoss against the 0.1 / 0.5 / 0.9 "
            "quantiles. A well-calibrated 80% interval should capture 80% "
            "of true values; less = over-confident, more = under-confident."
        )

        # KPIs
        in_band = ((preds["y_pred_p10"] <= preds["y_true"]) &
                   (preds["y_true"] <= preds["y_pred_p90"])).mean() * 100
        below = (preds["y_true"] < preds["y_pred_p10"]).mean() * 100
        above = (preds["y_true"] > preds["y_pred_p90"]).mean() * 100
        mean_width = (preds["y_pred_p90"] - preds["y_pred_p10"]).mean()
        crps_proxy = (preds["y_pred_p90"] - preds["y_pred_p10"]).median()

        k1, k2, k3, k4 = st.columns(4)
        with k1: kpi("Empirical 80% coverage", f"{in_band:.1f}%",
                     delta=f"{in_band - 80:+.1f} pp vs 80%")
        with k2: kpi("Mean interval width", f"{mean_width:.1f} dt/ha")
        with k3: kpi("Truth below P10", f"{below:.1f}%")
        with k4: kpi("Truth above P90", f"{above:.1f}%")

        # Interval scatter — predictions with P10-P90 error bars.
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=preds["y_true"], y=preds["y_pred"],
            mode="markers",
            error_y=dict(
                type="data", symmetric=False,
                array=preds["y_pred_p90"] - preds["y_pred"],
                arrayminus=preds["y_pred"] - preds["y_pred_p10"],
                thickness=0.8, width=0,
            ),
            marker=dict(size=5, opacity=0.55, color=preds["year"],
                        colorscale="Viridis", showscale=True,
                        colorbar=dict(title="Year")),
            customdata=preds[["landkreis_name", "landkreis",
                              "y_pred_p10", "y_pred_p90"]].to_numpy(),
            hovertemplate=(
                "%{customdata[0]}<br>NUTS-3: %{customdata[1]}<br>"
                "Year: %{marker.color}<br>"
                "Actual: %{x:.1f} dt/ha<br>"
                "P50: %{y:.1f}  (P10 %{customdata[2]:.1f}, P90 %{customdata[3]:.1f})"
                "<extra></extra>"),
            name="P50 with P10–P90 bars",
        ))
        lo = float(min(preds["y_true"].min(), preds["y_pred_p10"].min()))
        hi = float(max(preds["y_true"].max(), preds["y_pred_p90"].max()))
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                                 line=dict(dash="dash", color="grey"),
                                 name="y = x", showlegend=False))
        fig.update_layout(
            height=460, title="Predicted P50 ± P10/P90 vs actual",
            xaxis_title="Actual yield (dt/ha)",
            yaxis_title="Predicted yield (dt/ha)",
        )
        st.plotly_chart(fig, width="stretch")

        # Where the truth lands relative to the interval — sanity histogram.
        # Position 0 = at P10, 0.5 = at P50, 1.0 = at P90; truth outside [P10,P90]
        # is clamped beyond.
        denom = (preds["y_pred_p90"] - preds["y_pred_p10"]).replace(0, np.nan)
        position = ((preds["y_true"] - preds["y_pred_p10"]) / denom).dropna()
        fig = px.histogram(
            x=position, nbins=40,
            title="Where the truth lands within the predicted P10–P90 band",
            labels={"x": "(y_true − P10) / (P90 − P10)"},
        )
        fig.add_vline(x=0, line_dash="dot", line_color="grey")
        fig.add_vline(x=1, line_dash="dot", line_color="grey")
        fig.add_vline(x=0.5, line_dash="dash", line_color="green",
                      annotation_text="P50")
        fig.update_layout(height=340)
        st.plotly_chart(fig, width="stretch")

    # Top errors — show district name alongside the NUTS-3 code.
    divider("Top 20 largest absolute errors")
    top_err = (preds.assign(abs_err=lambda d: d["residual"].abs())
                    .sort_values("abs_err", ascending=False)
                    .head(20)[["landkreis_name", "landkreis", "year",
                               "y_true", "y_pred", "residual"]]
                    .rename(columns={"landkreis_name": "District",
                                     "landkreis": "NUTS-3",
                                     "year": "Year"}))
    st.dataframe(top_err.style.format({"y_true": "{:.1f}", "y_pred": "{:.1f}",
                                       "residual": "{:+.1f}"}),
                 width="stretch", hide_index=True)

    # Download
    divider("Downloads")
    csv = preds.to_csv(index=False).encode()
    st.download_button("Download test predictions (CSV)", csv,
                       file_name=f"{model}_{crop}_predictions.csv",
                       mime="text/csv")

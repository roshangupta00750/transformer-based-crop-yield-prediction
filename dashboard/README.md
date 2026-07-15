# Crop Yield Benchmark — Dashboard

Streamlit dashboard wrapping the multi-model benchmark in
`Crop_Yield_Transformer_Preliminary.pptx`. Lightweight, single-process,
file-based (no DB).

**Run with the wrapper script** (recommended — pins OpenMP/MKL to avoid
macOS Intel segfaults from competing libomp.dylib copies):

```bash
conda activate agri
./dashboard/run.sh
```

Or invoke streamlit manually. Prefer the grouped-sidebar entrypoint —
it collapses the 19 pages into 7 named sections:

```bash
conda activate agri
streamlit run dashboard/nav.py
```

`streamlit run dashboard/Home.py` also works but shows the auto-discovered
flat 19-entry sidebar (same env vars are set at the top of `Home.py`,
`nav.py`, and `utils.py`, so any entrypoint is safe on macOS Intel).

The default URL is <http://localhost:8501>.

### macOS segfault note

On Intel Macs, torch, xgboost, lightning, and numpy/MKL each ship their
own copy of `libomp.dylib`. When two of those end up loaded in the same
process — which happens the moment a per-model page imports a checkpoint
— the OpenMP runtimes collide and the process segfaults. The fix is to
set `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, and
`KMP_DUPLICATE_LIB_OK=TRUE` *before* Python imports any of those libraries.
The dashboard does this in three places (`run.sh`, `Home.py`,
`utils.py`) so any reasonable entry point picks it up.

## Pages

| Page | What it does |
|---|---|
| **Home** | Project overview + leaderboard table (best R² per model) + horizon plot. |
| **Data** | Source provenance (DWD, MODIS, SoilGrids, Regionalstatistik, GISCO) + live previews of the interim parquets. |
| **Train** | Launch any of the six training scripts in a background subprocess. Live log tail; non-blocking. |
| **XGBoost** / **LSTM** / **DLinear** / **PatchTST** / **iTransformer** / **TFT** | Per-model page — hyperparams, in-season metrics, predictions vs. truth scatter, residuals, top errors, CSV download. |
| **Geographic** | Folium choropleth of actual / predicted / residual yield across the ~400 German Landkreise. Per-district year trend. |

## Architecture

```
dashboard/
├── Home.py                  # leaderboard + KPIs
├── utils.py                 # cached loaders, paths, subprocess wrapper
├── _model_template.py       # one render() function shared by all 6 model pages
└── pages/
    ├── 1_Data.py
    ├── 2_Train.py
    ├── 3..8_<model>.py      # 3-line wrappers per model
    └── 9_Geographic.py
```

- **Caching**: `@st.cache_data(ttl=300)` on unified parquet load, 60 s on
  metrics; pages stay snappy after the first hit.
- **Training jobs**: `subprocess.Popen` with `start_new_session=True` so they
  outlive any page rerun. Status polled via `os.kill(pid, 0)`; logs tailed
  from `/tmp/dashboard_<model>.log`.
- **No DB**: predictions are recomputed on demand by loading the saved
  `<model>.pt` + `<model>.pipeline.json` checkpoint pair. Computation is
  bounded (~5 s per page) and `@st.cache_data` re-uses results within a
  session.

## Production notes

This is intentionally not Kubernetes-grade. For a Praktikum demo it's
fine; if you want to harden it later:

- Front with `streamlit run --server.headless true` behind nginx + TLS.
- Replace `subprocess.Popen` with a real job queue (RQ + Redis or Celery)
  so training survives dashboard restarts.
- Persist run metadata to SQLite for richer history (currently we keep
  only the latest `<model>_meta.json`).

## Footprint

| Component | Disk | RAM (idle) |
|---|---|---|
| `dashboard/*.py` | <50 KB | — |
| Streamlit deps already in `agri` env | included | ~200 MB |
| Per-session memory | — | ~150–300 MB depending on data preview |

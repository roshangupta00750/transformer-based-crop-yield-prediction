# YieldFormer-DE dashboard

A one-screen Streamlit dashboard for the project's results: the district estimates of YieldFormer-DE (v18) for
2010-2025, the sealed 2026 forecast, and every comparison. It runs on the local machine only.

## Start

```bash
./dashboard/run.sh          # http://127.0.0.1:8502
```

`run.sh` first runs `scripts/63_dashboard_data.py`, which rebuilds the tables in `results/dashboard/` from the
evaluation results, and then starts Streamlit with `dashboard/main.py`. It also pins OpenMP and MKL to one thread,
which avoids crashes on macOS when several libraries bring their own copy of libomp. The first page load takes
15 to 20 seconds; after that a click takes about a second.

## What is on the screen

- Filters on the left: crop, years, region (North, East, South, West), state, district, the comparison
  (5-year average, long-term average or the XGBoost reference) and the model (YieldFormer-DE or one of its three
  members). They apply to every card and view, and Reset sets them back.
- Five cards: accuracy, seasons better than the comparison, share of estimates within 10 %, poor years detected,
  and the 2026 forecast. The small (i) on each card explains it.
- Five views:
  - Map: the 2026 forecast or any harvest from 2010 to 2025, accuracy, gain and crop area; click a district to
    open its card; click a region or a season to filter.
  - District look-up: forecast, normal yield, accuracy, crop area and all years of one district.
  - Table & download: all estimates of the selection as CSV, also in a German Excel format.
  - Forecasts: the sealed files and a first check against preliminary national yields.
  - Compare: simple averages and members, seasons and regions, published models, JRC MARS 2025, poor years and
    computing cost.
- Spotlight: click a card, table or chart to light it up and dim the rest of the page (Esc clears it). Switch it
  off in the left panel.

## Files

- `main.py`: entry point. It shows only `app.py`, so the older pages in `pages/` cannot be opened.
- `app.py`: the dashboard.
- `SPEC.md`: what the dashboard must contain (part A) and the contract of the old app (part B).
- `static/de_nuts3.json`: district shapes, written by `app.py` and loaded once by the browser.
- `run_old.sh`: the earlier multi-page dashboard, kept as a backup (port 8503).

## Tests

```bash
pytest tests/test_dashboard_app.py
```

The tests open every view with and without a district and crop selected and check that the required elements are
there.

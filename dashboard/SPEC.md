# Dashboard specification — YieldFormer-DE

## Part A — the dashboard (`dashboard/app.py`, entry `dashboard/main.py`, `dashboard/run.sh`, http://127.0.0.1:8502)

Decided by the user on 2026-10-15 after two reviews (dashboard designer 4/10, end user 5/10: "a report with one
dashboard page attached"): the dashboard is ONE screen, like a dashboard, not a report. Explanations live in the
written report (`Paper_Agriculture/thesis.pdf`). `tests/test_dashboard_app.py` guards this part. Second review round
the same day (designer 6/10, end user 7/10) led to the changes listed under "Rules".

- **Entry**: `main.py` runs `app.py` through `st.navigation(..., position="hidden")`, which turns off the automatic
  discovery of `dashboard/pages/`; the archive pages and the training console are not reachable from this app.
- **Header**: one line (name + two chips: estimates never saw their year; 2026 forecast sealed date and data end) and
  one visible "Limits" line (weather up to harvest; at 30 June no better than the 5-year average; large single-district
  losses underestimated).
- **Filter panel** (left sidebar, user 2026-10-16: stays in view while the page scrolls, so filters can be changed from anywhere; applies to every card and view; Reset all filters at its foot): Crop (fixed crop order) · Years (estimates) · State (German
  names) · District (type to search; also set by clicking the map or picking a district card crop) · Compare with
  (5-year average, Long-term average, XGBoost + memory as reference only; tooltip explains all three) · Model
  (YieldFormer-DE (v18) default; members shown as "-T · TFT member (v17)" etc.) · Reset (callback, explicit defaults).
- **Five cards** (user, 2026-10-16: number first, one short line, the rest in a short ⓘ tooltip; no yellow anywhere) (each with an ⓘ tooltip): Accuracy (hero, comparison below, "± pp" chip blue/red) · Better in N of M
  years · Within ±10% of official · Poor crop-years detected (≥10% below normal; rule "estimate ≥5% below normal"
  named on the card; false alarms with how many fall in years 5–10% below normal) · 2026 forecast (one crop: dt/ha, %
  vs normal, typical district range; all crops: area-weighted % vs normal + crops above normal). Below the cards one
  selection line with the district and estimate counts.
- **One view at a time**: **Map** (map year: 2026 forecast or any harvest 2010–2025 (user, 2026-10-16: "why just 2026?"); for a harvest year the layers are official yield vs normal, official dt/ha, estimate error [signed for one crop, size for all crops]; district shapes served once from `dashboard/static/` (enableStaticServing) instead of being re-sent on every click; layers 2026 vs normal [all crops = census-area-weighted mean of the crops' %],
  Accuracy [≥5 estimates, colours 2nd–98th percentile], Gain [diverging, ≥5 estimates]; with one crop also 2026 dt/ha
  and Crop area in grey; zooms to a chosen state; formatted hovers; horizontal colour bar; click → district card
  beside the map; click a year in the side chart → that year) · **District look-up** (picker; district card: forecast
  + 80% range, normal, accuracy, crop area, official vs estimate + 2026 forecast, all 16 years on one page, all crops in
  the district, CSV with NUTS code and model) · **Table & download** (user, 2026-10-31: official vs estimate side by side at every level — "Level" switch
  District / State / Region / Germany; above district level the district yields of the selection are averaged with
  census crop area as weight, with the number of districts; columns official, estimate, error, comparison, its error,
  gain in pp; search box; one-line note that an estimate is a backtest forecast for a season the model never saw;
  paged table; CSV of the table as shown at the chosen level, CSVs of the selection with NUTS code, model, issue date
  and the selection in the file name; German Excel option; data dictionary) · **Forecasts** (user, 2026-10-31: sealed
  line with the filtered count; forecast vs normal per crop, or the district distribution for one crop; beside it
  "Sealed forecast files": every file of `results/forecasts/FROZEN.sha256` with season, issue date, recorded
  fingerprint and a check recomputed on every run (✓ unchanged / ✗ changed / file missing), how sealing works in
  one caption, download of each sealed file unchanged and of FROZEN.sha256; full-width "District forecasts for <year>"
  with a Season switch over every sealed year (later years are district normals, said in a caption), search box and,
  once published, the official yield and error; "Revealed: sealed <year> forecast vs official yield" — says "Not
  revealed yet" until `results/dashboard/official_sealed_years.csv` (written by scripts/63 from
  data/interim/yield.parquet) has rows for that year, then turns on by itself with accuracy vs the normal-year
  forecast, districts closer, within ±10 % and a per-crop table; preliminary national check as a table with both
  aggregations, wording computed)
  · **Compare** ("Benchmark": simple averages and members on the same rows; published models with case results,
  "higher/lower is better" per measure, all counts computed from `papers.csv`; JRC MARS 2025 headed "not
  like-for-like", MARS grey diamonds; poor years with the three rules, false alarms and the −10 % / −5 % lines).
- **Colour and spotlight** (user, 2026-10-16: "too white"; "spotlight" so the audience focuses on one part): header in a blue band; every table and chart sits in its own lightly tinted panel (`panel(name, tint)` → `st.container(key="pnl-<tint>-<name>")`; tints sky, teal, sand, violet, green, slate are decoration only, data colours keep their meaning); summary cards carry a coloured top stripe. Spotlight (sidebar toggle, on by default): a click on a panel or summary card lights it with a blue ring and dims everything else in the main area; a click on empty space or Esc clears it; the sidebar is never dimmed.
- **Seasons and regions** (user, 2026-10-16; placed after three reviews: designer, jury presenter, statistics): Region filter (North, East, South, West = groups of federal states, ⓘ lists them) between Years and State; it narrows State and District, zooms the map, and the map keeps the national colour scale. Map view: "Which region does better?" under the map — both answers (user, 2026-10-16): a headline "Most accurate: X · Gains most over the comparison: Y", accuracy as dots and gain as zero-based bars side by side, regions in fixed N/E/S/W order (click a region = filter); the same headline above the region table in Compare, "Gain by crop" and "Gain by season" on the right. All per-group charts are zero-based gain bars in fixed order (crops, N/E/S/W, seasons in time order), blue/red, with a thin 90% range from resampling whole seasons; range line and label are separate layers so bar clicks work. Compare › Seasons & regions: no "winner" claim — within 0.5 pp of the season's best, largest shortfall, switch-to-last-season's-winner check, member-vs-ensemble strip (T/i/P, XGBoost grey), season table with "(tie)" for leads under 0.5 pp, region table with accuracy, gain, 90% range, seasons better, gain without 2018, gain with every crop equal, TFT minus ensemble. Forecasts per crop: area-weighted bar (as the card) with the district spread; MARS: signed errors with an ours↔MARS connector.
- **Computing cost** (user, 2026-10-16; Compare › Computing cost): GPU hours of the final model, size of the three networks, GPU types, an energy upper bound (GPU hours × maximum board power), a per-network table (parameters, file size, runs, GPU per run, median training and fine-tuning time, runs stopped by the 120-min limit, GPU hours; XGBoost as a labelled reference) and GPU hours per network (training + fine-tuning). Data: `results/dashboard/compute_jobs.csv` and `compute_models.csv`, written by `scripts/79_compute_cost.py` from the SLURM job logs and checkpoints, copied locally.
- **Season lines and method box (user, 2026-10-31)**, at the foot of the Map view, all following the filters:
  "Forecast vs official, season by season" (lines in % vs the district normal so crops can be pooled: official black,
  backtest forecast blue, comparison grey, sealed forecast as a diamond with its 80% range; click a season = filter),
  "Accuracy by season" (model, comparison and district normal as lines), "Accuracy by forecast date" (30 June and
  31 July from the in-season runs v20, harvest from the final model, 5-year average as a line; only the Crop filter
  applies; the gaps are computed into the caption) and a "Data & method" box (question, data sources, model, test,
  links to code, model and data).
- **Wording after two blind reviews (user, 2026-10-31)**: 2010–2025 are "backtest forecasts" (tables: season as
  "2025 (harvest)", value column "Forecast"), 2026 onwards "sealed forecast"; never "estimate" on screen. A reading line
  under the cards says the first four cards are a plain mean over the backtest forecasts, that the design was chosen
  while these results were visible (changes kept only by the written rule on the decision window), that the 2026 card
  is area-weighted and that the sealed forecast is not scored yet. Accuracy card: "not a hit rate" in its tooltip and
  the 90% range of the gain from resampling whole seasons. One name for the long-term reference: "District normal"
  (data column "Long-term average" unchanged). "Poor crop-years" card. Sealed files labelled by role (final model,
  pre-registered test v1, earlier main model, district normal); the Revealed panel scores both the pre-registered
  27 Sep file and the final model's file against the district normal. The preliminary national check says the
  figures were public (August 2026) before every 2026 file was sealed.
- **Rules**: "backtest forecast" = 2010–2025, "sealed forecast" = 2026 onwards; blue = our model / above normal / better,
  red = below / worse, grey = comparison or neutral quantity, black = official; no heatmaps; tables are plain HTML
  (no internal scrolling, no header-click sorting, paging instead); crops always in the fixed order, never ranked; one
  decimal, "+"/"−" signs, "–" for missing; every count computed from the data (none typed in); no training controls
  or archive pages; localhost only.

## Part B — backup: the earlier multi-page app (`dashboard/nav.py`, `dashboard/run_old.sh`, port 8503)

Kept unchanged as a fallback. The contract below applies to it. Every page and every element below must exist.
`tests/test_dashboard_spec.py` runs each page headlessly and fails if a page errors or a
required element (by its heading text) disappears. Change this file first, then the page,
then the test — never remove an element silently.

Audience: everyone. Each page opens with a plain-language answer, then numbers, then charts,
then detail. Every number says which model it belongs to. All data comes from
`results/dashboard/` (built by `scripts/63_dashboard_data.py`) or `results/metrics/`.

Model name: **YieldFormer-DE** = version v18 (equal-weight average of iTransformer, TFT and PatchTST + district error memory; adopted 2026-10-11). Naming (user, 2026-10-12): only the ensemble is called **YieldFormer-DE**. Its members, each a single network + the district error memory, are **YieldFormer-DE-i** (iTransformer, v15; main model until 2026-10-11), **YieldFormer-DE-T** (TFT, v17) and **YieldFormer-DE-P** (PatchTST, v17). Never call a single network "YieldFormer-DE".

## Dashboard (default page)

One interactive screen, little text (explanations in hover tooltips), Power-BI style: **no scrolling tables** —
data is shown as stat tiles, charts and paged drill-down tables that fit the screen. Global sidebar filters —
model, comparison model, crops, years, states, district (also set by clicking the map) — drive
every element. **Crops are not ranked**: the goal is to show how well the method works for every
crop, so crops always appear in the fixed crop order (states A–Z, years in time order) and no view
sorts or labels crops as weakest/strongest. Chart form follows the question (no heatmaps of numbers, no colour-coded tables,
no arrows on neutral reference values). Required: "Filters"; six stat tiles "Accuracy", "Typical
error", "Within ±10%", "National accuracy", "Bad years warned", "Estimates" (value + plain reference
line; colour only on a real gain/loss); tabs —
- "Overview": clickable district map (gain / accuracy / mean yield); "Accuracy by year" (or "Real
  yield vs estimate" for one crop) as lines; "Accuracy by crop" as dumbbell (model dot vs comparison
  tick); "How close the estimates are" (within ±5/10/20%, grouped bars).
- "Drill-down table" (second tab, the working tool): breadcrumb All crops › crop › state › district;
  click a row to go one level deeper (crop → state → district → individual years); per level a
  table with plain column names that fits the width (no sideways scroll), 15 rows per page with
  ◀ ▶ page buttons (no scrolling inside the table), search box, "sort by" (standard order: crops in
  the fixed crop order, states/districts A–Z, years in time order; or most estimates), columns incl.
  "Years improved" (share of years the model beat the comparison), progress bar for "Within ±10%",
  CSV download of the current level. Sidebar filters apply.
- "Accuracy": "Gain over the comparison by year" (diverging bars); "Accuracy by state" (dumbbell).
- "Errors": over/under-estimate by year (diverging bars); error vs crop share (grouped bars);
  "Data-quality check: estimates far from the official figure" (10 bars, framed as data issues).
- "Model comparison", two views (radio):
  1. "Published models" (default): tiles "vs AgroForecast-DE / vs CropFusionNet / vs Srivastava 2022 /
     All published models" with comparisons won; per paper "YieldFormer-DE vs <paper>": grouped bars per
     measure (published model orange, **YieldFormer-DE blue, outlined = highlighted**, historical average
     grey) on the paper's own split + a result table (✔ ours better / ✖ paper better). YieldFormer-DE*
     = v12 network where the paper's split forbids last year's yields (AgroForecast-DE, CropFusionNet);
     v15 on Srivastava's single-year tests. Crop filter applies; year/state filters do not.
  2. "Our versions & simple averages": all models on the selected estimates (bars, ★ YieldFormer-DE
     highlighted) + "Scorecard" (plain table, best value per column in bold). Includes the v16
     baseline "Trend + memory" (no network; same rows), also selectable in the sidebar. **Model family
     rule (user, 2026-10-10): the candidates are transformer models; simple baselines (averages, trend +
     memory) and ONE non-transformer reference, "XGBoost + memory (reference)" (v16 GE16, same inputs and rows), appear
     grey and labelled "reference, not a candidate". It can be chosen as the comparison, never as the model.
     No other tree / gradient-boosting variant is shown.**
- "Weather": crop + signal selectors; "Which weather matters for <crop>" (diverging bars of r);
  each-year scatter.
- "Crop areas": 4 stat tiles, area map, share map, "Top 12 districts" (bars).
- "Trends": "Real vs estimate by crop" (small multiples, one line pair per crop); three CSV downloads.

## Report (explanatory pages)

| # | Page | Required elements (heading text) |
|---|---|---|
| 1 | Overview | model name "YieldFormer-DE"; 4 headline tiles; "Which model is which"; "Why the counts differ"; "How it works"; "Germany's harvest: real vs estimate"; "Top insights" |
| 2 | Key insights | at least 10 numbered insights, each with a number and a chart: "1." … "10." |
| 3 | Predictions vs actual | "Year by year: Germany"; "Every estimate"; filters (crop, year, state, district); CSV download; "Year × crop table" |
| 4 | Where crops are grown | "Crop area by district" map; "Share of farmland"; "Top districts"; "Area by state"; "Average yield by district" map |
| 5 | Accuracy | tabs "By crop", "By year", "Map", "Forecast vs real", "One district" |
| 6 | Error analysis | "Error distribution"; "Error by state"; "Error vs crop area"; "Over- or under-estimate by year"; "Where the error comes from"; "Errors repeat"; "Cautious estimates"; "Largest misses" |
| 7 | Weather and yield | "Which weather matters for which crop" (correlation heatmap); "One crop, one weather signal" (scatter); "Drought years in the weather" |
| 8 | Bad years | warning-rule table "three rules side by side"; "Every bad harvest"; "All years at a glance"; "Limits, stated plainly" |
| 9 | Compared with papers | "What this means"; per-paper charts; "Every version, every comparison"; "How the comparison changed"; "JRC MARS, Germany 2025" (operational benchmark, error per crop vs official, census-area weighted) |
| 10 | Model journey | "All versions on the same forecasts"; "How sure is the gain" (final model v18: 90% ranges vs v15 and the 5-year average, data-quality flags, multiple testing); "Is the transformer needed?" (v16 ablation incl. the v18 point, both test periods, XGBoost reference); "Every version"; "What we learned" |
| 11 | Forecasts | "YieldFormer-DE forecasts" (2026 and 2027 frozen files with fingerprint; "Germany: forecast vs normal" bars per year; "Per district" map + table for the chosen year); "Earlier forecast from v1" with the v1 warning ("not the final model") and its own "Germany: 2026 forecast vs normal"; "How the forecasts did in 2024 and 2025" (backtests: each model never saw the year; district accuracy per model + Germany real vs forecast per crop, labelled as backtests, not frozen forecasts); "2026: first check against preliminary official yields" (census-area weighted, not the registered score) |

## Background

| # | Page | Required elements |
|---|---|---|
| 12 | How it works | "The question"; "The data"; "The model"; "How we test it honestly"; "Glossary" |
| 13 | Data | "Yield trend per crop"; "One district, one season"; "Data coverage" |

## Archive
The earlier six-model benchmark pages ("6-model test", "6-model map", "6-model results", "Training tool", "Evidence notes"), kept unchanged; no month in their labels.

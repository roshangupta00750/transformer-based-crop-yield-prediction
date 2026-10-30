# Pre-registration v14: DWD HYRAS 1 km gridded weather

**Written:** 6 October 2026, before the HYRAS data is downloaded or aggregated and
before any v14 model is trained. Scoring and the model-change rule as v6 §4-§6.
Base: v12 arm Q12 (v13 C13 was not adopted).

## 1. Why

Weather source is the largest single effect measured so far: replacing DWD station
weather by ERA5-Land everywhere cost 0.042 macro skill on the hindcast (v11, P 0.99).
Q12 uses DWD stations inside the district polygon where present and ERA5-Land
(9 km, reanalysis) elsewhere, so about 80 districts and every station gap still rely
on ERA5-Land, and a district's value comes from one or a few stations that may not
lie on its farmland. HYRAS-DE (DWD, v6-1) interpolates all DWD stations (precipitation
about 2,000-2,500 per day) to a 1 km grid with terrain-aware methods: one consistent,
observation-based source for every district and day.

## 2. Change (arm H14)

1. **Source:** HYRAS-DE v6-1, 1 km, daily, 2000-2025: `tasmax`, `tasmin` (°C), `pr`
   (mm, DWD climate day), `hurs` (%, converted to a fraction).
2. **Averaging:** district mean over cropland, weight of a 1 km cell = (area of the
   cell inside the district) x (share of the cell that is CORINE 2018 class 211),
   the same cropland definition as the satellite and ERA5-Land inputs (H3). A
   district with no cropland weight uses the plain area mean.
3. **Use:** these four replace `temperature_air_max_2m`, `temperature_air_min_2m`,
   `precipitation_height` and `humidity` for all districts and days; every derived
   input (water balance, heat/frost/heavy-rain flags, crop-stage totals) is
   recomputed from them. `wind_speed` stays ERA5-Land (HYRAS has no wind);
   radiation stays excluded.
4. Everything else as Q12: corrected yields, reliable training series, arm-T
   configuration, per-crop fine-tuning, rolling folds 2010-2025, seeds 42/43/44.

## 3. Decision

H14 replaces Q12 if, on the hindcast 2010-2015 (all corrected-panel rows), v6 §6
holds: macro MSE skill vs y_hist higher with year-block bootstrap P(Δ > 0) >= 0.95,
pooled MAPE lower, no crop worse in every year. Reported: 2016-2025, 2019-2025,
per crop, districts with vs without an own DWD station, and the three papers'
protocols. Sanity check before training (not a decision): correlation of HYRAS
with the DWD station series where both exist.

## Change log

| Date | Change | Reason |
|---|---|---|
| 2026-10-06 | Plan written | - |
| 2026-10-06 | **Data built, before any v14 model is trained.** `scripts/61_hyras_weather.py`: 104 yearly files (tasmax/tasmin/pr/hurs 2000-2025), 3,808,297 district-days, 401 districts, no missing days; DE273 and DE27A have no CORINE 211 cells and use the area mean. Sanity check vs DWD station-in-district series (v4, ~2.8 M district-days): tmax r 0.995 (bias +0.07 °C), tmin r 0.991 (−0.03), daily precipitation r 0.940 (−0.05 mm), monthly r 0.949, humidity r 0.972. ERA5-Land on the same rows: tmax r 0.987 (−0.76 °C), daily precipitation r 0.682 (+0.31 mm), monthly r 0.859. 327 of 401 districts have their own station tmax on >= 50% of days. Aggregation ran on the Mac (julia2 CPU queue ~18 h); dataset build on an L40 node with QoS normal. | Data description; plan unchanged. |
| 2026-10-07 | **Decision result (hindcast 2010-2015; train 3458951, fine-tune 3458952 tasks 1-18, eval 3464844 `--hindcast-only`; L40).** 17,718 rows: accuracy H14 88.22% vs Q12 88.32% (B5 87.65, y_hist 86.89), macro skill Δ −0.016 (90% CI −0.041..+0.014, P(Δ>0) 0.24), no crop worse in every year → **v6 §6 not met; Q12 stays.** Per crop (pp): winter barley +0.35, rapeseed +0.16, sugar beet +0.03, summer barley +0.03, wheat −0.04, oats −0.21, rye −0.26, triticale −0.36, potatoes −0.37, silage maize −0.38. Districts with own DWD station 88.15 vs 88.27 (271 districts); without 88.44 vs 88.47 (89 districts), so the gridded weather does not help even where Q12 relies on ERA5-Land. Evaluated before the last 2016 fine-tuning task finished (preempted 4 times); the decision window is complete, 2016-2025 and the papers' protocols are reported when done. | Registered outcome. |
| 2026-10-10 | **Papers' protocols (late; stalled comparison jobs cancelled after all 15 training tasks had finished; eval job 3476260, scored with `scripts/40`).** H14 wins 11 of 18 comparisons (v12 Q12: 11/18). Not decisive; v14 was not adopted. | Registered outcome (reporting). |

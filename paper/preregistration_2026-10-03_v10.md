# Pre-registration v10: corrected data foundation

**Written:** 3 October 2026, before the v10 dataset is built and before any v10
model is trained. Scoring, references and the model-change rule as v6 §4-§6.

## 1. Why (audits of 2026-10-03)

Three independent audits (code, data, literature) found that every result so far
(v1-v9) was trained and scored on a damaged panel:

1. **Yield-to-district matching by name.** `destatis_yield._normalize_name` strips
   "Kreisfreie Stadt"/"Landkreis", so 18 city/Landkreis pairs collide: yields are
   averaged or attached to the wrong NUTS-3 code (e.g. Landkreis Hof DE249 has no
   yields; its values sit on Hof city DE244), and 9 districts are dropped
   (Lübeck, Friesland, Erfurt, Gera, Weimar, Eisenach, Saarbrücken, Aachen, ...).
2. **Weather only from DWD stations inside a district polygon:** 64 districts with
   yields get no weather and fall out (Vechta, Gütersloh, Wesel, Neuss, ...); 13
   more have mostly missing temperature that is mean-imputed.
3. **Trend baseline** referenced to the mean year over all crops instead of the
   district's own training years (sugar beet offsets up to ~79 dt/ha).
4. **MODIS values** stamped at the start of their 8/16-day composite period.

Fixing 1-2 raises the labelled panel from 282 to ~366 districts (+32% district-
crop-years).

## 2. Data v10 (`dataset_v5`)

1. **Yields:** Destatis 41241-01-03-4 matched by AGS code: state + city/Landkreis
   type + name, plus the official successor table for the 2007 (Saxony-Anhalt),
   2008 (Saxony), 2009 (Aachen), 2011 (Mecklenburg-Vorpommern) and 2016
   (Göttingen) reforms. A current-code value always wins; a reformed district's
   year without a current-code value is filled only from a single predecessor
   that maps to it alone (no unweighted averaging of several predecessors).
2. **Weather:** ERA5-Land daily, averaged over CORINE 2018 cropland per district,
   for all districts: tmax, tmin, precipitation, relative humidity (from mean and
   dew-point temperature), plus new wind speed. It replaces the DWD station
   series under the same column names, so every derived input (climatic water
   balance, heat/heavy-rain/frost flags, crop-stage totals) is recomputed from it.
3. **Trend:** slope x (year − the district-crop's own mean training year).
4. **MODIS:** NDVI/EVI stamped at period start + 16 days, LAI/fPAR + 8 days.
5. Everything else as `dataset_v4` (2000-2025, phenology v3, ERA5-Land drivers,
   cropland satellite, soil, terrain).

## 3. Model and decision

Arm D10 = v4 arm F's configuration (arm T + per-crop fine-tuning) on `dataset_v5`.
Rolling folds 2010-2025, seeds 42/43/44, L40; the three papers' protocols as v4 §5.

- **Decision:** D10 replaces v4 F if, on the hindcast 2010-2015 and on the rows
  both panels share, v6 §6 holds (macro skill vs y_hist higher with
  P(Δ > 0) >= 0.95, MAPE lower, no crop worse in every year). Because the panel
  itself changes, results on the full 366-district panel are reported alongside.
- **Reported:** 2016-2025 (development), the 18-row paper table, per-crop accuracy.

## Change log

| Date | Change | Reason |
|---|---|---|
| 2026-10-03 | Plan written | - |
| 2026-10-04 | **Results (build 3435589; rolling 3435592/93, eval 3435594; papers 3435595/96; L40).** dataset_v5: 367 districts, 69,418 district-crop-years with yields, training frame 67,589 crop-years (v4: 50,741); ERA5-Land vs DWD where both exist: tmax r 0.987 (bias −0.72 °C), daily precipitation r 0.686, mean 2.41 vs 2.12 mm/d. Decision rows (same district-crop-years, same yields), hindcast 2010-2015: accuracy D10 87.67% vs v4 F 88.27%, macro skill Δ −0.077 (90% CI −0.113..−0.030, P(Δ>0) = 0.003), triticale worse in every year → **not adopted; D10 is worse.** 2016-2025: Δ 0.000; 2019-2025: Δ +0.008. Full corrected panel 2010-2015: D10 87.72% vs B5 87.65%, y_hist 86.89%. Papers: 11/18 (v4 F 12/18); AgroForecast-DE wheat R² 0.390 (v4 0.250); Srivastava rows now on 279-286 districts (theirs 271). Four changes were made at once (yields, weather source, trend reference, MODIS stamp), so the cause of the hindcast loss is not identified; the replacement of station weather by ERA5-Land is the leading suspect (daily precipitation r 0.69). | Registered outcome. |

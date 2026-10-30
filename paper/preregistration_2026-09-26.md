# Pre-registered experiment plan: phenology-aware transformer yield prediction

**Project:** Transformer-based Crop Yield Prediction using Multivariate Time Series Datasets
**Written:** 26 September 2026, before any of the runs below were started.
**Status:** fixed. Any later change to features, thresholds, arms or success
criteria must be recorded in the change log at the end, with the date and the
reason, and results must be reported against the plan as registered here.

## 1. Why this is written down first

We already know from the rolling evaluation (2016-2025) which years are hard:
2016 and 2024 (3-5 of 10 crops beat the historical average), while the 2018
drought is caught well. Choosing features after looking at those years would
risk fitting the answer. The additions below are taken from the published
literature and fixed here before running.

## 2. Prediction unit and target (unchanged)

- One sample = **Landkreis x crop x harvest year** (282 NUTS-3 districts,
  10 crops, harvest years 2003-2025).
- Target = that year's yield in dt/ha, modelled as the deviation from the
  district's historical mean for that crop, computed on training years only
  (plus a train-only linear trend where statistically clear, p < 0.01).
- Prediction = historical mean + predicted deviation.

## 3. Evaluation protocol (identical for every arm)

- Rolling origin, one test year at a time, Y = 2016, 2017, ..., 2025:
  train on harvest years 2003..Y-2 (see §4.4 for 2003), early-stop on Y-1,
  test on Y. Normalisation and historical means fitted on training years only.
- Model: iTransformer (variate attention), existing hyperparameters
  (`configs/models/itransformer.yaml`), max 120 min per run.
- Seeds 42, 43, 44; the reported model per arm is the 3-seed average
  prediction. Individual seeds are also reported.
- GPU: NVIDIA H100 80GB on julia2 for every run.
- **Model size held constant:** every sample is resampled to 220 input
  positions in every arm, so longer windows are compressed, not the model
  enlarged.
- **Same samples in every arm:** only crop-years present in all arms are
  used (see §4.4).

## 4. Arms (what changes)

| Arm | Season window | Added inputs |
|---|---|---|
| **A** (control) | winter crops 1 Jan -> harvest; spring crops sowing -> harvest | current inputs incl. ERA5-Land/fPAR drivers |
| **B** | **full crop year**: sowing -> harvest for all crops (winter crops include the previous autumn) | as A |
| **C** | as B | B + wet-stress and winter indicators (§4.2) |
| **D** | as B | C + cropland-only satellite averaging (§4.3) |

Spring crops have identical windows in A and B.

### 4.1 Crop-year dates (national typical dates, same for every district)

Day-of-year from `configs/data.yaml`; calendar dates for a non-leap year.
Observed per-district phenology dates are deliberately **not** used (they
reveal end-of-season information).

| Crop | Sowing | Harvest | Arm A window | Arms B-D window |
|---|---|---|---|---|
| winter wheat | 7 Oct (d280, previous year) | 8 Aug (d220) | 1 Jan-8 Aug, 220 d | 7 Oct-8 Aug, 306 d |
| winter barley | 27 Sep (d270, prev.) | 14 Jul (d195) | 195 d | 291 d |
| rye | 27 Sep (d270, prev.) | 29 Jul (d210) | 210 d | 306 d |
| triticale | 2 Oct (d275, prev.) | 3 Aug (d215) | 215 d | 306 d |
| winter rapeseed | 23 Aug (d235, prev.) | 19 Jul (d200) | 200 d | 331 d |
| summer barley | 21 Mar (d80) | 3 Aug (d215) | 136 d | same |
| oats | 31 Mar (d90) | 23 Aug (d235) | 146 d | same |
| silage maize | 30 Apr (d120) | 27 Sep (d270) | 151 d | same |
| potatoes | 15 Apr (d105) | 27 Sep (d270) | 166 d | same |
| sugar beet | 5 Apr (d95) | 27 Oct (d300) | 206 d | same |

The previous-autumn part of a winter-crop sequence is built from the same
district's daily weather, satellite and driver data of the previous calendar
year. Yield values from the previous year are never used as inputs.

### 4.2 Arm C: added daily indicators (definitions fixed)

Motivated by: excess spring water as a main loss driver in Germany (Peichl
et al. 2021, HESS 25:6523), water excess vs drought for wheat (Zampieri et
al. 2017, ERL), the compound warm-autumn / wet-spring failure of 2016
(Ben-Ari et al. 2018, Nat. Commun. 9:1627), winter conditions for winter crops.

| Name | Definition | Source |
|---|---|---|
| `heavyrain` | 1 if daily precipitation >= 20 mm (ETCCDI R20mm), else 0 | DWD precipitation |
| `wetsoil` | 1 if top-layer soil moisture (swvl1) >= the district's 90th percentile of swvl1 over 2003-2015, else 0 | ERA5-Land |
| `soiltemp` | daily mean soil temperature, 0-7 cm (`soil_temperature_level_1`), deg C | ERA5-Land |
| `snowdepth` | daily snow depth (`snow_depth`), m | ERA5-Land |
| `barefrost` | 1 if tmin <= -15 C and snow depth < 0.05 m (frost without snow cover), else 0 | DWD tmin + ERA5-Land |
| `transp_ratio` | actual / potential evaporation (`total_evaporation_sum` / `potential_evaporation_sum`), clipped to [0, 1.5] (see change log) | ERA5-Land |

The 2003-2015 reference period for `wetsoil` precedes every test year
(first test year 2016), so the threshold never uses a test year.

### 4.3 Arm D: cropland-only averaging of satellite inputs

- Mask: CORINE Land Cover 2018, class 211 (non-irrigated arable land), GEE
  `COPERNICUS/CORINE/V20/100m`, one static mask for all years.
- Applied to MODIS NDVI, EVI (MOD13Q1), LAI and fPAR (MOD15A2H): district
  means over masked pixels only. Same MODIS products, scale factors and gap
  handling as the current pipeline, so the mask is the only difference.
- Districts with < 1% arable pixels keep the whole-district mean (flagged).
- Weather and ERA5-Land (about 9 km cells) are not masked.
- Motivation: crop masks are standard (CY-Bench, JRC MARS, Peichl 2021,
  Baatz 2025, CropFusionNet); a controlled study reports about 0.3 t/ha lower
  error with crop-group masks (Comput. Electron. Agric. 2023,
  doi:10.1016/j.compag.2023.107633).

### 4.4 Sample set

Crop-year 2003 winter crops have no autumn-2002 data in the dataset, and a
few district-years lack the previous year's data (94.1% of winter
crop-years are complete). These crop-years are excluded from **all arms**,
including A, so every arm is trained and scored on the same samples.

## 5. Metrics (per crop, per harvest year; never pooled only)

For each crop-year: R², RMSE (dt/ha), MAE (dt/ha), Pearson r, and
**skill = 1 - SSE_model / SSE_historical-mean** (0 = no better than the
district historical mean). Also: whether the predicted direction of the
year's national deviation matches the observed one. Aggregates over years
are reported in addition to, never instead of, per-year results. Winter wheat
and winter barley are reported separately.

## 6. Hypotheses and success criteria (fixed)

Comparisons use the 3-seed average prediction, per crop-year skill.
One-sided sign tests, alpha = 0.05.

| # | Question | Comparison | Crop-years | Success if |
|---|---|---|---|---|
| H1 | Does the full crop year help winter crops? | B vs A | 5 winter crops x 10 years = 50 | B better in **>= 32 of 50** |
| H2 | Do wet-stress and winter indicators help? | C vs B | 10 crops x 10 years = 100 | C better in **>= 59 of 100** |
| H3 | Does cropland-only satellite averaging help? | D vs C | 100 | D better in **>= 59 of 100** |

Additional conditions for calling a hypothesis confirmed:
1. The direction holds in each of the three seeds individually
   (majority of crop-years better in every seed).
2. No crop in the comparison is worse in >= 7 of its 10 years.

Reported descriptively, not as success criteria: results for 2016 and 2024,
the 2018 drought, winter wheat and winter barley separately, and the count of
crop-years beating the historical mean per arm.

Note on metrics: for the same crop-year, the two arms share the samples and
the historical mean, so "higher skill" is equivalent to "lower RMSE" and
"higher R²". The success criteria are therefore the same whichever of these
is used; R², RMSE and MAE are all reported per year.

A negative or mixed result will be reported as such.

### 6.1 Forecast horizons (30/60/90/120 days before harvest), arms A and B

- The models are trained on full windows. For each horizon h, the test input
  keeps its full-season timeline, and every day after (harvest - h) is set to
  missing: normalised value 0 (the training mean) with its missing-indicator
  flags set where they exist. The sequence is **not** shortened, because the
  iTransformer resamples each input to 220 positions and a shortened season
  would be stretched, shifting every day to the wrong growth stage.
- Same metrics and per-year reporting as §5. Labelled as a first look:
  proper per-horizon retraining is a planned follow-up.

### 6.2 Uncertainty (all arms)

- Each prediction gets an 80% interval: prediction +/- the 10th/90th
  percentiles of that crop's errors in **earlier test years only** (so it
  starts from 2018, with two years of error history), per arm.
- Reported per crop-year: interval coverage (share of districts whose actual
  yield falls inside; 80% = well calibrated) and interval width (dt/ha).
- Also reported: the spread across the three seeds per crop-year.

## 7. Prospective 2026 forecast

- After the experiment, the arm with the most crop-years beating the
  historical mean over 2016-2025 (ties: higher median skill) is selected by
  this rule, with no other choice involved.
- It is trained on harvest years up to 2024, early-stopped on 2025, three
  seeds, and predicts **every district x crop for harvest year 2026**.
- The predictions are frozen before official 2026 district yields are
  published: saved as a file with its SHA-256 hash recorded here, committed,
  and sent to the supervisor with a timestamp.
- They are scored with the same metrics once Destatis publishes the 2026
  district yields.

## 8. Not done in this experiment (to keep it controlled)

- No hyperparameter changes between arms.
- No observed per-district phenology dates, no lagged yields as inputs.
- Antecedent pre-sowing periods, separate winter/spring models, WOFOST
  crop-model outputs, SPEI, AgERA5 and MODIS land surface temperature are
  follow-up experiments, not part of this plan.
- 30/60/90/120-day horizons with per-horizon retraining are a follow-up;
  this plan includes only the no-retraining first look in §6.1.

## Results as registered (27 September 2026)

120/120 runs (julia2, H100), evaluated by `scripts/34_eval_prereg.py`;
decisions in `results/metrics/prereg_hypotheses.json`. Reported against the
criteria above without change.

| # | Better | Threshold | Seeds (42/43/44) | Worst crop (years worse) | Verdict |
|---|---|---|---|---|---|
| H1 (B vs A, winter) | 25 / 50 | 32 | 26 / 23 / 27 | winter wheat 8/10 | **not confirmed** |
| H2 (C vs B) | 44 / 100 | 59 | 42 / 51 / 49 | oats, summer barley 8/10 | **not confirmed** |
| H3 (D vs C) | 67 / 100 | 59 | 65 / 58 / 55 | 4/10 | **confirmed** |

Crop-years beating the historical mean (of 100): A 62, B 55, C 55, D 62
(median skill A +0.057, B +0.030, C +0.017, D +0.061). The §7 rule selects
**arm D** for the prospective 2026 forecast.

Open confound for H1 (to be tested as exploratory, not confirmatory): the
crop-year window compresses 291-331 days into 220 positions, reducing
resolution in May-July relative to arm A.

### Frozen forecasts (§7)

Issued 27 September 2026, before any official 2026 district yields exist.
Arm D, 3 seeds (train <= 2024, early stop 2025); data end 19 Sep 2026
(silage maize and potatoes forecast 8 days before the registered harvest date,
sugar beet 38 days before; all other crops after harvest).

| File | SHA-256 |
|---|---|
| `results/forecasts/forecast_2026_issued_2026-09-27.csv` | `49234718f55fef17e460e499239920ec38b0f8162bee4b0dccef7bac7290251e` |
| `results/forecasts/forecast_2027_baseline_issued_2026-09-27.csv` | `fca9cf9fb9219850cce53feecf3d45504b2b6553e32ac6d5c9a03560101cf944` |

Checkpoint SHA-256: s42 `404b5d62...f500`, s43 `fe9fe7f3...64cd57`, s44 `9da7eaff...c00`
(full hashes: `shasum -a 256 results/checkpoints/itransformer_preD_fc2026_s4?.pt`).

## Change log

| Date | Change | Reason |
|---|---|---|
| 2026-09-26 | Plan written | - |
| 2026-09-26 | Implementation clarifications (no change of intent): (1) `transp_ratio` = 1.0 when potential evaporation < 0.1 mm/day (ratio undefined; no demand = no water stress). (2) §4.4 sample rule: a winter crop-year is kept only if BOTH its 1 Jan -> harvest part and its previous autumn are each >= 90% complete, applied identically in all arms (a total-only check let a complete autumn hide a winter gap and gave 23 extra samples in the crop-year arms). (3) Spring-crop and winter-crop previous-autumn rows come from the same crop's previous-year rows, else another crop's rows for the same district and date. | Found while implementing, before any run. |
| 2026-09-26 | `transp_ratio` redefined from transpiration/PET to actual/potential evaporation (AET/PET), same clipping and near-zero-demand rule | The ERA5-Land transpiration band in GEE `ECMWF/ERA5_LAND/DAILY_AGGR` is 0 on every day (checked: DE265, 2018, all months), so the registered variable would have been constant noise. AET/PET is the standard water-stress ratio measuring the same thing. Changed before any run. |
| 2026-09-26 | Added §6.1 horizons (no-retraining, full timeline with missing tail) and §6.2 uncertainty; clarified metric equivalence | Supervisor's requests (horizons "if already supported"; paper arc ending in uncertainty/generalisation). Added before any run. |
| 2026-09-27 | §7 implementation, fixed before any 2026 prediction exists: (1) 2026 inputs are built by the same code on a separate file (`scripts/35`), refused unless its 2025 rows reproduce the training dataset. (2) Crops harvested after the last day with data (silage maize, potatoes, sugar beet) keep their full timeline with the missing tail, as in §6.1; `lead_days` is recorded. (3) 80% interval = arm D ensemble backtest errors 2016-2025 per crop (§6.2 rule). (4) Output frozen as CSV + SHA-256 (`results/forecasts/FROZEN.sha256`), never overwritten. | Needed to run §7 at all. |
| 2026-09-27 | Added: 2027 forecast and a 2023-2027 scoreboard (`scripts/37`). 2023-2025 are labelled **backtests** (made after the yields were public, arm chosen on 2016-2025), never prospective. 2027 is issued in stages: while the lead is beyond the range where the backtests show the model beating the historical baseline, the registered 2027 forecast is the baseline (train-only district mean + trend); the arm D model forecast is issued once the 2027 season data exist. 2027 uses the 2026 models (train <= 2024, val 2025) unless 2026 district yields are published before the 2027 harvest, in which case it is retrained train <= 2025, val 2026. | User request (forecasts 2023-2027, scored against actual yields as they are published). Added before any 2026 or 2027 prediction was made. |
| 2026-09-27 | Known build quirk, not changed: `build_unified_dataset` forward-fills weather gaps with `limit=7` **rows**, and each date has one row per crop, so the fill reaches fewer days where more crops are present. The 2026 input rebuild therefore leaves a few gaps that the training build had filled (0.002% of 2025 weather cells; largest effect on a 2025 prediction 2.3 dt/ha = 0.25%, sugar beet DEB3H). Fixing it would change the registered training data, so it is left as is and applies equally to all arms. | Found by the end-to-end check: same model, 2025 predicted from the training dataset vs the new build (1,520 samples, mean difference 0.005 dt/ha). |
| 2026-09-27 | **Added before any run: replication of other papers' protocols (comparison only, no effect on H1-H3).** Arm D, unchanged config and hyperparameters, 3 seeds, ensemble mean, one run per protocol, results reported as they come out (no re-runs, no tuning on test years). (1) **AgroForecast-DE**: train <= 2019, validate 2020-22, test 2023-24; scored on the district-years matched to their published predictions (`scripts/38`), plus all 10 crops. (2) **CropFusionNet**: train <= 2018, validate 2019-21, test 2022-24; reported (a) on the clean test years 2022-24 and (b) pooled over 2019-24 as their code does (validation years included, same caveat as theirs), wheat, silage maize, winter barley. (3) **Srivastava 2022**: for Y = 2017, 2018, 2019 train <= Y-1 with a random (not year-blocked) 15% of training crop-years held out for early stopping, as their random CV; winter wheat. Their observed phenology-date inputs are NOT added (our leakage policy). Metrics as in the comparison table (R², RMSE, MAE, error %) and skill vs the historical average. | User request: settle the comparison on equal protocols. |
| 2026-09-27 | Replication runs executed on L40 GPUs (`standard` partition) instead of H100: the H100 partition was full, and the free L40S slots would have preempted another user's running jobs. **Correction (same day, found by audit):** the `standard` partition mixes L40 and L40S cards, so 8 runs ran on L40 and 7 on L40S (same Ada architecture), with seeds of one protocol mixed; the earlier statement that all 15 used the same GPU type was wrong. The rolling/pre-registered runs (H100) are reported alongside only as a reference column. | Scheduling; decided before any replication result was seen. |
| 2026-09-27 | Disclosures for the replication comparison, found by audit before reporting: (1) our training history is shorter than the other papers' (data start 2003; winter crops need the previous autumn, so from 2004): AgroForecast-DE trains on 2001-2019 (19 years) vs ours 16-17; Srivastava from 1999 (18-20 years) vs ours 13-15 before the 15% holdout; CropFusionNet's start year is not documented. (2) The Srivastava protocol trains on all 10 crops, the random 15% holdout differs per seed, and the historical-average baseline is averaged over the three seeds. (3) The stopping epoch is not recorded by the trainer (`best_epoch` null); best-validation weights are restored. (4) AgroForecast-DE rows are now matched by district code (their own district list), fixing one 2023 winter-wheat row mis-assigned by value matching (their wheat R² 0.180 -> 0.183); `scripts/38` is superseded by `scripts/40`. | Two independent audits of the comparison. |

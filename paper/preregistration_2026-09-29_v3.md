# Pre-registration v3: harvest date, crop-stage weather, district anomalies

**Written:** 29 September 2026, before any v3 feature is computed on the
training data and before any v3 model is trained. Follows v1
(`preregistration_2026-09-26.md`) and v2 (`preregistration_2026-09-28_v2.md`);
their results, the frozen 2026 forecast and their data are not changed.

## 1. Why (root-cause analysis of 2026-09-28)

Retrained on the other papers' splits we were better in 4 of 18 comparisons.
The analysis found:

1. The published models used inputs we left out: observed sowing, flowering
   and **harvest** dates, with weather aligned to the crop's own stages
   (Srivastava 2022), and inputs cut at the observed maturity date
   (CropFusionNet).
2. Our model adds little over the historical average (median skill 0.06-0.07),
   and XGBoost lands in the same place, so the limit is in the inputs rather
   than the architecture.
3. Most of the remaining squared error is in the **district pattern within a
   year** (correlation of within-year deviations 0.36). The inputs are
   normalised globally, so a model mainly sees where a district usually is,
   not how this year differs from its own normal.

v2 showed that observed stage dates from sowing to ripeness, used as on/off
steps, do not help on their own (H4 not confirmed). What v3 adds is (a) the
harvest date and (b) weather accumulated over the crop's observed stages,
which is how the stage dates enter crop models and Srivastava's CNN.

## 2. Inputs (all causal: a value on day d uses only what was known on day d)

### 2.1 Harvest date
The v2 phenology table gets a sixth slot, **harvest** (DWD phase 24; silage
maize: 24 or 39 "silage harvest"), same district rule, same quality rule, same
proxies (triticale = mean of winter wheat and rye; summer barley = oats before
its first observed year). Output `data/processed/static/phenology_obs_v3.parquet`;
the v2 file is not changed. Encoded like the other slots: `ph_harvest` = 0
before the district's observed harvest date, 1 from that date on. The harvest
is observed before the end of our input window in 32-92% of district-years
(by crop), and earlier in drought years. A harvest after the forecast date is
never visible.

### 2.2 Crop-stage weather (running totals, daily)
From daily Tmax, Tmin, precipitation and climatic water balance (CWB):
- since the observed sowing date (window start if none): growing degree days
  (base 0 °C), precipitation, CWB, heat days (Tmax >= 30 °C);
- over grain fill / main growth, from the observed heading or flowering date
  to the observed ripeness or harvest date (whichever comes first), frozen
  after it: days, growing degree days, precipitation, CWB, heat days;
- days since the most recently observed stage.
Each is a running total up to day d, so the final value is only known at the
end of the phase.

### 2.3 District anomalies
For Tmax, Tmin, precipitation, CWB, VPD, root-zone soil moisture, solar
radiation and the four cropland satellite indices: an extra input = the value
minus that district's mean for the same calendar month, with the mean computed
on the **training years only** of each run and stored with the model. Districts
without training data get 0.

## 3. Arms (dataset_v4, crop-year window, common sample, arm D inputs)

| arm | inputs |
|---|---|
| D0 | arm D (v2 checkpoints reused, identical config) |
| H | D0 + phenology incl. harvest (full network) |
| S | H + crop-stage weather |
| A | S + district anomalies |

Unchanged: hyperparameters, `resample_len` 220, 120-min budget, early
stopping, seeds 42/43/44, rolling test years 2016-2025 (train <= Y-2,
validate Y-1, test Y), 3-seed ensemble mean, 10 crops, one model, L40 GPUs.
No national component (v2 H5/H6 not confirmed).

## 4. Hypotheses (same test as v1/v2)

Sign test over 100 crop-years, >= 59/100 better by skill vs the historical
mean, majority in each seed, no crop worse in >= 7 of 10 years:

- **H7** H vs D0: phenology with the harvest date improves the model.
- **H8** S vs H: crop-stage weather improves the model.
- **H9** A vs S: district anomalies improve the model.

Descriptive: within-year district correlation and national swing ratio per
arm; each arm vs v1 arm D.

## 5. After the test

The best arm among D0, H, S and A by the v1 rule (most crop-years beating the
historical mean, ties by median skill) is re-run on the three papers' protocols
exactly as in the v1 change log of 2026-09-27, except that the observed
phenology inputs are now included (causal encoding, §2). The comparison table
is reported as it comes out, with the same disclosures.

Caveat fixed in advance: a model using the harvest date is, for most crops,
an estimate made at harvest, not a forecast weeks ahead. Results at earlier
horizons are reported separately and are not part of H7-H9.

## Change log

| Date | Change | Reason |
|---|---|---|
| 2026-09-29 | Plan written | - |
| 2026-09-29 | **Stage-date year rule corrected before any v3 run** (smoke tests only, 1 epoch, discarded). The v2 rule filed winter-crop dates from August on under the next harvest year for all slots, which put many August harvests (and some ripeness dates) in the wrong year: the harvest step was on from the first day of the season and grain fill counted 0 days. Found because wheat grain fill averaged 11-38 days on real data instead of ~45-55. Now only sowing and emergence move to the next year (`scripts/43 --with-harvest`); `phenology_obs_v3.parquet` rebuilt; v2's table unchanged. | Data check before the runs. |
| 2026-09-29 | **Results (eval job 3398161, 90/90 runs on L40 + 30 reused D0).** H7 H vs D0: 57/100, seeds 50/62/53, winter barley worse 7/10 → NOT confirmed. H8 S vs H: 49/100, seeds 62/43/55, potatoes worse 8/10 → NOT confirmed. H9 A vs S: 57/100, seeds 43/54/55 → NOT confirmed. Beats historical mean: D0 61, H 62, S 64, A 65 (median skill 0.064 / 0.074 / 0.078 / 0.086). Within-year district correlation 0.365 / 0.376 / 0.393 / 0.399; national correlation 0.45 / 0.52 / 0.50 / 0.48. **Not registered, reported as exploratory:** A vs D0 directly = 65/100 better, seeds 57/56/65, no crop worse in >= 7 years; largest gains in 2018-2019 (median skill 0.31→0.41, 0.05→0.31), losses for winter wheat and rye under A. Per §5 the best arm is A (65). | Registered outcome. |
| 2026-09-29 | **§5 result: arm A on the papers' protocols** (jobs 3399907/3399908, 15 runs on L40, `compare_equal_protocol_v3A.csv`). Ours better than theirs in 5/18 (arm D: 4/18); arm A better than arm D in 10/18. New wins: AgroForecast-DE silage maize (R² 0.457 vs 0.398, RMSE 53.8 vs 56.6, MAPE 9.9 vs 10.9), CropFusionNet winter barley (0.637 vs 0.445 on 2022-24; 0.552 vs 0.445 pooled). Lost: AgroForecast-DE winter wheat (0.051 vs 0.183). Srivastava 2018 improved (RMSE 12.0 → 10.3, theirs 8.4) but 2017/2019 worsened. Same districts and true yields as arm D. **Cause of the wheat losses (diagnosed after the result):** the historical-average baseline fits a per-crop linear trend on the training years; with 2000-2002 included the winter-wheat trend changes from 0 to +0.27-0.30 dt/ha/yr (2000-2002 yields 67-74 dt/ha), so the baseline over-predicts wheat by ~3 dt/ha in 2023-24. This also lowers D0 vs v1 arm D. Any fix is a new registered change, not applied here. | Registered step, reported as it came out. |

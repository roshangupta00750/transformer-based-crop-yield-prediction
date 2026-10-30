# Pre-registration v4: trend rule, per-crop fine-tuning

**Written:** 29 September 2026, before any v4 model is trained. Follows v1-v3
(`preregistration_2026-09-26.md`, `_2026-09-28_v2.md`, `_2026-09-29_v3.md`);
their results, the frozen 2026 forecast and their data are not changed.

## 1. Why

v3 arm A (harvest date + crop-stage weather + district anomalies) beats the
historical mean in 65/100 crop-years (v1 arm D: 62) and wins 5/18 paper
comparisons (arm D: 4/18). Two weaknesses were found after the v3 results:

1. **Trend rule.** The baseline carries a per-crop linear yield trend forward
   when the training years show a significant one (p < 0.01 and large enough
   to matter). The test counts every district-year as independent (~4,000
   points for wheat), but a national good or bad year moves all districts
   together, so the real evidence is closer to one point per year (~15-20).
   With 2000-2002 in the training data (wheat 67-74 dt/ha) the wheat slope
   becomes "significant" (0 → +0.27-0.30 dt/ha/yr) and the baseline
   over-predicts 2023-24 wheat by about 3 dt/ha. Found after seeing results;
   the fix below is the statistically correct version of the existing rule,
   not a value chosen to fit the test years.
2. **One model for ten crops.** The model labels a crop's good / average /
   bad year correctly in 52/100 crop-years, but only 2/10 for winter wheat and
   oats (silage maize 10/10). Srivastava 2022 trained one model for wheat only;
   CropFusionNet one per crop.

## 2. Changes

### 2.1 Trend rule (arm T)
Same rule, same thresholds (p < 0.01; slope x horizon > 0.25 x residual SD),
but the regression is fitted on the **yearly mean** residual per crop (one
point per training year) instead of on district-years. Slope, reference year
and use are unchanged. Everything else is arm A.

### 2.2 Per-crop fine-tuning with validation-only selection (arm F)
For each rolling fold (test year Y, seed s): start from arm T's all-crop model
for (Y, s), keep its input scaling, baseline and district climatology, and
for each crop train a copy further on that crop's training years only
(<= Y-2). Candidates:

| candidate | learning rate | dropout |
|---|---|---|
| 0 | no fine-tuning (arm T's model as is) | - |
| 1 | 1e-4 | 0.15 |
| 2 | 1e-4 | 0.30 |
| 3 | 3e-5 | 0.15 |
| 4 | 3e-5 | 0.30 |

Up to 20 epochs each, early stopping (patience 4) and best-epoch weights on
that crop's validation year Y-1. The candidate with the lowest validation loss
on Y-1 is kept for that crop; the test year Y is never used for any choice.
Candidate 0 means fine-tuning can only be chosen when the validation year
supports it. All other settings as arm A.

## 3. Arms (dataset_v4, rolling test years 2016-2025, seeds 42/43/44, L40)

| arm | what |
|---|---|
| A | v3 arm A (checkpoints reused, unchanged) |
| T | A + trend rule on yearly means (30 runs) |
| F | T + per-crop fine-tuning, selected on the validation year (30 jobs x 10 crops) |

## 4. Hypotheses (same test as v1-v3)

Sign test over 100 crop-years, >= 59/100 better by skill vs the historical
mean, majority in each seed, no crop worse in >= 7 of 10 years:

- **H10** T vs A: the corrected trend rule improves the model.
- **H11** F vs T: per-crop fine-tuning improves the model.

Note: skill is measured against each arm's own historical-average baseline,
which changes under the trend rule. H10 is therefore also reported as RMSE in
dt/ha per crop-year (T vs A, same rows), and the trends each rule selects are
listed per crop and fold.

Descriptive: good/average/bad year labels (tercile rule of 2026-09-29, per
crop over the 10 test years), within-year district correlation, national
swing ratio, which candidate each crop picked, and results split by
autumn-sown vs spring-sown crops.

## 5. After the test

The best arm among A, T and F (most crop-years beating the historical mean,
ties by median skill) is re-run on the three papers' protocols as in v3 §5.
For arm F the per-crop fine-tuning is repeated inside each protocol, selected
on that protocol's own validation data (Srivastava: its random 15% holdout).
The table is reported as it comes out.

## Change log

| Date | Change | Reason |
|---|---|---|
| 2026-09-29 | Plan written | - |
| 2026-09-29 | Pre-run check on training yields only (no test years, Destatis 2000-2023, not the modelling sample): slopes selected per rule, dt/ha/yr. Train <= 2014: district-year rule keeps potatoes 1.83, sugar beet 12.70, summer barley 0.59, triticale 0.44, winter barley 0.40, rapeseed 0.52, wheat 0.47; yearly rule keeps sugar beet 12.88 only. Train <= 2023: district-year keeps silage maize -1.26, sugar beet 8.72, summer barley 0.22, winter barley 0.46, rapeseed 0.17, wheat 0.19; yearly keeps sugar beet 8.44, winter barley 0.46. | Record of what the rule does before any run. |
| 2026-09-30 | **Results (eval job 3401681; T 30 runs, F 30 jobs x 10 crops, L40; A reused).** H10 T vs A: 54/100 better by skill, seeds 59/53/52 → NOT confirmed; as pre-specified, RMSE is also reported: T lower in 62/100 crop-years, median RMSE ratio 0.989. H11 F vs T: 50/100, seeds 51/48/47, winter wheat worse 7/10 → NOT confirmed; RMSE lower in 50/100. Beats historical mean: A 65, T 67, F 68 (median skill 0.086 / 0.089 / 0.107). Median skill autumn-sown / spring-sown: A 0.024 / 0.154, T 0.034 / 0.161, F 0.043 / 0.191. Good/average/bad labels right: A 52, T 44, F 49. Mean error 2016-2025 (pred − true, dt/ha): winter wheat A +3.42, T +0.57, F +0.75; triticale +4.07 / +2.15 / +1.79; rapeseed +3.28 / +1.04 / +0.96; winter barley +2.27 / −2.04 / −1.82 (its trend is dropped in the early folds). Fine-tuning choices (300 crop-folds, validation year only): none 100, lr 1e-4/dropout 0.15 54, 1e-4/0.30 95, 3e-5/0.15 16, 3e-5/0.30 35. **Not registered, exploratory:** F vs A 59/100 by skill, seeds 57/52/53, no crop worse in >= 7 years; RMSE lower in 66/100. Per §5 the best arm is F (68). | Registered outcome. |
| 2026-09-30 | §5 implementation, before any protocol run: `scripts/03_train.py` saves the random 15% validation keys (`<run>.valkeys.parquet`); `scripts/47` takes a protocol split (`--tag cmp<TAG> --train-end`, with `--val-years` or the saved keys); `scripts/39` scores per-crop checkpoints (`CMP_ARM=v4F`). Job `scripts/sbatch_compare_v4F.sh`: per task, the arm-T model on the protocol split, then per-crop fine-tuning on that split's own validation data. | Needed for §5. |
| 2026-09-30 | Descriptive, not a §5 result: v3 arm A and v4 arm F **rolling-test** predictions (train <= Y-2, validate Y-1) scored on each paper's rows (AgroForecast-DE matched district-years, CropFusionNet years, Srivastava years; same n as the equal-protocol table), `results/metrics/compare_rolling_v3_v4.csv`. Better than theirs: v3 5/18, v4 12/18; v4 better than v3 in 17/18 (one tie). Training years differ from the papers' splits (more for AgroForecast-DE and CropFusionNet rows, fewer for Srivastava), so this is not the equal-protocol comparison, which §5 reports. | User request while §5 runs. |
| 2026-09-30 | **§5 result: arm F on the papers' protocols** (jobs 3405027/3405028, 15 tasks on L40, `compare_equal_protocol_v4F.csv`; same n as arm D and v3 A). Ours better than theirs in 12/18 (arm D 4/18, v3 A 5/18); v4 F better than v3 A in 17/18. Wins: AgroForecast-DE wheat and silage maize (all 3 metrics each), CropFusionNet winter barley (both rows), Srivastava 2017 and 2018 (RMSE 5.64 vs 6.6, 7.70 vs 8.4). Losses: CropFusionNet wheat and silage maize (4 rows), Srivastava 2019. The historical average is better than v4 F in 2 of the 12 wins (AgroForecast-DE wheat R² 0.303 vs 0.250, RMSE 8.79 vs 9.12, MAPE 11.04 vs 11.33) and in Srivastava 2019. | Registered step, reported as it came out. |
| 2026-09-30 | RCA of the 6 remaining losses (descriptive, v4 F protocol predictions). (1) Forecasts too cautious: within-year district swing 0.23-0.40 of the real one in the losing rows, calibration slope 1.35 (maize) to 1.83 (Srivastava 2019); a stretch fitted on CropFusionNet's validation years 2019-21 lifts 2022-24 maize R² only 0.591 → 0.600, wheat unchanged. (2) CropFusionNet wheat: the national level barely moves (2024 over-forecast +5.5 dt/ha); with the true national level R² would be 0.604 / 0.634 (> 0.597). (3) CropFusionNet maize: baseline off by up to +80 dt/ha (2022) and 2023 predicted in the wrong direction; the true national level alone gives 0.656 / 0.704 (< 0.752). (4) Srivastava 2019: regional drought in Saxony-Anhalt under-predicted by 20-26 dt/ha; worst 5% of district-years carry 35% of squared error; true national level alone gives 6.86 (> 6.4). (5) Paper-side: AgroForecast-DE (operational CropFusionNet) scores R² maize 0.700 on its validation years 2020-22 but 0.361 on unseen 2023-24 (wheat 0.469 → 0.196); CropFusionNet's reported 0.752 / 0.597 pool validation years. | User request. |
| 2026-09-30 | **Srivastava 2022: units and districts (found after the §5 result; the registered table is not changed).** (1) Their Table S2 (271 counties, 5,692 observations = complete 1999-2019 records; mean 6.3 t/ha, SD 0.98, min 2.2, max 9.5) is 0.86-0.87 x our Destatis winter-wheat statistics on every panel of 231-277 districts (mean 7.25-7.30, SD 1.13-1.15, min 2.57, max 11.04): their yields are very likely dry matter (x 0.86, 14% moisture removed), so their RMSE/MAE are ~14% smaller than the same error in our units. Not stated in the paper; inferred. (2) Their county rule (complete record 1999-2019) rebuilt from our data: 267 districts with >= 19/21 years (231 with 21/21; boundary reforms leave gaps in our older data); 210 of our 220 evaluated 2019 districts qualify; Saalekreis and Salzlandkreis (worst 2019 errors) have 14/21 years and are excluded. (3) On their rule and in our units (theirs / 0.86), v4 F: 2017 RMSE 5.57 vs 7.67, MAE 4.31 vs 6.05; 2018 7.33 vs 9.77, 5.99 vs 8.26; 2019 7.10 vs 7.44, 5.54 vs 5.81 (all six better). All districts, our units: 2019 RMSE 7.58 vs 7.44, MAE 5.84 vs 5.81. Files: `results/metrics/srivastava_district_check.csv`, `srivastava_district_list.csv`. | User request. |
| 2026-09-30 | Validation vs actual Destatis yields 2019-2025 (rolling test, each year unseen; `results/metrics/validation_2019_2025*.{parquet,csv}`): district MAPE historical average 13.7%, v3 A 12.9%, v4 F 12.2%; within ±10%: 53.4 / 55.0 / 56.1% of district-years; national good/bad-year direction right 39 (v3) / 44 (v4) of 70 crop-years; v4 worse than the historical average for winter wheat (MAPE 9.5 vs 8.9%, direction right 1/7), rapeseed and rye; clearly better for silage maize (9.7 vs 14.3%), sugar beet, oats, potatoes. | User request. |
| 2026-09-30 | RCA of the 2019-2025 validation (descriptive; `results/metrics/rca_national_*.csv`). (1) Different drivers: national yield anomaly (detrended, 2003-2025) vs June-July rain r = 0.92 maize, 0.81 sugar beet, 0.88 potatoes, but 0.08 wheat, 0.10 rye, 0.06 rapeseed; wheat's strongest links are previous-autumn rain (-0.32) and summer Tmax (-0.32). (2) Cross-crop transfer: v4 F's predicted wheat anomaly (2016-2025) correlates +0.51 with summer rain (actual +0.13) and -0.08 with autumn rain (actual -0.60); rye +0.87 vs +0.30; predicted wheat-maize co-movement +0.43 vs actual +0.14. (3) Limited predictability: best leave-one-year-out R² from <= 3 seasonal weather indicators (selected on the same data, so optimistic): wheat 0.04, rye 0.02, rapeseed -0.05, winter barley 0.45, sugar beet 0.59, potatoes 0.75, maize 0.81. (4) Indicative only (uses test years): keeping v4's district pattern but the historical-average national level lowers MAPE for wheat 9.46 → 8.52 (average 8.90), rye 13.87 → 13.05, rapeseed 11.72 → 10.70, but raises winter barley 9.15 → 10.05; a per-crop gate must be chosen on validation years. | User request. |

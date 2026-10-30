# Pre-registration v5: national gate, separate autumn- and spring-sown models

**Written:** 30 September 2026, before any v5 rule is applied to a rolling fold
and before any v5 model is trained. Follows v1-v4; their results, the frozen
2026 forecast and their data are not changed.

## 1. Why (root-cause analysis of 2026-09-30, v4 change log)

Validated against actual yields 2019-2025, v4 F beats the historical average
overall (district MAPE 12.2% vs 13.7%) but loses to it for winter wheat (9.5 vs
8.9%, national direction right 1/7 years), rye and rapeseed. The analysis found:

1. **Different drivers.** National yield anomaly vs June-July rain: r = 0.92
   maize, 0.81 sugar beet, 0.88 potatoes, but 0.08 wheat, 0.10 rye, 0.06
   rapeseed. Wheat's strongest links are previous-autumn rain (-0.32) and
   summer heat (-0.32).
2. **Cross-crop transfer.** The shared model gives wheat the maize response:
   predicted wheat anomaly vs summer rain +0.51 (actual +0.13), vs autumn rain
   -0.08 (actual -0.60); rye +0.87 vs +0.30.
3. **Limited predictability.** Leave-one-year-out R² of the national anomaly
   from <= 3 seasonal indicators: wheat 0.04, rye 0.02, rapeseed -0.05, winter
   barley 0.45, sugar beet 0.59, potatoes 0.75, maize 0.81.

These numbers used 2003-2025, i.e. also the test years; the rules below are
motivated by them but fixed now, before being applied to any fold.

## 2. Changes

### 2.1 National gate (arm G = v4 F + gate; arm PG = P + gate)
For each rolling fold (test year Y) and crop, using **training years only**
(harvest years <= Y-2 with yields in `dataset_v4`):

- national anomaly per year = mean over districts of (yield / district mean
  over the training years - 1) x 100, linearly detrended over the training years;
- predictors: nine national seasonal indicators (mean over districts of the
  daily values): rain Oct-Dec of the previous year, rain Mar-May, rain Jun-Jul,
  rain 15 Jul-15 Aug, climatic water balance Mar-May and Jun-Jul, mean Tmax
  Jun-Jul, days with Tmax >= 30 °C in Jun-Jul, mean root-zone soil moisture
  Jun-Jul (`data/processed/static/national_indicators_v4.parquet`);
- ridge regression (predictors standardised inside each fit, alpha = 5,
  missing values = training mean), scored by leave-one-year-out R².

If that R² is **below 0.20**, the gate is on for that crop and fold: the test
prediction keeps the model's district pattern but takes the historical-average
national level, i.e. from each district's prediction the mean over districts of
(prediction - historical average) for that crop and year is subtracted.
Otherwise the prediction is unchanged. The test year is never used.

### 2.2 Separate autumn- and spring-sown models (arm P)
Two all-crop models per fold instead of one, both with arm T's configuration:
autumn-sown (winter wheat, winter barley, rye, triticale, winter rapeseed) and
spring-sown (summer barley, oats, silage maize, potatoes, sugar beet). Each is
then fine-tuned per crop exactly as arm F (same five candidates, chosen on the
validation year). The group comes from the crop calendar
(sowing day > harvest day = autumn-sown), as already used by the data pipeline.

## 3. Arms (dataset_v4, rolling test years 2016-2025, seeds 42/43/44, L40)

| arm | what |
|---|---|
| F | v4 F (reused) |
| G | F + national gate (no retraining) |
| P | autumn-/spring-sown models + per-crop fine-tuning (60 + 30 jobs) |
| PG | P + national gate (descriptive) |

## 4. Hypotheses (same test as v1-v4)

Sign test over 100 crop-years, >= 59/100 better by skill vs the historical
mean, majority in each seed, no crop worse in >= 7 of 10 years:

- **H12** G vs F: the national gate improves the model.
- **H13** P vs F: separate autumn- and spring-sown models improve the model.

Descriptive: PG; which crops and folds the gate switches on; validation vs
actual yields 2019-2025 (district MAPE, national direction) per arm and crop,
autumn- vs spring-sown; predicted wheat response to summer and autumn rain.

## 5. After the test

The best arm among F, G, P and PG (most crop-years beating the historical mean,
ties by median skill) is scored on the three papers' protocols as in v4 §5,
with the gate (if part of the arm) computed on each protocol's training years.
The table is reported as it comes out.

## Change log

| Date | Change | Reason |
|---|---|---|
| 2026-09-30 | Plan written | - |
| 2026-09-30 | Gate decisions (training years only, `data/processed/static/gate_decisions_v5.csv`, job 3407551), recorded while the evaluation runs, before any v5 result is seen. Folds (of 10) with the gate on: winter wheat 10, winter rapeseed 10, triticale 10, summer barley 10, winter barley 9, rye 7, oats 7, sugar beet 3, potatoes 0, silage maize 0. Ridge LOYO R² with all nine indicators is lower than the best-3-indicator values of the RCA (e.g. winter barley -0.22 to 0.20 vs 0.45), so winter barley is gated although the (test-year-informed) RCA suggested it should not be; the rule is applied as registered. Operational note: the first gate job waited 9 h on a jammed CPU partition; rerun on the GPU partition (identical script). | Record. |
| 2026-10-01 | **Results (eval job 3406007; P 60 runs + 30 fine-tune jobs on L40; F reused).** H12 G vs F: 29/100 better, seeds 33/30/29, winter barley worse 8/10 → NOT confirmed (gate on in 66 crop-years: better 29, worse 37; off in 34, identical to F). By crop where gated: winter wheat 8 better / 2 worse, rapeseed 5/5, triticale 5/5, oats 4/3, rye 2/5, summer barley 3/7, winter barley 1/8. H13 P vs F: 47/100, seeds 49/53/41, summer barley worse 8/10 → NOT confirmed. Beats historical mean: F 68, G 75, P 66, PG 75 (median skill 0.107 / 0.086 / 0.119 / 0.092; mean skill F 0.108, G 0.105). District MAPE vs actual 2019-2025: F 12.24, G 12.59, P 12.38, PG 12.54 (average 13.66); winter wheat F 9.46, G 8.52, PG 8.41 (average 8.90; G beats the average in 10/10 wheat test years). **§5 selection:** G and PG tie at 75, PG wins on median skill, so the registered rule selects PG, although PG has higher MAPE than F and H12/H13 failed; the count-of-crop-years rule rewards gating mechanically (gated national level = baseline, so skill is more often small and positive). | Registered outcome. |
| 2026-10-01 | **Decision recorded before the §5 run.** Main model after v5 stays v4 F: H12 and H13 not confirmed, so neither gate nor split is adopted. §5 is run as written (PG on the three protocols, gate from each protocol's training years) and reported as it comes out; it does not change the main model. This departs from the v2-v4 practice of carrying the §5 winner forward because the count rule is improper: F shrunk toward the average goes from 68 to 85/100 while macro MSE skill falls 0.182 → 0.004. 2016-25 ensemble, F vs PG: macro MSE skill 0.182 vs 0.143 (Δ −0.039, 90% CI −0.094..+0.029), total-SSE skill 0.371 vs 0.270, MAPE 12.23 vs 12.73; PG ahead only with area weights (0.200 vs 0.185); nothing significant. Count rule replaced from v6 on (`preregistration_2026-10-01_v6.md`). Basis: 26-agent design/verification workflow wf_8e73993e-b7e (scratchpad `wf/`). | Improper selection rule. |

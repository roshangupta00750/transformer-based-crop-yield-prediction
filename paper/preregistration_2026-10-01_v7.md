# Pre-registration v7: recency inside the transformer

**Written:** 1 October 2026, before any v7 model is trained. Follows v6
(`preregistration_2026-10-01_v6.md`), whose scoring (§4), references (§5) and
model-change rule (§6) apply unchanged. The main model stays v4 F until §6 says otherwise.

## 1. Why

On 2016-2025 the only robust gain was recency (v6 §9, F+R: macro MSE skill
+0.064, positive in 10/10 years, every crop), and a plain 5-year district mean
(B5) beats v4 F on winter wheat, rye, triticale and rapeseed. The model's
baseline is the long-term district mean over all training years, and the model
never sees recent district yields, so it cannot express that a district's level
has moved. v7 builds recency into the model instead of post-processing it.

## 2. Change (arm R7)

Everything as v4 arm F (arm-T configuration, per-crop fine-tuning chosen on the
validation year), except:

1. **Recent baseline.** For district d, crop c, year t the target is the
   anomaly from b(d,c,t) = mean of the district's reported yields in t-5..t-1
   (at least 3 years), plus the fold's trend slope x (t - mean of those years);
   with fewer than 3 years, the v4 baseline is used.
2. **Recent-yield inputs.** Four extra daily input channels (constant over the
   season): y(t-1)/m - 1, y(t-2)/m - 1, y(t-3)/m - 1, where m is the mean of
   y(t-5..t-1), and m / mean(all years < t) - 1 (0 where missing).

Both use only reported yields of years < t, and for any test year only years up
to the fold's last non-test year (rolling: Y-1, which is known at harvest of Y;
paper protocols: their last training/validation year). The yield table is
Destatis via `dataset_v4`.

## 3. Evidence and decision

- **Decision test (v6 §6): hindcast test years 2010-2015** (never scored by any
  model; semi-clean as stated in v6 §7). Both v4 F and R7 are trained for
  2010-2015 (train <= Y-2, validate Y-1, seeds 42/43/44). R7 replaces F only if,
  on these 6 years: macro MSE skill vs y_hist higher with year-block bootstrap
  P(Δ > 0) >= 0.95, pooled MAPE lower, and no crop worse in every year.
- **Development (reported, not decisive):** R7 on rolling 2016-2025 against F,
  F+R, y_hist and B5, all v6 §4 metrics.
- **Prospective:** if adopted, R7 joins the v6 §8 forecasts from 2027.

## Change log

| Date | Change | Reason |
|---|---|---|
| 2026-10-01 | Plan written | - |
| 2026-10-02 | **Results (train 3416863, fine-tune 3416864, eval 3416865; 66+66 L40 jobs).** Hindcast 2010-2015 (decision test, 12,926 rows): MAPE R7 11.26 vs F 11.68 (B5 12.34, y_hist 13.15); macro MSE skill vs y_hist R7 0.243 vs F 0.196, Δ +0.047 (90% CI −0.022..+0.102), P(Δ>0) = 0.874 < 0.95; no crop worse in every year → **v6 §6 not met: F stays the main model.** R7 better than F on 8/10 crops in the hindcast (worse: silage maize, rapeseed). Development (not decisive): 2016-2025 MAPE R7 12.34 vs F 12.23, Δ skill −0.037 (P 0.19; triticale −0.21, rapeseed −0.15, wheat −0.12); 2019-2025 MAPE 11.74 vs 12.24, Δ +0.044 (P 0.95). | Registered outcome. |
| 2026-10-02 | Rapeseed diagnosis (descriptive, after v7 results; `results/metrics/rca_rapeseed_indicators.csv`). National anomaly 2004-2025 vs flowering rain (15 Apr-15 May) +0.43, pod-fill heat days (Tmax >= 27 °C, 15 May-30 Jun) −0.41, May-Jun soil moisture +0.37, bare-ground frost −0.35; F's predicted rapeseed anomaly 2010-2025 instead follows autumn sowing rain (+0.56 vs actual −0.13) and July rain (+0.38 vs +0.01) and ignores pod-fill heat (+0.01 vs −0.40). LOYO R² from the spring drivers 0.02 (optimistic). No break after 2014 (mean 36.9 vs 36.6 dt/ha). Component skill vs y_hist: rapeseed national F −0.30 (dev) / 0.01 (hindcast), district 0.14 / −0.11; wheat, rye, triticale national skill is positive in the hindcast (0.46, 0.33, 0.63) and negative in 2016-2025 (−0.82, −0.22, −0.56). | User request. |
| 2026-10-02 | Added before any run: R7 (config `itransformer_v7R` + per-crop fine-tuning, as v7 §2) is scored on the three papers' protocols exactly as v4 §5 (AgroForecast-DE: train <= 2019, validate 2020-22, test 2023-24; CropFusionNet: train <= 2018, validate 2019-21, test 2022-24; Srivastava: train <= Y-1 with random 15% validation, Y = 2017-2019). Yield history for the recency inputs ends at each protocol's last training/validation year. All 18 rows are reported as they come out, next to v4 F's table; descriptive, it does not change the main model. H100 partition (user request). | User request. |
| 2026-10-03 | **Paper-protocol result for R7** (jobs 3422880-3422882, H100; `compare_equal_protocol_v7RF.csv`, same rows as v4 F). Better than the published value in 8/18 rows (v4 F: 12/18); R7 better than v4 F in 10/18. Gains: AgroForecast-DE wheat (R² 0.332 vs v4 0.250, theirs 0.183), CropFusionNet maize R² 0.663 (2022-24) / 0.706 (2019-24) vs v4 0.591 / 0.670 (theirs 0.752), wheat 0.502 / 0.550 (theirs 0.597). Losses: all Srivastava rows (2018 RMSE 10.42 vs v4 7.70, theirs 8.4; 2017 7.21 vs 5.64). Descriptive; main model unchanged. | Registered step, reported as it came out. |
| 2026-10-03 | Audit (user request). No implementation error: all 63 v7 pipelines' yield history ends at the intended year (rolling Y−1; AgroForecast-DE 2022; CropFusionNet 2021; Srivastava Y−1), lag inputs present, all fine-tuned checkpoints present and linked to v7 bases. Cause of the Srivastava losses is a design weakness: after the record wheat years 2013-2016 the 5-year baseline sits too high (2018: actual 68.7, v7 baseline 77.9 vs v4 73.5 dt/ha; baseline RMSE 13.71 vs 10.85) and the model's correction is the same size as v4's (−2.6 vs −2.4), so it does not learn mean reversion. Same pattern in the rolling test (wheat RMSE v4 vs v7: 2016 8.98/11.69, 2017 5.55/7.11, 2018 7.94/10.17). | Audit. |

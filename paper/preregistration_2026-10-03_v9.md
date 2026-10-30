# Pre-registration v9: blended baseline (recency with mean reversion)

**Written:** 3 October 2026, before any v9 model is trained. Scoring, references
and the model-change rule as v6 §4-§6. Main model: v4 F.

## 1. Why (v7 audit, 2026-10-03)

v7 replaced the long-term district baseline by the 5-year mean. After the record
wheat years 2013-2016 that baseline was too high (2018: actual 68.7, v7 baseline
77.9, v4 73.5 dt/ha) and the model did not learn to correct it, so v7 lost every
Srivastava row and wheat 2016-2018 in the rolling test, while helping crops whose
level genuinely shifts. This flaw was found on 2016-2025 and on the paper
protocols, so neither can validate the fix.

## 2. Change (arm B9)

Everything as v7 R7 (recent-yield inputs, per-crop fine-tuning), except the
baseline. For district d, crop c, year t, from reported yields of years < t only:

- LT = mean of all years < t (>= 3), B5 = mean of years t-5..t-1 (>= 3),
  each moved along the fold's trend slope to year t (slope x (t - mean of the years used));
- baseline = w_c x B5 + (1 - w_c) x LT; only LT available -> LT; neither -> v4 baseline;
- w_c = clip( sum (y - LT)(B5 - LT) / sum (B5 - LT)^2 , 0, 1 ), summed over the
  fold's TRAINING crop-years of crop c with both B5 and LT available
  (least-squares weight of the recent level, estimated on training years only).

## 3. Evidence and decision

- **Decision (v6 §6):** hindcast test years 2010-2015 against v4 F: macro MSE
  skill vs y_hist higher with year-block bootstrap P(Δ > 0) >= 0.95, pooled MAPE
  lower, no crop worse in every year. The 2010-2015 v7 results are known, so this
  test is semi-clean (as v6 §7); stated in the paper.
- **Reported, not decisive:** rolling 2016-2025; the three papers' protocols
  (all 18 rows, same procedure as v4 §5); the per-crop weights w_c per fold.
- **Prospective:** if adopted, B9 joins the v6 §8 forecasts from 2027.

Runs on L40 (`standard`), as the v4 F runs it is compared with.

## Change log

| Date | Change | Reason |
|---|---|---|
| 2026-10-03 | Plan written | - |
| 2026-10-03 | Pre-run check on Destatis yields only (no model; indicative, the training sample differs slightly): w_c for folds 2010 / 2016 / 2020 / 2025 — winter wheat 0.76 / 0.95 / 0.43 / 0.55, rye 0.86 / 0.62 / 0.33 / 0.45, rapeseed 1.00 / 0.87 / 0.52 / 0.39, winter barley 0.50 / 1.00 / 0.73 / 0.30, silage maize 0.00 / 0.26 / 0.38 / 0.51, sugar beet 0.16 / 0.70 / 0.37 / 0.50. Weights vary across folds; wheat's is high just before the 2017-2018 drop, so v9 may not fix those years. Applied as registered. | Record before any run. |
| 2026-10-03 | **Results (rolling 3429765/66, eval 3429767; papers 3429768/69; L40).** Hindcast 2010-2015: accuracy B9 88.63% vs v4 F 88.32% (B5 87.66, y_hist 86.85), macro skill Δ +0.039, P(Δ>0) = 0.84 < 0.95 → **not adopted; v4 F stays main.** Development: 2016-2025 87.67 vs 87.77 (Δ −0.032); 2019-2025 88.31 vs 87.76 (Δ +0.051, P 0.97). Papers (same rows): better than published in 8/18 (v4 F 12/18, v7 8/18); best of all versions on AgroForecast-DE (wheat R² 0.372, maize 0.503) and closest yet to CropFusionNet wheat (0.565 / 0.594 vs 0.597); loses all Srivastava rows except a tie in 2019, as anticipated in the pre-run check. | Registered outcome. |

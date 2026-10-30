# Pre-registration v8: rapeseed — flowering index and national level

**Written:** 2 October 2026, before the flowering index is exported or any v8
model is trained. Scoring and references as v6 §4-§5. Main model: v4 F.

## 1. Why (v7 change log, 2026-10-02)

Rapeseed is the only crop where every model is worse than the historical
average or the 5-year mean in every period. Its national yield follows spring
conditions (flowering rain +0.43, pod-fill heat −0.41, May-June soil moisture
+0.37), which v4 F largely ignores while following autumn and July rain. The
national level is not predictable from weather (LOYO R² 0.02), and v4 F's
national-level skill for rapeseed is 0.01 (hindcast 2010-2015) and −0.30
(2016-2025). Yield depends on flower and pod density, which weather does not
measure.

## 2. Changes

### 2.1 Flowering index (arm Y)
New daily input `ndyi_crop` = (green − blue) / (green + blue) from MODIS MCD43A4
(daily NBAR, 500 m, bands 4 and 3, mandatory quality 0 only), averaged over
CORINE 2018 class 211 cropland per district, 1 March - 30 June of each year
2000-2025; outside that window and where missing it is filled like every other
input (training mean), with `ndyi_observed` = 1/0. It is also added to the
district-anomaly inputs (district-month mean from training years only,
v3 §2.3). Everything else as v4 F (arm-T configuration + per-crop fine-tuning).

### 2.2 Rapeseed national level (rule N_r)
For winter rapeseed only, the forecast keeps the model's district pattern and
takes the historical-average national level (v5 §2.1 apply_gate, rapeseed
only). Basis: the model's own national skill for rapeseed is ≤ 0.01 in both
the semi-clean hindcast and the development years. Because this rule was
chosen after seeing both, it is evaluated only prospectively (v6 §8, from 2027)
and reported descriptively before that.

## 3. Decision (arm Y, on the hindcast 2010-2015)

Y replaces F for winter rapeseed if, on 2010-2015 rapeseed district-years:
SSE skill vs y_hist higher with year-block bootstrap P(Δ > 0) >= 0.95 and
MAPE lower; and, over all crops, macro MSE skill not lower by more than 0.01.
Y replaces F for all crops only under the v6 §6 rule. Development years
2016-2025 are reported, not decisive.

## Change log

| Date | Change | Reason |
|---|---|---|
| 2026-10-02 | Plan written | - |
| 2026-10-02 | **Results (train 3419137, fine-tune 3419138, eval 3419139; 48+48 L40 jobs).** Hindcast 2010-2015, rapeseed: MAPE Y 14.87 vs F 15.19 (B5 14.18, y_hist 14.68), SSE skill Y 0.004 vs F −0.035, P(Y better) = 0.83 < 0.95 → **not adopted for rapeseed**; all crops: MAPE 11.83 vs 11.68, macro skill Δ −0.023 (P 0.12). Development (not decisive), rapeseed: 2016-2025 MAPE 11.32 vs F 12.44 (B5 13.27), skill 0.171 vs 0.031, P = 1.00; 2019-2025 10.91 vs 11.72. Rule N_r (descriptive): rapeseed MAPE with the average national level 14.95 (hindcast), 11.68 (2016-2025). | Registered outcome. |

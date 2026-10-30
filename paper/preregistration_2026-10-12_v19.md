# Pre-registration v19: swing calibration and interval calibration for YieldFormer-DE

**Written:** 12 October 2026, before any v19 arm is computed. Scoring and the model-change rule as v6 §4-§6.
Main model: YieldFormer-DE = v18 XE18 (equal-weight ensemble of YieldFormer-DE-i, -T, -P + district error
memory). No retraining: v19 only post-processes the existing rolling forecasts.

## 1. Why (descriptive, measured before this plan)

On XE18's 37,383 rolling forecasts 2010-2025: 54% of the squared log error is district-specific, 25% is the
national crop-year component and 20% the state crop-year component. Removing the national component (oracle)
would raise accuracy from 88.81% to 90.63%; national + state to 92.29%. Across the 160 crop-years, the
regression slope of the forecast's national deviation from normal on the real deviation is **0.42**; in the 42
crop-years with a real national deviation ≥ 10%, the forecast captures about half of it. The model is
systematically too cautious in good and bad years (2025 cereals, 2026 heat).

**Disclosure:** these numbers use all years 2010-2025, including test years, and motivate arm S19. They are
not a result. The calibration factor of S19 is chosen causally (§2).

## 2. Arm S19: national swing calibration

For crop c and test year Y, with district forecast p (= XE18) and normal n (= y_hist):

- a = log(p / n) (forecast anomaly), ā_{c,Y} = mean of a over the districts of crop c in year Y
  (the forecast's national anomaly), r = a − ā (district part).
- Calibrated forecast p' = n · exp(κ_Y · ā_{c,Y} + r): only the national part is scaled; district parts
  are untouched.
- κ_Y is one number for all crops, chosen from {0.5, 0.6, …, 3.0} to minimise pooled MAPE of p' over all
  years 2010…Y−1 (all strictly before Y). Fewer than 3 such years (Y ≤ 2012) → κ_Y = 1 (no change).

Rolling forecasts start in 2010, so in the decision window S19 can differ from XE18 only in 2013-2015.
Accepted in advance, as for v15.

**Decision (v6 §6, hindcast 2010-2015):** S19 replaces XE18 if macro MSE skill vs y_hist is higher with
year-block bootstrap P(Δ > 0) ≥ 0.95, pooled MAPE is lower, and no crop is worse in every year. Reported,
not decisive: 2016-2025, 2010-2025, extreme crop-years (|national deviation| ≥ 10%) vs the rest, κ_Y per
year, the swing slope after calibration, per crop. Multiple testing: this is hindcast decision 13
(Bonferroni level 0.9962), reported.

## 3. Arm C19: interval calibration (reporting rule)

Current 80% interval (scripts/36, v16 §4.1): forecast + per-crop 10th/90th percentiles of past absolute
errors. Realised coverage 2019-2025: 78.0, 71.6, 83.8, 75.1, 80.8, 80.2, 73.9% (misses mostly above).

C19 interval for year Y, built on the forecast that is main after §2 (S19 if adopted, else XE18):
per-crop 10th/90th percentiles of past **log-ratio** errors log(y/p) over 2016…Y−1, each half-width
multiplied by φ_Y, chosen from {1.0, 1.1, …, 2.0} as the smallest value whose pooled coverage on the years
2016…Y−1 (each year's interval rebuilt from its own earlier years) is ≥ 80%. Evaluated 2019-2025.

**Adoption rule:** C19 replaces the current interval for future frozen forecasts if its mean absolute
coverage gap |coverage − 80%| over 2019-2025 is smaller than the current interval's, and its mean coverage
is between 77% and 85%. Also reported: share of misses above vs below, mean interval width.

## 4. What does not change

The frozen 2026-2028 forecasts are never changed. If S19 or C19 is adopted, a new 2026 forecast is frozen
separately (same data end 2026-09-19) and scored next to the existing ones.

## Change log

| Date | Change | Reason |
|---|---|---|
| 2026-10-12 | Plan written | - |
| 2026-10-12 | Implementation detail fixed before running (`scripts/78_eval_v19.py`): in §3 the "earlier years" used to choose φ_Y are 2017…Y−1 (2016 has no earlier year to build its own interval from); if none exist φ = 1. | Clarification before results. |
| 2026-10-12 | **Results (`scripts/78_eval_v19.py`).** **S19:** hindcast 2010-2015 88.79% vs XE18 88.80%, Δ skill +0.006 (90% CI −0.042..+0.033), P 0.635 → **v6 §6 not met; XE18 stays.** 2016-2025 worse: 88.57 vs 88.81 (Δ −0.035, CI −0.090..−0.003, P 0.005). κ_Y 1.3-1.6 for 2013-2016, then 1.0-1.2. Extreme crop-years (|deviation| ≥ 10%) 87.25 vs 87.11; other years 89.22 vs 89.45; swing slope 0.42 → 0.49. **C19:** mean coverage 2019-2025 85.4% (current 77.6%), mean |gap| 5.37 vs 3.73 → **not adopted; current interval stays** (C19 overshoots: φ 1.2-1.4, width 38-43% vs 32-34%). **Diagnosis (descriptive):** forecast vs real national deviation r = 0.66 (sd 6.0% vs 9.4%); same sign in 71% of crop-years, 93% when |real| ≥ 10%. By crop r: potatoes 0.86, silage maize 0.84, sugar beet 0.83, winter barley 0.66, triticale 0.65, oats/summer barley/rye 0.57-0.58, wheat 0.42, rapeseed 0.39. The caution is not a scaling error: for cereals and rapeseed the model does not know the size or direction of the swing well enough, so amplifying it amplifies wrong signals too. The lever is better information on the national/state swing for cereals and rapeseed, not post-hoc calibration. | Registered outcome. |

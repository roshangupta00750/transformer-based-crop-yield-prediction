# Pre-registration v20: crop-stress inputs at the critical stages, and forecasting during the season

**Written:** 12 October 2026, before any v20 feature is computed or model trained. Scoring and the
model-change rule as v6 §4-§6. Main model: YieldFormer-DE = v18 XE18 (equal-weight ensemble of
YieldFormer-DE-i, -T, -P + district error memory).

## 1. Why

v19 showed the remaining learnable error is the national/state crop-year swing (45% of squared log error;
oracle ceiling 92.3% vs 88.8%), and that for cereals and rapeseed the forecast hardly knows that swing
(correlation of forecast vs real national deviation: wheat 0.42, rapeseed 0.39, oats/spring barley/rye
0.57-0.58; root crops and maize 0.83-0.86). Post-hoc scaling failed (v19). The inputs already include
weather totals since sowing and over grain filling (heading → ripeness), but not the classic damage signals
at the critical stages.

## 2. Arm 20: stress inputs (part A)

New daily, causal inputs from the observed DWD development dates (`phenology_obs_v3`, network "full"), the
same weather columns as now (tmax, tmin, precipitation, climatic water balance `cwb`, vapour pressure deficit
`vpd`, root-zone soil moisture `swroot7`). Each is a running value that only uses stages already observed and
weather up to that day; 0 before its window opens.

| Input | Window | Value |
|---|---|---|
| `se_prec`, `se_cwb` | stem elongation: observed stem date → observed heading | running sum |
| `se_heat` | same | days with tmax ≥ 30 °C |
| `se_swr` | same | running mean of `swroot7` |
| `fl_hdd` | flowering: first 14 days after observed heading | Σ max(tmax − 30, 0) |
| `fl_tmax` | same | running maximum of tmax |
| `fl_vpd`, `fl_cwb` | same | running mean of `vpd`; running sum of `cwb` |
| `gf_hdd` | grain filling: heading → ripeness/harvest (as existing `gf_*`) | Σ max(tmax − 30, 0) |
| `gf_vpd`, `gf_swr` | same | running means of `vpd`, `swroot7` |
| `frost_stem` | from observed stem date on | days with tmin ≤ −2 °C |

Root crops without heading dates get 0 for the heading-based inputs (no change for them in expectation).

**Members:** Q20 / T20 / P20 = the v12 Q12 / v17 T17 / v17 P17 configurations + these inputs (everything else
identical: data, folds 2010-2025, train ≤ Y−2 / validate Y−1, seeds 42-44, per-crop fine-tuning, budgets).
**XE20** = mean(Q20, T20, P20) + the v15 §2 error memory on XE20's own past errors (λ causal), exactly as v18.

**Decision (the only one, v6 §6, hindcast 2010-2015):** XE20 replaces XE18 if macro MSE skill vs y_hist is
higher with year-block bootstrap P(Δ > 0) ≥ 0.95, pooled MAPE lower, no crop worse in every year. Hindcast
decision 14 (Bonferroni level 0.9964), reported. Not decisive, reported: 2016-2025; each member vs its v18
counterpart; the national-swing correlation per crop (target: cereals and rapeseed); the 2025 and 2026
cereal/rapeseed misses (2026 against the preliminary national figures only).

## 3. Part B: forecasting during the season (reporting, no decision)

The v18 members' rolling checkpoints predict every year 2010-2025 with all inputs after a cut date set to
missing (`scripts/36` masking: feature → training mean, missing/stale flags → 1, observed → 0): cut dates
**30 June** and **31 July** of the harvest year, plus at harvest (existing). The ensemble mean gets the same
error memory as v18 (the memory uses only earlier years' final yields, so it is known at every cut date).
Reported per cut date and crop: district and national accuracy vs the 5-year and long-term averages, and the
2025 national errors next to the JRC MARS June/July 2025 bulletins (if obtainable). If Part A is adopted, the
same is done for XE20.

## 4. What does not change

Frozen forecasts are never changed. If XE20 is adopted, a separate 2026 forecast is frozen (data end
2026-09-19) and scored next to the existing ones.

## Change log

| Date | Change | Reason |
|---|---|---|
| 2026-10-12 | Plan written | - |
| 2026-10-12 | Implementation: `add_stress_features` in `src/data/dataset.py` (tests `tests/test_v20_stress.py`: windows and causality), config flag `stress_features`, configs `itransformer_v20Q`, `tftnet_v20T`, `patchtst17_v20P`, jobs `scripts/sbatch_v20.sh`, scoring `scripts/79_eval_v20.py`. Smoke test passed (job 3484981; fold 2015, seed 42, 1 epoch, all 12 inputs present; no smoke number is used). Part B in-season jobs 3484982-87 (cut 06-30 and 07-31 for the v18 members). | Operational. |
| 2026-10-12 | **Part B result (in-season, v18 members; reporting).** District accuracy 2010-2025 (37,383 rows): cut 30 June 86.99% (5-yr avg 86.88, long-term 86.39), cut 31 July 88.01%, at harvest 88.81%. Hindcast 2010-2015: 86.88 / 88.14 / 88.80 (5-yr avg 87.65); 2016-2025: 87.09 / 87.90 / 88.81 (86.19). National accuracy 2010-2025: 92.18 / 93.61 / 94.70 (5-yr avg 92.42). **At 30 June the ensemble is no better than the 5-year average; from 31 July it is clearly better (+1.1 district, +1.2 national).** By 31 July winter crops are almost at their harvest accuracy (wheat 91.4 vs 91.7, winter barley 90.3 vs 90.3, rapeseed 87.5 vs 87.5); summer crops still gain later (silage maize 87.6 → 90.3, sugar beet 88.8 → 90.9, potatoes 84.8 → 86.5). 2025 national errors at 31 July: rapeseed −0.5, winter barley −4.5, triticale −6.2, wheat −7.3, rye −7.8, summer barley −14.3, oats −18.8%. Note: the members were trained on full seasons; later days are filled with training means (scripts/36 method), which is crude for early cuts. | Reporting. |
| 2026-10-13 | **Part A result (`scripts/79_eval_v20.py --part A`; all 144+144 tasks and 3 predictions completed, no failures).** **Decision:** XE20 vs XE18, hindcast 2010-2015: 88.82% vs 88.80%, macro skill Δ +0.003 (90% CI −0.023..+0.021), P 0.563, no crop worse every year → **v6 §6 not met; XE18 (YieldFormer-DE v18) stays.** Reported: 2016-2025 88.90 vs 88.81 (P 0.980); 2010-2025 88.87 vs 88.81 (P 0.860). **Members (reported, not decisive):** iTransformer QE20 vs E15 88.24 vs 88.48 (P 0.019, worse) / 88.35 vs 88.38; TFT TE20 vs TE17 88.99 vs 88.76 (P 0.883) / 89.09 vs 88.91 (P 0.910), 2010-2025 P 0.972; PatchTST PE20 vs PE17 88.15 vs 87.81 (Δ +0.040, CI +0.001..+0.067, P 0.957, would pass v6 if it were a decision arm) / 88.02 vs 87.95. **The stress inputs help TFT and PatchTST and hurt the iTransformer.** National-swing correlation XE18 → XE20: rapeseed 0.39 → 0.53, wheat 0.42 → 0.47, winter barley 0.66 → 0.70, triticale 0.65 → 0.69, rye 0.57 → 0.58; oats 0.57 → 0.48, maize/sugar beet/potatoes −0.02..−0.04. 2025 national errors XE18 → XE20: wheat −8.8 → −6.5, triticale −8.0 → −6.4, sugar beet −1.5 → 0.0, maize −2.3 → −1.3; spring barley −15.5 → −16.0, oats −13.3 → −13.6. **Not tested (would be selection on seen test years):** an ensemble of TFT and PatchTST with stress inputs and the iTransformer without; recorded as a hypothesis only. | Registered outcome. |

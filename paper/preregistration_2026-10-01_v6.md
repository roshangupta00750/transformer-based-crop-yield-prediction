# Pre-registration v6: proper scoring, closed development years, clean tests

**Written:** 1 October 2026. This is before the v5 §5 protocol run, before any model is trained for test years 2010-2015 (never used as test years), and before any v4 F prediction for 2026 or 2027 exists. Committed and pushed to the institutional GitLab (tag `prereg-v6`, commit `<hash>`) before any of these runs.

It follows v1-v5. Their results, verdicts, frozen forecasts and data are not changed. v1-v5 were kept as dated local files and first committed on 2026-10-01; this is disclosed.

## 1. Why

1. **v5 outcome (2026-10-01).** H12 and H13 were not confirmed. The §5 count rule selects PG, although PG has higher error than F under every pooled squared-error aggregate except area-weighting, and under MAPE (2016-2025, ensemble: macro MSE skill 0.143 vs 0.182, MAPE 12.73 vs 12.23).
2. **The count rule is not a proper score.** For pred = b + λ(ŷ − b), a crop-year's skill is > 0 exactly when 0 < λ < 2⟨y − b, ŷ − b⟩ / ‖ŷ − b‖², so the count grows as λ → 0. For v4 F: 68/100 at λ = 1, and 85/100 at λ = 0.01 with macro MSE skill 0.004.
3. **2016-2025 is used up.**
   - 21 distinct arms were scored on it. Choosing among them by leave-one-year-out gives 0.159 instead of the in-sample 0.182, an optimism of about 0.023.
   - Differences below about 0.02-0.09 cannot be detected with 10 years.
   - The trend rule, gate, per-crop tuning and the recency adjustment (§9) were all derived from analyses of these years.
4. **The registered reference is weak.**
   - A 5-year district mean beats it by 0.109 macro MSE skill in 2008-2015 (baselines only) and by 0.070 in 2019-2025.
   - v4 F's skill against the 5-year mean is 0.191 over 2016-2025 but 0.052 over 2019-2025 (winter wheat −0.21, rye −0.25, triticale −0.10).

## 2. Main model (fixed)

v4 arm F as registered in v4 §2.2: arm-T all-crop iTransformer plus per-crop fine-tuning chosen on Y−1, seeds 42/43/44, 3-seed mean. Code at commit `<hash>`, trained on julia2 L40 with `~/agri-venv`.

It is the main model because H12 and H13 failed. This departs from the v2-v4 practice of carrying the §5 count-rule winner forward (§1.2).

No gate, split, calibration, ensemble change or new input is added.

## 3. Development years

2016-2025 is development data from now on. This covers the rolling test, the validation of 2019-2025, and the paper protocols' test years 2017-2024.

Every number on these years is reported as development evidence, together with §1.3. No arm, layer, threshold, window or metric is chosen on them after this date.

## 4. Scoring

- **Common rows and reference.** All forecasts in one comparison are scored on the same district-crop-years against the same reference r.
- **Primary: macro MSE skill.** S_r = (1/10) Σ_c [1 − Σ_{d,t∈W} (ŷ − y)² / Σ_{d,t∈W} (r − y)²], pooled over window W for each crop, with crops weighted equally (10 prediction tasks on scales from about 30 to 700 dt/ha).
- **Co-primary.** Pooled district MAPE against actual Destatis yields.
- **Secondary (reported, never decisive):**
  - area-weighted MSE skill, weights proportional to 1000 ha: winter wheat 2900, silage maize 2100, winter barley 1250, winter rapeseed 1050, rye 600, sugar beet 400, triticale 350, summer barley 350, potatoes 265, oats 160;
  - total dt/ha SSE skill;
  - pooled squared relative error skill;
  - per crop, per year and per seed;
  - national component (squared crop-year mean error), within-year component (deviations from it) and within-Land component (errors minus each Land's mean error).
- **Uncertainty.** Year-block bootstrap, B = 10,000, `numpy.random.default_rng(0)`, same draws for paired forecasts; 90% intervals.
- **Descriptive only.** Counts of crop-years with skill > 0, and median crop-year skill.

## 5. References

- **y_hist:** each fold's registered baseline (district mean over years ≤ Y−2, plus the v4 yearly-mean trend rule).
- **B1:** the same rule refitted on all years ≤ Y−1.
- **B5:** district mean of the years Y−5 to Y−1, if at least 3 of them are reported; otherwise y_hist.

B5 (the standard recent-average reference) is the primary reference for statements about added value. y_hist and B1 are secondary. B1, B5 and H+R were already computed on 2008-2025 before this plan.

## 6. Model changes

This replaces the rule "most crop-years beating the historical mean, ties by median skill".

A challenger replaces the main model only if, on at least 3 years not used to design or select it, all three conditions hold:

1. its S_{y_hist} is higher, with bootstrap P(Δ > 0) ≥ 0.95;
2. its pooled MAPE is lower;
3. no crop has higher SSE under the challenger in every assessed year.

Otherwise the main model stays, including in a tie. This applies from v6 on.

The verdicts and selections of v1-v5 stand. A descriptive appendix re-scores them under §4 and §6. Known changes: v2 would select P instead of D0, v5 would select F instead of PG, and H10 would pass.

## 7. Test A: hindcast on never-scored years 2010-2015 (semi-clean)

- **Design.** The v4 F pipeline unchanged, for test years Y = 2010 to 2015: train on years ≤ Y−2 from dataset_v4's first year, validate on Y−1, seeds 42/43/44. That is 18 arm-T runs and 18 per-crop fine-tuning jobs on L40; nothing else is trained. Each is run once; a failed job is rerun with identical settings and logged.
- **Status.**
  - These were never test years for any model, but they were training years for all 2016-2025 folds.
  - The national yield-weather correlations for 2003-2025, and the baseline-only results for B1, B5 and H+R on 2008-2015, were computed before this plan.
  - The folds have 8-13 training years, against 14-23 in development, so lower skill is expected.
- **E1 (estimate).** S_{B5} (primary) and S_{y_hist} of F, pooled over 2010-2015, with 90% intervals, per crop and per fold.
  - **Wording rule for the paper:** "outperforms" only if the interval's lower bound is above 0; "not distinguishable" if the interval covers 0; "underperforms" if the upper bound is below 0.
  - If 2016-2025 behaviour carried over, the chance of "outperforms" would be about 0.75 against y_hist and 0.45 against B5.
- **Descriptive.** National and within-year components; G (F plus the v5 gate computed on each fold's training years); F+R (§9).
- **Consequence.** The result changes wording in the paper only. The model does not change.

## 8. Test B: prospective forecasts

- **Forecast set** (per district and crop): F (main), F+R (challenger), y_hist, B1, B5.
- **Models.**
  - Fold 2026: train ≤ 2024, validate 2025, trained October 2026.
  - Fold 2027: train ≤ 2025, validate 2026, trained once after Destatis publishes the 2026 district yields.
  - Checkpoint SHA-256 values are recorded in `results/forecasts/FROZEN.sha256` before issuing.
- **Issue procedure.**
  - Issue date: 15 April of Y+1 (DWD annual-reporter phenology for Y appears around March), or the last working day before an earlier-announced Destatis district release.
  - Inputs: all data available that day, built by the extended `scripts/35` with its Y−1 reproduction check. Preliminary official yield estimates are never inputs.
  - The CSV and its SHA-256 are committed, tagged and e-mailed to the supervisor that day. No re-issue.
- **2026 is exposed.** Preliminary 2026 yields were published before this plan by:
  - Destatis (first estimate, 24 Sep 2026: cereals 69.2 dt/ha, winter wheat 72.7, potatoes 388.7);
  - the BMLEH Erntebericht 2026;
  - the Länder statistical offices;
  - industry (sugar beet, silage maize).

  2026 is scored and reported descriptively (all components, plus within-Land skill) and is never used for a decision.
- **2027 onward is clean** for all ten crops. The procedure is fixed now, and no choice remains once the season starts.
- **E2 (estimate).** F's S_{B5} and S_{y_hist} per clean year and pooled, with the §7 wording rule. Each year is reported as interim, next to the development range: per-year macro skill of F against B5 in 2016-2025 was −0.32 to +0.60, positive in 6 of 10 years. An inconclusive E2 after three years is likely, and will be reported as inconclusive, not as failure.
- **H14 (decision).** F+R replaces F if §6 is met on the clean years 2027-2029. The verdict is given when 2029 is scored.
  - Development evidence (2016-2025, selected after inspection): Δ S_{y_hist} = +0.064 (90% CI +0.046 to +0.085), positive in 10 of 10 years, MAPE 11.63 vs 12.23.
  - If future years resemble 2016-2025, §6 passes with high probability. If R's gain was a 2016-2025 reporting pattern, it fails.
- **v1 forecasts.** The v1 §7 forecasts (2026 arm D, the 2027 baseline and the staged arm-D forecast) continue as registered.

## 9. Recency adjustment R (exactly as evaluated on 2016-2025)

For crop c and year Y, using only Destatis district yields y_{d,t} for 2003 ≤ t ≤ Y−1:

- w_{d,t} = ln y_{d,t} − mean of ln y over the districts reporting c in year t;
- flat_{d,t} = mean of w_{d,s} over s ≤ t−2;
- x_{d,t} = mean of the available w_{d,t−1}, w_{d,t−2}, w_{d,t−3}, minus flat_{d,t} (0 if none is available);
- λ_c(Y) = max(0, Σ x_{d,t}(w_{d,t} − flat_{d,t}) / Σ x_{d,t}²), summed over all districts and t = 2008 to Y−1;
- c_d = y_hist_d · (exp(λ_c(Y) · x_{d,Y}) − 1), minus its mean over all districts F forecasts for (c, Y);
- F+R = F + c_d. The national mean is unchanged.

## 10. Not done

None of the following is evaluated on 2016-2025:

- a new arm or gate;
- a national calibration (C_N, TNS-FP, PSD, FQ, S);
- an F+P average;
- extra seeds;
- a weather-driver input;
- a metric change.

These, the weather-driver audit and the national/district decomposition are reported as exploratory.

## Change log

| Date | Change | Reason |
|---|---|---|
| 2026-10-01 | Plan written; v1-v6 committed and pushed (tag `prereg-v6`) | - |

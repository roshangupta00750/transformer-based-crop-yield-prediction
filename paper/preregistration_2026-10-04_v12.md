# Pre-registration v12: corrected yields, station-first weather, reliable training series

**Written:** 4 October 2026, before any v12 model is trained. Scoring and the
model-change rule as v6 §4-§6. Main model: v4 F.

## 1. Why (v11, 2026-10-04)

The leave-one-out test found two harmful v10 changes on the hindcast 2010-2015:
replacing DWD station weather by ERA5-Land everywhere (+0.042 macro skill when
undone, P 0.99) and the corrected/added district series (+0.030, P 1.00, measured
on the unchanged districts). Trend reference and MODIS dating had no detectable
effect. The data audit showed that short district-crop series are much noisier
(outlier rate 7.2% for <= 5 reported years vs 1.6% for 20-23 years).

## 2. Arm Q12

- **Data:** `dataset_v5_hybridW` (built for v11): AGS-matched yields, DWD station
  weather where present and ERA5-Land only where a value is missing, district trend
  reference, MODIS end-of-period dating.
- **Reliable training series (training only):** a district-crop series is used for
  training (all-crop model and per-crop fine-tuning) only if it has reported yields
  in at least max(5, 0.6 x n) of the fold's n training years. Counted on the
  training years only (causal); validation and test rows are never filtered.
- Everything else as v4 F (arm-T configuration, per-crop fine-tuning).

## 3. Decision and reporting

- **Decision (v6 §6):** hindcast 2010-2015 on the v10 decision rows (same reported
  yield in both panels) against v4 F.
- **Reported:** the full corrected panel (accuracy vs y_hist and B5), 2016-2025,
  and the three papers' protocols (18 rows).

## Change log

| Date | Change | Reason |
|---|---|---|
| 2026-10-04 | Plan written | - |
| 2026-10-05 | **Results (rolling 3444399/400, eval 3444401; papers 3444402/403; L40).** Decision rows (12,294; same yields in both panels), hindcast 2010-2015: accuracy Q12 88.27% vs v4 F 88.27%, macro skill Δ −0.001 (90% CI −0.025..+0.025, P 0.50), no crop worse every year → v6 §6 not met (tie; incumbent stays under the accuracy rule). 2016-2025: 87.92 vs 87.80 (Δ −0.001); 2019-2025: 88.01 vs 87.81 (Δ +0.003). Full corrected panel 2010-2015: 88.32% on 360 districts (B5 87.65, y_hist 86.89; skill vs B5 +0.084). Papers: 11/18 (v4 F 12/18), now on 279-286 Srivastava districts (theirs 271) and 760-818 CropFusionNet rows. The v10 loss is fully recovered: corrected yields + station-first weather + reliable training series match v4 F's accuracy on the unchanged districts while covering ~80 more districts with correctly matched yields. | Registered outcome. |

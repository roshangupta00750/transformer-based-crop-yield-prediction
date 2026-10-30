# Pre-registration v11: which v10 change hurt (leave-one-out)

**Written:** 4 October 2026, before any v11 dataset or model exists. Follows v10,
whose arm D10 (four data changes at once) was worse than v4 F on the hindcast
2010-2015 (accuracy 87.67% vs 88.27%, macro skill Δ −0.077, P(Δ>0) = 0.003).

## Arms (each = D10 with exactly one change undone; v4 arm T config + per-crop fine-tuning)

| arm | yields | weather | trend reference | MODIS dating |
|---|---|---|---|---|
| D10 (v10) | AGS-matched | ERA5-Land everywhere | district | end of period |
| L-W | AGS-matched | **DWD stations where present, ERA5-Land only where a value is missing** | district | end of period |
| L-T | AGS-matched | ERA5-Land everywhere | **all-crop mean year (v4)** | end of period |
| L-M | AGS-matched | ERA5-Land everywhere | district | **start of period (v4)** |
| L-Y | **name-matched (v4)** | ERA5-Land everywhere | district | end of period |

Test years 2010-2015 only (the window where D10 lost), seeds 42/43/44, L40.

## Reading

Scored on the v10 decision rows (district-crop-years in both panels with the same
reported yield), macro MSE skill vs v4 F's y_hist and MAPE, year-block bootstrap.
A change is called **harmful** if undoing it raises macro skill over D10 with
P(Δ > 0) >= 0.90 and lowers MAPE; **helpful** if undoing it lowers skill with
P(Δ < 0) >= 0.90; otherwise **no detectable effect**. Diagnostic only; no arm is
adopted from this test.

## Change log

| Date | Change | Reason |
|---|---|---|
| 2026-10-04 | Plan written | - |
| 2026-10-04 | **Results (train 3439352, fine-tune 3439353, eval 3439354; 12,294 decision rows, hindcast 2010-2015).** Accuracy: v4 F 88.27, D10 87.67, undo-W 88.02, undo-T 87.69, undo-M 87.71, undo-Y 87.92 (y_hist 86.80). Undoing minus D10, macro skill: ERA5 weather everywhere +0.042 (90% CI +0.015..+0.062, P 0.99) → **harmful**; AGS-matched yields +0.030 (+0.014..+0.049, P 1.00) → **harmful on these rows**; MODIS end dating +0.014 (−0.025..+0.050, P 0.68) → no detectable effect; district trend reference +0.000 (P 0.52) → no detectable effect. The effects add up to the D10 loss (0.117 + 0.042 + 0.030 + 0.014 ≈ 0.203 vs F 0.194). Caveat on yields: the decision rows exclude the districts the correction changed, so this measures the effect of the added/corrected districts on the OTHER districts only. | Registered outcome. |

# Pre-registration v13: crop-specific satellite averaging

**Written:** 5 October 2026, before the crop-specific satellite data is exported or
any v13 model is trained. Scoring and the model-change rule as v6 §4-§6.
Base: v12 arm Q12 (corrected yields, station-first weather, reliable training series).

## 1. Why

Our satellite inputs average MODIS NDVI/EVI/LAI/fPAR over all cropland in a
district (CORINE class 211), so a wheat field's signal is mixed with maize,
grassland-like fodder and other crops. Averaging over the crop's own fields is
the largest data difference to CropFusionNet and CY-Bench, and it extends our
one confirmed pre-registered result (H3, cropland mask over whole district, 67/100).

## 2. Change (arm C13)

1. **Crop masks:** EUCROPMAP (JRC, `JRC/D5/EUCROPMAP/V1`, 10 m), union of the 2018
   and 2022 maps per class, used as a static mask for all years (as CropFusionNet
   uses a multi-year union): wheat 211 -> winter wheat; barley 213 -> winter and
   summer barley; rye 214; oats 215; maize 216 -> silage maize; triticale 218;
   potatoes 221; sugar beet 222; rapeseed 232.
2. **Averaging:** for each MODIS pixel the crop's area fraction (10 m mask averaged
   to the MODIS grid) is the weight of that pixel in the district mean
   (MOD13Q1 NDVI/EVI at 250 m, MOD15A2H LAI/fPAR at 500 m; same scaling, quality
   handling and end-of-period dating as v10).
3. **Use:** for each crop the four satellite inputs `ndvi_crop, evi_crop, lai_crop,
   fpar_crop` are replaced by that crop's own averages. If the crop's mapped area
   in a district is below 0.5% of the district, the cropland average is kept
   (flag `cs_fallback` = 1).
4. Everything else as v12 Q12.

## 3. Decision

C13 replaces Q12 if, on the hindcast 2010-2015 (all corrected-panel rows), v6 §6
holds: macro MSE skill vs y_hist higher with year-block bootstrap P(Δ > 0) >= 0.95,
pooled MAPE lower, no crop worse in every year. Reported: 2016-2025 and the three
papers' protocols. Caveat fixed now: the 2018/2022 masks postdate 2010-2015, so
the hindcast uses future field maps (crop rotation shifts fields; the union of two
years reduces but does not remove this); this is stated with the result.

## Change log

| Date | Change | Reason |
|---|---|---|
| 2026-10-05 | Plan written | - |
| 2026-10-05 | **Data built, before any v13 model is trained.** 53 GEE exports (cropshare, vics/laics 2000-2025), `cs_sat.parquet` 401 districts, 2000-02..2026-01. Crop separation is weak where fields are smaller than a MODIS pixel: in 191 districts with wheat and maize > 10% of area, mean maize−wheat NDVI is −0.02 (May) / +0.03 (late July), r = 0.984; in DE80J (large fields, 2018) −0.15 / +0.17. Fallback share (< 0.5% mapped area): wheat 1%, maize 1%, barley 4%, rapeseed 10%, sugar beet 41%, potato 59%, rye 62%, oats 81%, triticale 93%. Submitted with QoS normal so no other user's job is preempted (train 3454787, fine-tune 3454788, eval 3454789; papers 3454790/91; L40). | Data description; plan unchanged. |
| 2026-10-06 | **Results (train 3454787, fine-tune 3454788, eval 3454789; L40, QoS normal).** Hindcast 2010-2015, 17,718 rows: accuracy C13 88.25% vs Q12 88.32%, macro skill Δ −0.019 (90% CI −0.036..+0.003, P(Δ>0) 0.08), no crop worse every year → **v6 §6 not met; Q12 stays.** 2016-2025: 87.77 vs 87.90 (Δ −0.002, P 0.42); 2019-2025: 87.85 vs 87.94 (Δ +0.007, P 0.66; triticale worse every year). Per crop (hindcast, pp): triticale +0.23, rapeseed +0.12, winter barley +0.01, others −0.03..−0.31. East German districts (large fields, where the masks separate crops) are not better either: 86.08 vs 86.36 (2010-2025). At MODIS 250-500 m the crop-specific series adds noise but little information. | Registered outcome. |
| 2026-10-10 | **Papers' protocols (late; the queued comparison jobs had stalled and were cancelled, all 15 training tasks had finished; eval job 3476259, scored with `scripts/40`).** C13 wins 10 of 18 comparisons (v12 Q12: 11/18). Not decisive; v13 was not adopted. | Registered outcome (reporting). |

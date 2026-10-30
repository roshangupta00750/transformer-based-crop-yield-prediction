# Pre-registration v15: district error memory, area-weighted reporting, area-weighted training

**Written:** 8 October 2026, before arm E15 is computed on any rolling fold and before any
v15 model is trained. Scoring and the model-change rule as v6 §4-§6. Base: v12 arm Q12
(v13 C13 and v14 H14 were not adopted).

## 1. Why (error decomposition of Q12, 2026-10-07, descriptive)

On the 37,383 Q12 rolling predictions 2010-2025 (log error vs the district normal):
national year effect 26% of the squared error, state year effect 20%, district-own 54%.
Better inputs at finer resolution (v13 crop masks, v14 1 km weather) did not help because
weather and satellite inputs are spatially smooth while most of the error is district-own.
Two district-level patterns:

1. **Persistent district error.** A district's error correlates +0.24 with its mean error of
   the previous 1-3 years, and −0.24 with log(5-year average / long-term normal): the model's
   district normal lags real level shifts.
2. **Noise where a crop is rare.** Mean absolute error 13.4% in the quarter of district-crops
   with the smallest crop area share (median 0.1% of the district), 9.5% in the largest
   quarter (median 17%); the 5-year average shows the same gradient (14.3% / 11.2%).

**Disclosure:** before this plan was written, a descriptive check applied a fixed fraction λ of
the previous-3-year mean error to all 2011-2025 predictions: λ 0.25 → accuracy 88.38% vs
88.01% (34,074 rows), λ 0.5 → 88.36%, λ 0.75 → 87.96%. That check used test years to pick λ
and is not a result; it motivates arm E15, whose λ is chosen causally (§2).

## 2. Arm E15: district error memory (no retraining)

For test year Y and each district-crop, with Q12's rolling prediction p_t and reported
yield y_t (both from folds that never saw year t):

- r_t = log(p_t / y_t) for every earlier year t < Y with a prediction and a reported yield;
- m_Y = mean of r_t over t ∈ {Y−3, Y−2, Y−1} that exist (at least one, else m_Y = 0);
- corrected forecast p'_Y = p_Y · exp(−λ_Y · m_Y).

λ_Y is one number for all crops, chosen from {0, 0.05, …, 1.00} to minimise pooled MAPE of the
corrected forecasts over all years 2011…Y−1 (the years where m exists, all strictly before Y).
If fewer than 2 such years exist (Y ≤ 2012), λ_Y = 0, i.e. E15 = Q12. Rolling predictions
start in 2010, so in the decision window E15 can differ from Q12 only in 2013-2015 and uses at
most 3 years of history; this weakens the decision test and is accepted in advance.

**Decision (v6 §6):** E15 replaces Q12 if, on the hindcast 2010-2015 (all corrected-panel rows),
macro MSE skill vs y_hist is higher with year-block bootstrap P(Δ > 0) ≥ 0.95, pooled MAPE is
lower, and no crop is worse in every year. Reported: 2016-2025, 2019-2025, per crop, the chosen
λ_Y per year.

## 3. Area-weighted accuracy (reporting only, never a decision metric)

Weight of a district-crop = mapped crop area = EUCROPMAP 2018∪2022 crop share
(`data/processed/static/cs_share.csv`) × district area (NUTS-3 2021 polygons, EPSG:3035).
Area-weighted accuracy = 100 − Σ w·|error%| / Σ w. Reported next to the unweighted value for
Q12, E15, the 5-year average and the long-term average. Caveat: the crop maps postdate
2010-2015 and are a static proxy for crop area.

## 4. Arm W15: area-weighted training (planned; needs crop-area data)

Training loss of the all-crop model and the per-crop fine-tuning weighted by the district-crop's
crop area, from official crop-area statistics (to be acquired; source and causal timing fixed in
a change-log entry before any W15 model is trained). Same decision rule as §2. Not run before
that entry exists.

## Change log

| Date | Change | Reason |
|---|---|---|
| 2026-10-08 | Plan written | - |
| 2026-10-08 | **E15 result (`scripts/64_eval_v15.py`, Mac, no retraining; inputs = Q12 rolling predictions).** λ_Y chosen causally: 0 for 2010-2012, 0.15/0.15/0.20 for 2013-2015, 0.25-0.35 for 2016-2025. Hindcast 2010-2015, 17,718 rows: MAPE E15 11.52 vs Q12 11.68 (accuracy 88.48% vs 88.32%), macro skill Δ +0.024 (90% CI +0.004..+0.050), P(Δ>0) = 0.984, no crop worse in every year → **v6 §6 met: E15 replaces Q12 as the main model.** Only 2013-2015 can differ (λ = 0 before): 89.75 vs 89.63, 87.01 vs 86.79, 89.43 vs 88.77. Development: 2016-2025 accuracy 88.38 vs 87.90 (Δ skill +0.051, P 0.9995); 2019-2025 88.58 vs 87.94 (Δ +0.075, P 1.00); E15 better in 9 of 10 development years (2016 worse, 88.67 vs 88.99). Per crop 2016-2025 (pp): potatoes +1.18, oats +1.05, silage maize +0.61, triticale +0.51, summer barley +0.49, sugar beet +0.48, winter barley +0.44, rye +0.37, rapeseed +0.19, wheat +0.03; hindcast all crops ≥ 0 except rapeseed −0.03. National accuracy (mean over districts) 94.24 vs 94.32: E15 corrects district levels, not the national year signal. §3 area-weighted accuracy: hindcast E15 90.10, Q12 90.02, B5 88.82, y_hist 88.27; 2016-2025 E15 90.09, Q12 89.76. Caveat (§1): the idea was motivated by a descriptive check on test years; λ itself is causal. | Registered outcome. |
| 2026-10-08 | **§4 fixed before any W15 weight is computed or model trained.** Area source: Duden, Nacke & Offermann (2024), Sci. Data 11:95, OpenAgrar DOI 10.3220/DATA20231117103252-0, `Final_data.csv` (CC-BY 4.0; 2020 district boundaries; all 363 of our districts matched by NUTS-3 id). Crop area (ha) exists only for census years 1999, 2003, 2007, 2010, 2016, 2020. **Causal area:** A(d,c,Y) = value of the latest census year ≤ Y with a reported area for (d,c); none → the crop's 5th-percentile area in that census. **Weight:** w = √A divided by the mean √A over the fold's training rows of the same crop (so every crop keeps its total weight), clipped to [0.25, 4]. √ instead of A because the error gradient between the smallest and largest quarter of crop share is ~1.4× (13.4% vs 9.5%), far flatter than the ~170× area ratio. Weights apply to the training loss of the all-crop model and of the per-crop fine-tuning; validation loss, early stopping and fine-tune candidate selection stay unweighted. **Arm W15** = Q12 config + weights. **Decision (v6 §6, hindcast 2010-2015):** WE15 = W15 with the §2 error memory (λ chosen causally as in §2) replaces E15 (the main model since today). Also reported: W15 vs Q12 without memory, 2016-2025, the papers' protocols. Jobs submitted at the account's default QoS (computervision). | Data now available; rule fixed before use. |
| 2026-10-08 | W15 smoke test passed (job 3468463: area weights on 26,229 training crop-years, mean 1.005). Submitted, all at QoS computervision from submission: train 3468467, fine-tune 3468468, eval 3468469 (`scripts/66_eval_v15W.py`), papers 3468470/3468471 (L40). | Operational. |
| 2026-10-08 | **W15 result (train 3468467, fine-tune 3468468, eval 3468469, papers 3468470/71; no preemptions at QoS computervision).** Decision rows hindcast 2010-2015 (17,718): WE15 88.45% vs E15 88.48%, macro skill Δ −0.002 (90% CI −0.023..+0.014), P(Δ>0) 0.39, oats and summer barley worse in every year → **v6 §6 not met; E15 stays the main model.** Without memory, W15 vs Q12: 88.29 vs 88.32 (Δ −0.003, P 0.35, oats worse every year). Development: WE15 vs E15 2016-2025 88.37 vs 88.38 (Δ +0.012, P 0.94); 2019-2025 88.55 vs 88.58 (Δ +0.014, P 0.89). Per crop hindcast (pp): wheat +0.40, winter barley +0.47, rapeseed +0.17, sugar beet +0.13; summer barley −0.48, silage maize −0.43, oats −0.24. Area weighting moves accuracy from small-area crops (oats, summer barley) to the large ones (wheat, winter barley) with no net gain. Papers' protocols (W15 + per-crop fine-tuning, no memory): 12/18 vs v12 11/18; Srivastava 2018 RMSE 8.26 vs 8.40 published (new win); AgroForecast-DE wheat R² 0.271 (v12 0.234). | Registered outcome. |

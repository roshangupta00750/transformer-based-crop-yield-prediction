# Pre-registration v16: is the transformer needed, how sure are the numbers, forecasts 2026 and 2027

**Written:** 9 October 2026, before any v16 arm is computed, before any gradient-boosting model
is trained, before the cleaning rule is applied, and before any YieldFormer-DE (v15) forecast
for 2026 or 2027 exists. Main model: YieldFormer-DE = v15 arm E15 (v12 Q12 iTransformer +
district error memory). Scoring as v6 §4-§6. Nothing in this plan can change the main model.

## 1. Why

Open questions from the 2026-10-09 review:

1. YieldFormer-DE's gain could come from the trend normal and the error memory alone, without
   the network. Shown nowhere so far.
2. Headline accuracies have no uncertainty ranges; ~a dozen model decisions were taken on the
   same 2010-2015 hindcast (multiple testing).
3. The largest misses are mostly implausible official records (e.g. Köln silage maize 2010,
   45 dt/ha; Cottbus winter wheat 2024, 10.7 dt/ha). **Disclosure:** these were seen in the
   dashboard's "largest misses" list before this plan; the rule in §3.1 is fixed now and applied
   to all rows.
4. 2025 winter wheat: YieldFormer-DE under-estimates Germany by ~15% (already present in Q12).
5. No frozen YieldFormer-DE forecast for a future season exists (the frozen 2026 forecast is
   the v1 arm D model).

## 2. Ablations: does the network add skill? (Q1)

All arms use the same rows as YieldFormer-DE: the 37,383 district-crop-years 2010-2025 of
`results/metrics/v15_predictions.parquet`, each from a fold that never saw its year.

| Arm | Forecast | Network |
|---|---|---|
| **A16a** trend + memory | Q12's own rolling long-term normal y_hist, corrected by the v15 §2 memory rule using y_hist's **own** past errors | none |
| **A16b** 5-year average + memory | the 5-year average B5, corrected the same way with B5's own past errors | none |
| **G16** gradient boosting | XGBoost on the same inputs (below) | none (trees) |
| **GE16** gradient boosting + memory | G16 corrected by the v15 §2 rule with G16's own past errors | none (trees) |

Memory rule exactly as v15 §2 (m_Y = mean log error of Y−3..Y−1 that exist; λ_Y one number,
grid 0..1 step 0.05, chosen on 2011..Y−1 by pooled MAPE; λ = 0 if fewer than 2 fit years).

**G16 specification (fixed, no tuning).** For each test year Y in 2010-2025: load the fitted
Q12 feature pipeline of fold Y, seed 42 (`itransformer_v12Q_roll{Y}_s42.pipeline.json`), so
features, scaling, district climatology anomalies, the target framing (z relative to y_hist)
and y_hist are identical to the network's. Rows: train = years ≤ Y−2 after the v12
reliable-series filter (as Q12 trains), validation = Y−1, test = Y. Tabular inputs: each
past-observed feature averaged over 8 equal consecutive segments of the crop-year sequence
(ordered by date), the static features, and crop as a categorical code. XGBoost 3.3,
`tree_method=hist`, `max_depth=6`, `learning_rate=0.03`, `n_estimators=3000`,
`early_stopping_rounds=100` on the validation year, `subsample=0.8`, `colsample_bytree=0.8`,
`min_child_weight=5`, `reg_lambda=1`, squared error on z. Seeds 42, 43, 44; forecast = mean of
the three. One model for all crops (Q12 also adds per-crop fine-tuning; G16 gets the crop code
instead). Jobs at the account's default QoS (computervision).

**Claim rule.** "The transformer adds skill" is stated only if YieldFormer-DE (E15) beats
**each** of A16a, A16b and GE16 under v6 §6 on the hindcast 2010-2015 (macro MSE skill vs
y_hist higher with year-block bootstrap P(Δ > 0) ≥ 0.95, pooled MAPE lower, no crop worse in
every year). Also reported: 2016-2025, per crop, G16 vs Q12 (both without memory). If an arm
beats YieldFormer-DE, that is reported as the result: the paper then says which part of the
gain comes from the network and which from the normal and the memory. Known asymmetry, stated
in advance: the iTransformer went through 15 design iterations, G16 through none.

## 3. Robustness (Q2-Q4)

### 3.1 Implausible official records (reporting only; training data unchanged)

A reported yield y(d,c,t) is **flagged** if both hold:

1. its log ratio to the median of the same district-crop's reported yields in t−3..t+3
   (excluding t, at least 4 values) exceeds log 2 in absolute value (less than half or more
   than double the district's own neighbouring years), and
2. that log ratio differs from the median log ratio of all other districts of the same state,
   crop and year (computed the same way, at least 3 districts) by more than log 2 (the shock
   is not shared by the district's neighbours, so it is not a regional drought or flood).

The rule uses later years and is a label-quality check, never a forecast input. Reported: the
number of flagged rows, the list, and accuracy of E15, Q12, B5 and y_hist with and without the
flagged rows in every window. Flags do not change any decision taken before today.

### 3.2 Uncertainty ranges

Year-block bootstrap (resample test years with replacement, 2000 draws, seed 0), 90% interval
for: accuracy (100 − MAPE) of E15, Q12, B5, y_hist, and the accuracy differences E15 − B5,
E15 − y_hist, E15 − Q12, in 2010-2015, 2016-2025 and 2010-2025. National accuracy (Germany, mean
over districts) gets the same interval.

### 3.3 Multiple testing (descriptive)

Count every model-change decision taken on the 2010-2015 hindcast (pre-registrations v4-v15).
With k decisions, the Bonferroni level is P ≥ 1 − 0.05/k. Reported for every adopted change,
together with the evidence from 2016-2025, which no decision used.

### 3.4 2025 winter wheat (descriptive diagnosis)

Error of E15, Q12, B5 and y_hist for winter wheat 2025: national, by state, and the national /
state / district split of the log error (as v15 §1); whether the 2025 inputs are complete
(coverage of each input source until harvest); the 2025 weather and satellite anomalies against
the district climatology; and whether earlier years with a similar input pattern were also
under-estimated. No model change follows from §3.4.

## 4. Forecasts with YieldFormer-DE (Q5)

### 4.1 2026

- **Network:** Q12 configuration, fold "2026": train ≤ 2024 (reliable series), validate 2025,
  seeds 42/43/44, then per-crop fine-tuning (`scripts/47`) exactly as the rolling folds.
- **Inputs:** the 2026 crop year built in the dataset_v5_hybridW format (ERA5-Land over
  cropland with DWD station values where present, MODIS end-stamped, drivers and soil as v5).
  **Data end = 2026-09-19, the data end of the frozen v1 2026 forecast**, so both 2026 forecasts
  use the same information period; every later day is set to missing as `scripts/36` does. The
  build is refused if the re-built 2025 rows differ from `dataset_v5_hybridW.parquet` in a
  model input beyond tolerance (as `scripts/35`).
- **Memory:** m_2026 from Q12's rolling errors 2023-2025; λ_2026 chosen on 2011-2025 by the
  v15 §2 rule.
- **Interval:** 80% = forecast + 10th/90th percentile of E15's backtest errors 2016-2025 for the
  same crop (as `scripts/36`).
- **Scope:** district-crop pairs with a district normal (reported yield up to 2024).
- Most 2026 harvests are over; official district yields are not yet published. This is a
  forecast of the official statistic, not of the harvest.

### 4.2 2027

No 2027 season data exist (winter crops are only being sown). Rule fixed now: if **A16a**
(trend + memory) beats y_hist under v6 §6 on the hindcast 2010-2015, the 2027 YieldFormer-DE
forecast is the fold-"2027" normal (feature pipeline fitted on years ≤ 2025, Q12 configuration;
no network) corrected by A16a's memory (errors 2024-2025; 2026 not yet known) with λ_2027 chosen on
2011-2025; otherwise it is the plain normal. In-season updates with the network may be issued
later, each frozen separately; this one is not overwritten.

### 4.3 Freezing and scoring

Each forecast is written once to `results/forecasts/forecast_<Y>_yieldformer_issued_<date>.csv`
with its SHA-256 appended to `results/forecasts/FROZEN.sha256`, next to the v1 forecasts.
Scored with `scripts/37` when Destatis publishes district yields: YieldFormer-DE, v1 arm D, B5
and y_hist side by side.

## Change log

| Date | Change | Reason |
|---|---|---|
| 2026-10-09 | Plan written | - |
| 2026-10-09 | ERA5-Land 2026 cropland weather export submitted before the plan was written (GEE task HI235SZHWB5IIXGWTZC3KIUI; input data only, no forecast or result) | Slowest step started first. |
| 2026-10-09 | Submitted: G16 job 3472091 (16 folds, L40, QoS computervision), fold-2026 network train 3472145 → fine-tune 3472146, normals 3472151. Implementation note: the v6 bootstrap is computed from per crop-year sums with the same random draws (`fast_score` in `scripts/69`); it reproduces `e7.score` exactly on v15's decision (P 0.984, Δ +0.0239). | Operational. |
| 2026-10-09 | **§2 result (`scripts/69_eval_v16.py`; G16 covers all 37,383 rows, y_hist identical to Q12's to 1e-13).** Hindcast 2010-2015, YieldFormer-DE (E15) 88.48% vs: GE16 87.60 (Δ skill +0.106, 90% CI +0.038..+0.144, P 0.988), G16 87.47 (P 0.989), A16a 87.02 (P 1.00), A16b 87.63 (P 0.992), B5 87.65, y_hist 86.89; no crop worse in every year in any comparison → **claim rule met: the transformer adds skill over the normal + memory and over gradient boosting on the same inputs.** Development 2016-2025 (not decisive, reported as registered): **GE16 is better than E15**, 88.60 vs 88.38 (Δ −0.042, CI −0.100..+0.005, P(E15 better) 0.071); G16 87.95 vs Q12 87.90 without memory (P(G16 better) 0.82). A16a 86.62, A16b 86.17. 2010-2025 pooled: E15 88.43 vs GE16 88.12 (P 0.81). Per crop 2016-2025, GE16 ahead for potatoes, rye, silage maize, sugar beet, winter wheat; E15 ahead for oats, summer barley, triticale, rapeseed. λ for GE16 0.25-0.40 (as E15). | Registered outcome. |
| 2026-10-09 | **§3 results.** §3.1: 29 official records flagged 2000-2025, 20 among scored rows; dropping them changes accuracy by +0.05..+0.09 pp for every forecast (E15 2010-2015 88.48 → 88.54, 2016-2025 88.38 → 88.45), ranking unchanged (`v16_flags.csv`, `v16_cleaned.csv`). §3.2 (year-block bootstrap, 90%): E15 accuracy 2010-2015 88.48 [87.46, 89.45], 2016-2025 88.38 [87.73, 88.88]; E15 − B5 +0.83 pp [+0.24, +1.45] and +2.19 [+0.57, +4.30]; E15 − Q12 +0.16 [+0.04, +0.33] and +0.48 [+0.30, +0.64]; national (Germany) E15 − Q12 +0.08 [−0.07, +0.25] and −0.18 [−0.38, +0.01] (`v16_intervals.csv`). §3.3: 9 hindcast decisions (v7-v15), Bonferroni level P ≥ 0.9944; E15's P 0.984 does **not** survive it; its 2016-2025 P 0.9995 does, but its idea came from a descriptive check on 2011-2025 (v15 §1 disclosure). §3.4: 2025 winter wheat, Germany real 77.6 dt/ha: E15 −15.0%, Q12 −14.7%, GE16 −8.0%, G16 −6.5%, B5 −5.6%, y_hist −5.0%; 80% of E15's squared log error is the national component; inputs complete (coverage ≥ 99.7% for every source); spring 2025 was the driest of 2010-2025 (precipitation Mar-Jun z −1.72, water balance z −1.55, top-soil moisture Apr-Jun z −1.36) while crop LAI May-Jun was the second highest (z +1.49). Across 2010-2025 E15's national wheat error correlates +0.41 with the spring water balance z: the network over-weights a dry spring for wheat; trees and averages do not show the 2025 miss (`v16_wheat2025.csv`, `v16_wheat_inputs.csv`, `scripts/72_wheat2025_inputs.py`). | Registered outcome (descriptive). |
| 2026-10-09 | **§4.2 2027 forecast frozen.** A16a vs y_hist on 2010-2015: P 0.855, accuracy 87.02 vs 86.89 → memory OFF; the 2027 YieldFormer-DE forecast is the fold-2027 district normal (Q12 pipeline fitted on ≤ 2025; the same code reproduces the rolling fold-2025 normal to 1e-13). 3,526 district-crop pairs, 80% interval from y_hist's 2016-2025 errors. `results/forecasts/forecast_2027_yieldformer_issued_2026-10-09.csv`, sha256 3497b8c9eeeb84ae85e8cd2f7d9b5b5502607b49c1da16c74f870866deebce9e. | Registered rule applied. |
| 2026-10-09 | §4.1 inputs built (`scripts/70_build_forecast_v5.py`, job 3472212): ERA5-Land 2026 export to 2026-10-01, DWD 2026 stations to 2026-09-25, MODIS to 2026-09-14; 367 districts × 10 crops. Reproduction check on 2025: all 666,125 training rows matched, every model input within tolerance (satellite edge after 2025-12-27 exempt, as `scripts/35`). Registered data end 2026-09-19 applied at prediction. Prediction job 3472220 queued after fine-tuning 3472146. | Operational. |
| 2026-10-09 | **§4.1 2026 forecast frozen.** Fold-2026 network: train 3472145, fine-tune 3472146 (all 3 seeds completed). Prediction job 3472220 failed (phenology/stage flags are int8 and `scripts/36`'s masking writes the training mean into them); fixed by casting those columns to float32 before masking, masking rule unchanged; rerun 3473266 (moved to the idle `test` partition, QoS computervision). Data end 2026-09-19 (equals source coverage). 3,525 district-crop pairs with a fold-2026 normal; λ_2026 = 0.35. Germany vs normal: winter wheat +2.9%, rye +7.5%, winter barley +2.2%, rapeseed +0.2%, triticale −0.1%, summer barley −0.1%, oats −1.5%, sugar beet −5.4%, silage maize −7.0%, potatoes −7.6%. On the 2,483 pairs both 2026 forecasts share, YieldFormer-DE is 0.3% above v1 for wheat and 3-8% below v1 for the other crops except potatoes (+1.0%). `results/forecasts/forecast_2026_yieldformer_issued_2026-10-09.csv`, sha256 42def6161aa824bb1b35988f01fc216c49cd4ccfd0f996115ef97c180a5303ea. | Registered rule applied. |
| 2026-10-10 | **Scope decision by the user:** the study keeps one model family (transformers) plus simple baselines. G16/GE16 stay in this record and in `results/metrics/v16_*` as registered; they are removed from the dashboard and from v17 reporting. The §2 claim rule (E15 beats A16a, A16b and GE16 on 2010-2015) was met before this decision. | User decision; registered results unchanged. |
| 2026-10-10 | Refined: GE16 shown on the dashboard and in v17 reporting as the one labelled non-transformer reference (comparison only); plain G16 stays in this record only. | User decision. |

# Pre-registration v2: longer history, observed phenology, national component

**Written:** 28 September 2026, before the 2000-2025 dataset is merged and before
any model below is trained. Follows `preregistration_2026-09-26.md` (v1; its
results, the frozen 2026 forecast and its data are not changed by this plan).

## 1. Why

The v1 diagnosis (rolling test 2016-2025, arm D) split the error into a national
part (the crop-year mean over districts) and a district part (deviations from it
within the year):

- Predicted national swings are about half the real size (ratio 0.42-0.73 by crop);
  for winter wheat and triticale the national direction is wrong more often than right.
- Oracle: keeping our district pattern but giving the true national level would
  beat the historical mean in 92/100 crop-years instead of 62 (median skill +0.214
  vs +0.061).
- The national signal is learned from only ~16-20 training years per crop.

Three changes target this: (1) more years, (2) observed crop development dates,
(3) an explicit national component.

## 2. Data

### 2.1 History from 2000
Same sources and code as v1, extended back: DWD weather from 1999-07-01,
ERA5-Land drivers and indicators from 1999, MODIS (NDVI/EVI/LAI/fPAR,
cropland-only and whole-district) from 18 Feb 2000, Destatis yields 1999-2002
(table 41241-01-03-4). Harvest years: spring crops from 2000, winter crops
from 2001 (they need the previous autumn). The new file is
`data/processed/dataset_v4.parquet`; `dataset.parquet` (v1) is not modified.
The build is accepted only if its 2003-2025 rows reproduce v1 column by
column (tolerance as `scripts/35`, known exemptions documented).

### 2.2 Observed phenology (DWD, open data)
Two networks: **full** = annual reporters + immediate reporters; **real-time** =
immediate reporters only (published during the season).

Stages used (DWD phase id), one generic slot each:

| crop | sowing | emergence | stem / closed stand | heading / flowering | ripeness |
|---|---|---|---|---|---|
| winter wheat | 10 | 12 | 15 shooting | 18 heading | 21 yellow |
| winter barley | 10 | 12 | 15 | 18 | 21 |
| rye | 10 | 12 | 15 | 18 | 21 |
| winter rapeseed | 10 | 12 | 17 bud formation | 5 flowering | 22 full ripeness |
| summer barley | 10 | 12 | 15 | 18 | 21 |
| oats | 10 | 12 | 15 | 66 panicle | 21 |
| silage maize | 10 | 12 | 67 height growth | 65 tassel | 19 milk ripeness |
| potatoes | 10 | 12 | 13 closed stand | 5 flowering | - |
| sugar beet | 10 | 12 | 13 closed stand | - | - |

Harvest dates (phase 24) are not used. Proxies, fixed now: **triticale** = mean
of the winter wheat and rye district dates per slot (DWD does not observe
triticale); **summer barley** before its first observed year (2014 annual /
2015 real-time) = oats dates.

District date = median of the station dates inside the district (station
coordinates, NUTS-3 2021 polygons); if none, median of the 5 nearest stations
within 75 km of the district centroid; else missing. Quality level and
DWD's own flags are respected (rows flagged as doubtful are dropped).

**Encoding, causal by construction:** for each slot a daily input
`ph_<slot>` = 0 before the district's observed date and 1 from that date on
(plus `ph_<slot>_missing` = 1 where no date exists). A stage observed after the
forecast date is therefore never visible, at any horizon.

### 2.3 National component
For each crop and harvest year: national anomaly = mean over districts of
(yield - historical baseline). Predicted by ridge regression, per crop, on
national season features: the cropland-area-agnostic mean over districts of
the monthly means of each daily input over the crop window (and, in arms
with phenology, the national median of each slot date as day of year).
Trained on the model's training years only; ridge penalty chosen by
leave-one-year-out inside the training years. The iTransformer is trained on
the within-year deviation (anomaly minus that year's national anomaly in the
training data). Prediction = baseline + national prediction + district
prediction.

## 3. Arms (all on dataset_v4, crop-year window, common sample, arm D inputs)

| arm | inputs | target |
|---|---|---|
| D0 | arm D (v1) | anomaly (as v1) |
| P | D0 + phenology (full network) | anomaly |
| N | D0 | national (ridge) + within-year (iTransformer) |
| PN | D0 + phenology (full) | national (ridge with phenology) + within-year |
| PN-rt | D0 + phenology (real-time only) | as PN (descriptive only) |

Unchanged from v1: hyperparameters, `resample_len` 220, 120-min budget, early
stopping, seeds 42/43/44, rolling test years 2016-2025 (train <= Y-2, validate
Y-1, test Y), 3-seed ensemble mean, 10 crops, one model for all crops.

## 4. Hypotheses (decided per crop-year, 100 per comparison)

Same test as v1 (sign test, >= 59/100 better by skill vs the historical mean,
majority in each seed, no crop worse in >= 7 of 10 years):

- **H4** P vs D0: observed phenology improves the model.
- **H5** N vs D0: the national component improves the model.
- **H6** PN vs P: the national component still helps with phenology present.

Descriptive (no hypothesis): D0 vs v1 arm D (effect of 3 more years, same test
years); PN-rt vs PN (cost of using only real-time phenology); national-level
diagnostics (swing ratio, sign right, correlation) per arm; 80% intervals.

## 5. After the test

The best arm by the v1 rule (most crop-years beating the historical mean,
ties by median skill) is re-run on the three papers' protocols exactly as in
the v1 change log of 2026-09-27, and reported as it comes out.

## Change log

| Date | Change | Reason |
|---|---|---|
| 2026-09-28 | Plan written | - |
| 2026-09-28 | Implementation fixed before any run: all 150 v2 runs on one GPU type (L40, `standard` partition), per the v1 audit; phenology inputs are added after the crop window (autumn rows are copied from Y-1 and must get Y's own stages; unit-tested); national target and ridge use the z-scored anomaly (= dt/ha anomaly / crop std, identical up to scale); sugar beet has no real-time phenology file in DWD's standard format, so arm PN-rt leaves it missing; `dataset_v4` is built on julia2 from the same input files that built v1 (md5-checked). | Found while implementing. |
| 2026-09-28 | **Results (job 3389089, 150/150 runs, L40).** H4 P vs D0: 57/100 better, seeds 47/57/57, summer barley worse 7/10 → NOT confirmed. H5 N vs D0: 55/100, seeds 51/63/50, summer barley worse 7/10 → NOT confirmed. H6 PN vs P: 49/100, seeds 57/53/58, summer barley worse 8/10 → NOT confirmed. Beats historical mean: D0 61, P 59, N 60, PN 60, PN-rt 60 (median skill 0.064 / 0.066 / 0.055 / 0.042 / 0.069). Descriptive: D0 vs v1 arm D better in 40/100 crop-years (62 vs 61 beat the mean; median 0.061 vs 0.064), so 3 extra years did not help. National swing ratio rose with the national component (0.58 → 0.72) but sign-right (~6/10) and correlation (0.45 → 0.46) did not, so swings got bigger, not more correct. Per §5 the best arm by the v1 rule is D0 (61), i.e. no change from v1 arm D; the v1 2026 forecast stays as issued. | Registered outcome. |
| 2026-09-29 | **Defect found after the v2 results (disclosed, v2 not re-run).** `scripts/43` filed every winter-crop stage dated August or later under the next harvest year; correct only for sowing/emergence. Measured on the v2 table (harvest years 2001-2025): stem and heading never affected; ripeness filed under the wrong year for 1.4% of winter-wheat, 0.6% rye, 0.4% triticale, 0.4% winter-rapeseed, 0% winter-barley district-years (full network), and for all real-time winter-rapeseed ripeness dates (arm PN-rt only). The ripeness dates they displaced were missing or came from neighbouring stations. H4-H6 used the full network, where the damage is about 1% of one of five slots; the decisions are unlikely to change but were not re-tested. v3 arm H re-tests phenology with the corrected rule. | Found by a data check before the v3 runs. |

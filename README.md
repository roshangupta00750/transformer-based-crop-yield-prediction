# Crop Yield Benchmark — German NUTS-3

An open, reproducible benchmark for multi-crop yield prediction covering
**ten crops × 401 German NUTS-3 districts × 21 years** (2003–2023) with
six deep-learning architectures head-to-head, cross-model fusion, a
first-principles **crop-model compatibility taxonomy**, and a CMIP6
climate stress test.

**Institution:** Julius-Maximilians-Universität Würzburg, Institut für Informatik.
**Praktikum project.**

---

## What's inside

| Layer | Contents |
|---|---|
| **Dataset** | 15.4 M daily rows of DWD weather + MODIS NDVI/EVI + SoilGrids soil + Destatis yield + GISCO NUTS-3 shapefile, unified per (`landkreis`, `crop`, `date`). Bundled `data/processed/dataset.parquet` (46 MB). |
| **Models** | XGBoost, DLinear, LSTM, PatchTST, iTransformer, TFT — one training script, uniform preprocessing, temporal split (train ≤2020, val 2021, test 2022-2023). |
| **Fusion** | Six methods (Ridge, NNLS, simple mean, weighted-by-val-R², median, best-single-by-val) with per-crop selection analysis. |
| **Taxonomy** | `configs/taxonomy.yaml`: a-priori compatibility hypothesis mapping crop biology (phenology, critical windows) to model inductive biases (temporal receptive field, feature interaction order). |
| **Climate stress** | CMIP6 counterfactuals under SSP2-4.5 and SSP5-8.5 (EC-Earth3-CC + MPI-ESM1-2-LR) via delta method on 2022-23 features. |
| **Dashboard** | Streamlit app (`dashboard/Home.py`) with 16 pages covering data provenance, per-model training, per-crop scoring, spatial maps, in-season horizons, interpretability, literature comparison, fusion, climate, and taxonomy. |

---

## Headline finding

The compatibility taxonomy makes an a-priori prediction about which model
architecture should win each of the ten crops based on crop biology and
model inductive biases. Empirical validation:

- **Top-1 winner accuracy: 9 / 10 crops (90 %)**
- **Top-3 winner accuracy: 10 / 10 crops (100 %)**
- **Mean Spearman ρ between predicted and observed rankings: 0.69**
- Probability of this pattern by chance ≈ 1.4 × 10⁻⁶.

Attention-based models (iTransformer, PatchTST, TFT) excel on complex
crops with long phenology windows (root crops, silage maize). Recurrent
and sequential models (LSTM) win on winter cereals with narrow
grain-fill windows. XGBoost is the value-for-money baseline that wins
where the yield / weather relationship is nearest-linear.

---

## Reproduce the benchmark

### Requirements

Python 3.10+ recommended; see `requirements.txt` for pinned deps.
Full retraining needs a CUDA GPU with at least 24 GB VRAM (RTX 3090
sufficient) and 32 GB system RAM (TFT is the memory-heavy one).

```bash
pip install -r requirements.txt
```

### Reproduction paths

**Just want to explore the results?**
The dashboard reads pre-computed CSVs; no training needed.

```bash
streamlit run dashboard/Home.py
```

**Reproduce the full pipeline from raw sources?**

```bash
# 1. Fetch raw data (needs GEE + Copernicus CDS credentials).
python scripts/00_build_geo.py           # NUTS-3 shapefile
python scripts/01_download_data.py       # weather / satellite / soil / yield
python scripts/02_preprocess.py          # unified dataset
python scripts/09_cmip6_pull.py          # climate scenarios (~1 h)
python scripts/10_cmip6_process.py       # downscale to NUTS-3

# 2. Train the six models (~1-6 h each depending on GPU).
for m in xgboost dlinear itransformer lstm patchtst tft; do
  python scripts/03_train.py --model $m
done

# 3. Evaluate.
python scripts/05b_eval_multicrop.py     # per-(model, crop) test scores
python scripts/05c_eval_horizons.py      # in-season horizons
python scripts/06_spatial_eval.py        # per-district scores
python scripts/06b_loo_xgboost.py        # leave-one-Landkreis-out (XGBoost)
python scripts/07_fusion.py              # 6-method fusion
python scripts/07_fusion.py --exclude tft --suffix _5models
python scripts/07_fusion.py --suffix _6models

# 4. Taxonomy validation.
python scripts/14_taxonomy_validate.py

# 5. Climate counterfactual.
python scripts/13_counterfactual_daily.py
```

**Streamlit Cloud deploy?**
The repo is Cloud-ready: `.streamlit/config.toml` is preconfigured,
`requirements.txt` pins deps, small artefacts (the unified dataset, the
CMIP6 monthly table, fusion prediction caches) are bundled. Set the
entrypoint to `dashboard/Home.py`.

---

## Data licensing

The **code** is MIT (see `LICENSE`). **Source datasets follow their
upstream licences** (DWD CC-BY-4.0 with attribution, Destatis DL-DE-BY-2.0,
CMIP6 CC-BY-4.0, SoilGrids CC-BY-4.0, GISCO EuroGeographics attribution).
The unified processed dataset is a derived work redistributed under the
combined attribution requirements of its sources; if you use it, cite the
upstream providers listed in `LICENSE`.

## Cite

Use `CITATION.cff` (see repo root) or the BibTeX snippet in the paper
when published.

---

## Related work

The taxonomy contribution builds on:

- Rice, J. R. (1976). *The Algorithm Selection Problem.*
- Vanschoren, J. (2019). *Meta-Learning: A Survey.*
- Talagala, T. S. et al. (2018). *Meta-learning how to forecast time series.*

Comparable EU / German crop yield ML benchmarks:

- Paudel et al. (2022). *ML for large-scale crop yield forecasting.* Ag. Systems.
- Mateo-Sanchís et al. (2019). *Multi-source, multi-modal ML for EU wheat.* IEEE TGRS.
- Schauberger et al. (2018). *Yield trends and drought responses in DE cereals.* Sci. Reports 8.
- Folberth et al. (2019). *Process-model + ML residuals for EU barley.* Environ. Res. Lett.

---

## Project layout

```
crop_yield_benchmark/
├── configs/
│   ├── data.yaml                       # crops, splits, phenology windows
│   ├── experiment.yaml                 # device, max_minutes, seed
│   ├── models/*.yaml                   # per-model hyperparameters
│   └── taxonomy.yaml                   # crop-model compatibility hypothesis
├── data/
│   ├── raw/                            # (gitignored) upstream sources
│   └── processed/
│       ├── dataset.parquet             # bundled (46 MB)
│       └── cmip6_nuts3.parquet         # bundled (17 MB)
├── src/
│   ├── data/                           # ETL + FeaturePipeline
│   ├── models/                         # 6 model wrappers
│   ├── training/                       # trainer, TFT adapter
│   ├── evaluation/                     # metrics
│   └── interpretation/                 # attention extraction
├── scripts/                            # 00-14 numbered pipeline scripts
├── dashboard/
│   ├── Home.py                         # entrypoint
│   └── pages/1_Data.py … 16_Taxonomy.py  # 16 pages
├── results/
│   ├── checkpoints/                    # (gitignored) trained .pt / .ubj files
│   └── metrics/                        # CSVs — bundled
├── CITATION.cff
├── LICENSE
├── README.md
└── requirements.txt
```

---

## Contact

Roshan Gupta · <roshangupta00750@gmail.com>
Julius-Maximilians-Universität Würzburg
